#!/usr/bin/env python3
"""Fine-tune a RoBERTa model on the LIAR2 fake news dataset with R-Drop regularization and data augmentation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )
except ImportError as exc:  # pragma: no cover - transformers required for training
    raise SystemExit(
        f"transformers is required. Install it with `pip install transformers`. (import error: {exc})"
    ) from exc

try:
    import nltk

    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('wordnet', quiet=True)
except ImportError as exc:  # pragma: no cover - nltk required for nlpaug
    raise SystemExit(
        f"nltk is required for nlpaug. Install it with `pip install nltk`. (import error: {exc})"
    ) from exc

try:
    import nlpaug.augmenter.word as naw
except ImportError as exc:  # pragma: no cover - nlpaug required for augmentation
    raise SystemExit(
        f"nlpaug is required for data augmentation. Install it with `pip install nlpaug`. (import error: {exc})"
    ) from exc

try:
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover - tqdm for progress bars
    raise SystemExit(
        f"tqdm is required for progress tracking. Install it with `pip install tqdm`. (import error: {exc})"
    ) from exc


def _format_numeric_token(value) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "0"
        if float(value).is_integer():
            return str(int(value))
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


class FakeNewsDataset(Dataset):
    """Torch dataset wrapping tokenised LIAR2 samples with optional data augmentation."""

    def __init__(
            self,
            dataframe: pd.DataFrame,
            tokenizer: AutoTokenizer,
            text_columns: List[str],
            numeric_columns: List[str],
            label_column: str,
            max_length: int,
            augment_ratio: float = 0.0,
            augment_columns: List[str] = ["statement", "justification"],
            augment_labels: Optional[List[int]] = None,
            is_training: bool = False,
            cache_dir: Path = None,
    ) -> None:
        self.tokenizer = tokenizer
        missing_cols = [col for col in text_columns if col not in dataframe.columns]
        if missing_cols:
            raise KeyError(f"Missing text columns in dataframe: {missing_cols}")

        # Initialize augmenters
        self.augment_ratio = augment_ratio
        self.augment_columns = augment_columns
        self.augment_labels = augment_labels
        self.is_training = is_training
        self.cache_dir = cache_dir

        if is_training and augment_ratio > 0:
            # Check for cached augmented data
            cache_path = None
            if cache_dir:
                cache_path = cache_dir / "augmented_train.csv"
                if cache_path.exists():
                    print(f"Loading cached augmented data from {cache_path}")
                    dataframe = pd.read_csv(cache_path)
                    print(f"Loaded {len(dataframe)} augmented samples")
                else:
                    print("No cached augmented data found, generating new augmentations")

            if not cache_path or not cache_path.exists():
                # Initialize nlpaug augmenters
                self.synonym_aug = naw.SynonymAug(
                    aug_src="wordnet", aug_p=0.3, stopwords=None
                )
                self.insert_aug = naw.SynonymAug(
                    aug_src="wordnet", aug_p=0.2, stopwords=None
                )
                self.swap_aug = naw.RandomWordAug(
                    action="swap", aug_p=0.2, stopwords=None
                )
                self.delete_aug = naw.RandomWordAug(
                    action="delete", aug_p=0.1, stopwords=None
                )

                # Filter dataframe for augmentation if specific labels are provided
                df_to_augment = dataframe
                if augment_labels is not None:
                    df_to_augment = dataframe[dataframe[label_column].isin(augment_labels)]
                    print(f"Augmenting only samples with labels: {augment_labels}")

                # Prepare augmented data
                augmented_dfs = [dataframe]
                num_augmented = int(len(df_to_augment) * augment_ratio)

                print(f"Generating {num_augmented} augmented samples...")
                for i in tqdm(range(num_augmented), desc="Augmenting data"):
                    aug_df = df_to_augment.iloc[[i % len(df_to_augment)]].copy()
                    for col in augment_columns:
                        if col in text_columns:
                            # Apply augmentations sequentially
                            aug_df[col] = aug_df[col].apply(
                                lambda x: self.synonym_aug.augment(str(x))[0]
                                if pd.notnull(x) else x
                            )
                            aug_df[col] = aug_df[col].apply(
                                lambda x: self.insert_aug.augment(str(x))[0]
                                if pd.notnull(x) else x
                            )
                            aug_df[col] = aug_df[col].apply(
                                lambda x: self.swap_aug.augment(str(x))[0]
                                if pd.notnull(x) else x
                            )
                            aug_df[col] = aug_df[col].apply(
                                lambda x: self.delete_aug.augment(str(x))[0]
                                if pd.notnull(x) else x
                            )
                    augmented_dfs.append(aug_df)

                # Concatenate original and augmented data
                dataframe = pd.concat(augmented_dfs, ignore_index=True)

                # Cache augmented data if cache_dir is provided
                if cache_dir:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    dataframe.to_csv(cache_path, index=False)
                    print(f"Cached augmented data to {cache_path}")

        base_text = (
            dataframe[text_columns]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        numeric_columns = numeric_columns or []
        missing_numeric = [col for col in numeric_columns if col not in dataframe.columns]
        if missing_numeric:
            raise KeyError(f"Missing numeric columns in dataframe: {missing_numeric}")

        if numeric_columns:
            numeric_df = dataframe[numeric_columns].fillna(0)
            numeric_strings = numeric_df.apply(
                lambda row: " ".join(
                    f"{col}:{_format_numeric_token(row[col])}" for col in numeric_columns
                ),
                axis=1,
            )
            combined = (base_text + " " + numeric_strings).str.replace(r"\s+", " ", regex=True)
            self.texts = combined.str.strip().tolist()
        else:
            self.texts = base_text.tolist()
        self.labels = dataframe[label_column].astype(int).tolist()
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        item = {key: torch.tensor(val) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/liar2"),
        help="Directory with train.csv, valid.csv, and test.csv (default: data/liar2)",
    )
    parser.add_argument(
        "--model-name",
        default="roberta-base",
        help="Hugging Face model identifier to use as the backbone if no pretrained model is provided (default: roberta-base)",
    )
    parser.add_argument(
        "--pretrained-model-path",
        type=Path,
        default=None,
        help="Path to a pretrained model directory to load for further training (default: None, uses model-name)",
    )
    parser.add_argument(
        "--text-columns",
        nargs="+",
        default=["statement", "justification", "context", "subject", "speaker_description"],
        help=(
            "Columns to concatenate as the input text. "
            "Default: statement justification context subject speaker_description"
        ),
    )
    parser.add_argument(
        "--numeric-columns",
        nargs="+",
        default=[
            "true_counts",
            "mostly_true_counts",
            "half_true_counts",
            "mostly_false_counts",
            "false_counts",
            "pants_on_fire_counts",
        ],
        help=(
            "Numeric columns to append as textual features (formatted as key:value). "
            "Default: true_counts mostly_true_counts half_true_counts mostly_false_counts false_counts pants_on_fire_counts"
        ),
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Column containing the target label (default: label)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/liar2_roberta"),
        help="Directory to store model checkpoints and logs",
    )
    parser.add_argument(
        "--num-epochs", type=float, default=3.0, help="Training epochs"
    )
    parser.add_argument(
        "--train-batch-size", type=int, default=16, help="Per-device train batch size"
    )
    parser.add_argument(
        "--eval-batch-size", type=int, default=32, help="Per-device eval batch size"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="AdamW learning rate"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay for AdamW"
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Proportion of total steps used for LR warmup (default: 0.1)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length for tokenisation (default: 256)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialisation consistency",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=1,
        help="Updates weights after this many forward passes (default: 1)",
    )
    parser.add_argument(
        "--tensorboard-log-dir",
        type=Path,
        default=None,
        help="Optional TensorBoard log directory. Defaults to <output_dir>/tensorboard if not provided.",
    )
    parser.add_argument(
        "--loss-type",
        choices=["ce", "focal", "rdrop"],
        default="ce",
        help="Loss function to use during training (default: ce).",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Gamma parameter for focal loss (default: 2.0).",
    )
    parser.add_argument(
        "--focal-alpha",
        type=str,
        default=None,
        help=(
            "Class balancing factors for focal loss. Provide comma-separated floats or 'auto' to derive"
            " inverse-frequency weights from the training set."
        ),
    )
    parser.add_argument(
        "--rdrop-alpha",
        type=float,
        default=5.0,
        help="Weight for KL divergence term in R-Drop loss (default: 5.0).",
    )
    parser.add_argument(
        "--augment-ratio",
        type=float,
        default=0.0,
        help="Ratio of augmented samples to generate per original sample (default: 0.0, no augmentation).",
    )
    parser.add_argument(
        "--augment-labels",
        type=int,
        nargs="*",
        default=None,
        help="List of label IDs to augment (default: None, augment all labels).",
    )
    return parser.parse_args()


def load_split(data_dir: Path, filename: str) -> pd.DataFrame:
    path = data_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Could not find required split: {path}")
    return pd.read_csv(path)


def load_label_mapping(model_dir: Path) -> Dict[str, Dict]:
    mapping_path = model_dir / "label_mapping.json"
    if not mapping_path.is_file():
        raise FileNotFoundError(
            f"label_mapping.json not found in {model_dir}. Ensure the pretrained model directory contains label mappings."
        )
    with mapping_path.open("r", encoding="utf-8") as fp:
        raw_mapping = json.load(fp)

    label2id = {str(k): int(v) for k, v in raw_mapping["label2id"].items()}
    id2label = {int(k): str(v) for k, v in raw_mapping["id2label"].items()}
    return {"label2id": label2id, "id2label": id2label}


def build_label_mapping(
        df: pd.DataFrame, label_column: str, existing: Dict = None
) -> Dict[str, Dict[int, str]]:
    if existing is None:
        unique_labels = sorted(df[label_column].dropna().unique())
        string_labels = [str(label) for label in unique_labels]
        label2id = {label: idx for idx, label in enumerate(string_labels)}
        id2label = {idx: label for label, idx in label2id.items()}
    else:
        label2id = existing["label2id"]
        id2label = existing["id2label"]

    missing_labels = {
                         str(label) for label in df[label_column].dropna().unique()
                     } - set(label2id.keys())
    if missing_labels:
        raise ValueError(
            f"Encountered unknown labels {missing_labels} not present in training set or pretrained model."
        )

    df[label_column] = df[label_column].astype(str).map(label2id)
    df[label_column] = df[label_column].astype(int)
    return {"label2id": label2id, "id2label": id2label}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    micro_f1 = f1_score(labels, preds, average="micro")
    return {"accuracy": accuracy, "macro_f1": macro_f1, "micro_f1": micro_f1}


class FakeNewsTrainer(Trainer):
    def __init__(
            self,
            *args,
            loss_type: str = "ce",
            focal_gamma: float = 2.0,
            focal_alpha: torch.Tensor | None = None,
            rdrop_alpha: float = 5.0,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self._focal_alpha = focal_alpha
        self.rdrop_alpha = rdrop_alpha

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        # First forward pass
        outputs1 = model(**inputs)
        logits1 = outputs1.logits

        if self.loss_type == "rdrop":
            # Second forward pass with different dropout
            outputs2 = model(**inputs)
            logits2 = outputs2.logits

            # Classification loss (using cross-entropy for both passes)
            ce_loss1 = F.cross_entropy(logits1, labels, reduction="mean")
            ce_loss2 = F.cross_entropy(logits2, labels, reduction="mean")
            ce_loss = (ce_loss1 + ce_loss2) / 2

            # KL divergence between the two forward passes
            log_probs1 = F.log_softmax(logits1, dim=-1)
            log_probs2 = F.log_softmax(logits2, dim=-1)
            kl_loss = (
                              F.kl_div(log_probs1, log_probs2.softmax(dim=-1), reduction="batchmean") +
                              F.kl_div(log_probs2, log_probs1.softmax(dim=-1), reduction="batchmean")
                      ) / 2
            loss = ce_loss + self.rdrop_alpha * kl_loss
        elif self.loss_type == "focal":
            log_probs = F.log_softmax(logits1, dim=-1)
            probs = log_probs.exp()
            alpha = None
            if self._focal_alpha is not None:
                alpha = self._focal_alpha.to(labels.device)
            ce_loss = F.nll_loss(log_probs, labels, weight=alpha, reduction="none")
            pt = torch.gather(probs, dim=1, index=labels.unsqueeze(1)).squeeze(1)
            focal_factor = (1 - pt).clamp(min=0.0) ** self.focal_gamma
            loss = (focal_factor * ce_loss).mean()
        else:
            loss = F.cross_entropy(logits1, labels)

        return (loss, outputs1) if return_outputs else loss


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = args.data_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split(data_dir, "train.csv")
    valid_df = load_split(data_dir, "valid.csv")
    test_df = load_split(data_dir, "test.csv")

    # Load label mappings, prioritizing pretrained model if provided
    if args.pretrained_model_path:
        if not args.pretrained_model_path.exists():
            raise SystemExit(f"Pretrained model directory does not exist: {args.pretrained_model_path}")
        label_maps = load_label_mapping(args.pretrained_model_path)
    else:
        label_maps = build_label_mapping(train_df, args.label_column)

    build_label_mapping(valid_df, args.label_column, existing=label_maps)
    build_label_mapping(test_df, args.label_column, existing=label_maps)

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_path if args.pretrained_model_path else args.model_name,
        use_fast=True
    )

    train_dataset = FakeNewsDataset(
        train_df,
        tokenizer,
        args.text_columns,
        args.numeric_columns,
        args.label_column,
        args.max_length,
        augment_ratio=args.augment_ratio,
        augment_labels=args.augment_labels,
        is_training=True,
        cache_dir=args.output_dir,
    )
    valid_dataset = FakeNewsDataset(
        valid_df,
        tokenizer,
        args.text_columns,
        args.numeric_columns,
        args.label_column,
        args.max_length,
        augment_ratio=0.0,  # No augmentation for validation
        is_training=False,
    )
    test_dataset = FakeNewsDataset(
        test_df,
        tokenizer,
        args.text_columns,
        args.numeric_columns,
        args.label_column,
        args.max_length,
        augment_ratio=0.0,  # No augmentation for test
        is_training=False,
    )

    num_labels = len(label_maps["label2id"])
    model = AutoModelForSequenceClassification.from_pretrained(
        args.pretrained_model_path if args.pretrained_model_path else args.model_name,
        num_labels=num_labels,
        id2label={idx: label for idx, label in label_maps["id2label"].items()},
        label2id=label_maps["label2id"],
    )

    steps_per_epoch = math.ceil(
        len(train_dataset) / (args.train_batch_size * args.gradient_accumulation)
    )
    total_steps = int(steps_per_epoch * args.num_epochs)
    warmup_steps = int(total_steps * args.warmup_ratio) if args.warmup_ratio > 0 else 0

    focal_alpha_tensor = None
    if args.loss_type == "focal":
        if args.focal_alpha:
            if args.focal_alpha.strip().lower() == "auto":
                counts = train_df[args.label_column].value_counts().sort_index()
                counts = counts.reindex(range(len(label_maps["id2label"])), fill_value=1)
                inv_freq = counts.sum() / counts.clip(lower=1)
                alpha_values = (inv_freq / inv_freq.sum()) * len(inv_freq)
            else:
                alpha_values = [float(x) for x in args.focal_alpha.split(",")]
                if len(alpha_values) != len(label_maps["id2label"]):
                    raise SystemExit(
                        "Number of focal alpha values must match number of labels "
                        f"({len(label_maps['id2label'])})."
                    )
        else:
            alpha_values = None

        if alpha_values is not None:
            focal_alpha_tensor = torch.tensor(alpha_values, dtype=torch.float)

    training_kwargs = {
        "output_dir": str(output_dir),
        "save_strategy": "epoch",
        "num_train_epochs": args.num_epochs,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": warmup_steps,
        "logging_dir": str(output_dir / "logs"),
        "logging_steps": 50,
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "save_total_limit": 2,
        "fp16": torch.cuda.is_available(),
        "report_to": "none",
        "seed": args.seed,
    }

    init_params = TrainingArguments.__init__.__code__.co_varnames
    if "evaluation_strategy" in init_params:
        training_kwargs["evaluation_strategy"] = "epoch"
    else:
        training_kwargs["eval_strategy"] = "epoch"

    training_args = TrainingArguments(**training_kwargs)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = FakeNewsTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        focal_alpha=focal_alpha_tensor,
        rdrop_alpha=args.rdrop_alpha,
    )

    tb_writer = None

    class TensorBoardMetricsCallback(TrainerCallback):
        """Write trainer logs to TensorBoard, including per-epoch summaries."""

        def __init__(self, writer: SummaryWriter) -> None:
            self.writer = writer

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            step = state.global_step
            epoch_value = logs.get("epoch")
            for key, value in logs.items():
                if not isinstance(value, (int, float)):
                    continue
                self.writer.add_scalar(key, value, step)
                if epoch_value is not None and key != "epoch":
                    try:
                        epoch_step = int(epoch_value)
                    except (TypeError, ValueError):
                        epoch_step = float(epoch_value)
                    self.writer.add_scalar(f"{key}/per_epoch", value, epoch_step)

        def on_train_end(self, args, state, control, **kwargs):
            self.writer.flush()
            self.writer.close()

    if SummaryWriter is not None:
        log_dir = args.tensorboard_log_dir or (output_dir / "tensorboard")
        tb_writer = SummaryWriter(log_dir=str(log_dir))
        trainer.add_callback(TensorBoardMetricsCallback(tb_writer))
        print(f"TensorBoard logging enabled at {log_dir}")
    elif args.tensorboard_log_dir is not None:
        print(
            "TensorBoard logging requested but torch.utils.tensorboard is unavailable. "
            f"Install tensorboard to enable logging. (import error: {SUMMARY_WRITER_IMPORT_ERROR})"
        )

    print("Starting training...")
    trainer.train()

    print("Evaluating on validation split...")
    valid_metrics = trainer.evaluate(eval_dataset=valid_dataset)
    print(valid_metrics)

    print("Evaluating on held-out test split...")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    print(test_metrics)

    best_dir = output_dir / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    # Keep a copy of the tokenizer in the root output directory for convenience.
    tokenizer.save_pretrained(output_dir)

    mapping_path = output_dir / "label_mapping.json"
    with mapping_path.open("w", encoding="utf-8") as fp:
        json.dump(label_maps, fp, indent=2)
    print(f"Label mapping saved to {mapping_path}")

    best_mapping_path = best_dir / "label_mapping.json"
    with best_mapping_path.open("w", encoding="utf-8") as fp:
        json.dump(label_maps, fp, indent=2)
    print(f"Best model saved to {best_dir}")


if __name__ == "__main__":
    main()