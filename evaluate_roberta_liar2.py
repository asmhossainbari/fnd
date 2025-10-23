#!/usr/bin/env python3
"""Evaluate a fine-tuned RoBERTa model on the LIAR2 dataset.

The script loads the saved model, tokeniser, and label mapping to compute:
  * Validation accuracy, F1 macro, F1 micro
  * Test accuracy, F1 macro, F1 micro
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - numpy required
    raise SystemExit(
        f"numpy is required for evaluation. Install it with `pip install numpy`. (import error: {exc})"
    ) from exc

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - pandas required
    raise SystemExit(
        f"pandas is required. Install it with `pip install pandas`. (import error: {exc})"
    ) from exc

try:
    import torch
except ImportError as exc:  # pragma: no cover - torch required
    raise SystemExit(
        f"PyTorch is required. Install it with `pip install torch`. (import error: {exc})"
    ) from exc

try:
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
except ImportError as exc:  # pragma: no cover - sklearn required
    raise SystemExit(
        f"scikit-learn is required. Install it with `pip install scikit-learn`. (import error: {exc})"
    ) from exc

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )
except ImportError as exc:  # pragma: no cover - transformers required
    raise SystemExit(
        f"transformers is required. Install it with `pip install transformers`. (import error: {exc})"
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


@dataclass
class EvaluationResult:
    split: str
    accuracy: float
    f1_macro: float
    f1_micro: float
    classification_report: str
    confusion_matrix: List[List[int]]


class FakeNewsDataset(torch.utils.data.Dataset):
    """Dataset wrapping tokenised LIAR2 samples for evaluation."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: AutoTokenizer,
        text_columns: List[str],
        numeric_columns: List[str],
        label_column: str,
        label2id: Dict[str, int],
        max_length: int,
    ) -> None:
        missing_cols = [col for col in text_columns if col not in dataframe.columns]
        if missing_cols:
            raise KeyError(f"Missing text columns in dataframe: {missing_cols}")

        base_text = (
            dataframe[text_columns]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.replace(r"\s+", " ", regex=True)
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
            self.texts = base_text.str.strip().tolist()

        mapped = dataframe[label_column].astype(str).map(label2id)
        if mapped.isnull().any():
            bad_values = dataframe[label_column][mapped.isnull()].unique()
            raise ValueError(f"Found labels not present in label mapping: {bad_values}")
        self.labels = mapped.astype(int).tolist()

        self.tokenizer = tokenizer
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
        help="Directory containing train.csv, valid.csv and test.csv (default: data/liar2)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/liar2_roberta"),
        help="Directory where the fine-tuned model and tokenizer are stored",
    )
    parser.add_argument(
        "--text-columns",
        nargs="+",
        default=["statement", "justification", "context", "subject", "speaker_description"],
        help=(
            "Columns to concatenate into the input text. "
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
        help="Target label column (default: label)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length for tokenisation (default: 256)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Per-device batch size during evaluation (default: 32)",
    )
    return parser.parse_args()


def load_split(data_dir: Path, filename: str) -> pd.DataFrame:
    path = data_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required split not found: {path}")
    return pd.read_csv(path)


def load_label_mapping(model_dir: Path) -> Dict[str, Dict]:
    mapping_path = model_dir / "label_mapping.json"
    if not mapping_path.is_file():
        raise FileNotFoundError(
            f"label_mapping.json not found in {model_dir}. Did you run training first?"
        )
    with mapping_path.open("r", encoding="utf-8") as fp:
        raw_mapping = json.load(fp)

    label2id = {str(k): int(v) for k, v in raw_mapping["label2id"].items()}
    id2label = {int(k): str(v) for k, v in raw_mapping["id2label"].items()}
    return {"label2id": label2id, "id2label": id2label}


def evaluate_split(
    trainer: Trainer,
    dataset: FakeNewsDataset,
    split_name: str,
    label_order: List[int],
    label_names: List[str],
) -> EvaluationResult:
    predictions = trainer.predict(dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids
    accuracy = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_micro = f1_score(labels, preds, average="micro")
    report = classification_report(
        labels,
        preds,
        labels=label_order,
        target_names=label_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=label_order)
    return EvaluationResult(
        split=split_name,
        accuracy=accuracy,
        f1_macro=f1_macro,
        f1_micro=f1_micro,
        classification_report=report,
        confusion_matrix=cm.tolist(),
    )


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir

    if not model_dir.exists():
        raise SystemExit(f"Model checkpoint directory does not exist: {model_dir}")

    mapping = load_label_mapping(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, id2label=mapping["id2label"], label2id=mapping["label2id"]
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    valid_df = load_split(args.data_dir, "valid.csv")
    test_df = load_split(args.data_dir, "test.csv")

    valid_dataset = FakeNewsDataset(
        valid_df,
        tokenizer,
        args.text_columns,
        args.numeric_columns,
        args.label_column,
        mapping["label2id"],
        args.max_length,
    )
    test_dataset = FakeNewsDataset(
        test_df,
        tokenizer,
        args.text_columns,
        args.numeric_columns,
        args.label_column,
        mapping["label2id"],
        args.max_length,
    )

    training_args = TrainingArguments(
        output_dir=str(model_dir / "eval_tmp"),
        per_device_eval_batch_size=args.batch_size,
        do_train=False,
        do_eval=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    label_order = sorted(mapping["id2label"].keys())
    label_names = [mapping["id2label"][idx] for idx in label_order]

    results = [
        evaluate_split(trainer, valid_dataset, "validation", label_order, label_names),
        evaluate_split(trainer, test_dataset, "test", label_order, label_names),
    ]

    for res in results:
        print(f"\n{res.split.capitalize()} metrics:")
        print(f"  Accuracy : {res.accuracy:.4f}")
        print(f"  F1 macro : {res.f1_macro:.4f}")
        print(f"  F1 micro : {res.f1_micro:.4f}")
        print("\nPer-label precision/recall/F1 support:")
        print(res.classification_report)
        print("Confusion matrix (rows true, columns predicted):")
        cm_df = pd.DataFrame(res.confusion_matrix, index=label_names, columns=label_names)
        print(cm_df.to_string())


if __name__ == "__main__":
    main()
