#!/usr/bin/env python3
"""
Train RoBERTa on the TruthSeeker dataset (text + optional numeric features).

This edition applies ONLY the requested fixes:
1) Old-Transformers branch now uses EPOCH evaluation (not STEPS).
2) valid/test are filtered to the same label map as train when a map is provided.
3) steps_per_epoch uses a correct ceil formula with gradient accumulation.

Everything else stays as in your revised script.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import f_classif, mutual_info_classif
from functools import partial
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.trainer_utils import IntervalStrategy  # used for old-HF eval strategy
from transformers import TrainerCallback

# ---------------------------------------------------------------------
# Helpers: support "--numeric-columns topk:K[:method]" and plain lists
# ---------------------------------------------------------------------

_SCORER_MAP = {
    "f": "f_classif",
    "f_classif": "f_classif",
    "mi": "mutual_info",
    "mutual_info": "mutual_info",
}

def parse_numeric_mode(spec: str) -> Tuple[str, Optional[int], Optional[Callable]]:
    """
    Parse --numeric-columns.

    Supported special form:
      "topk:<K>[:<method>]" where method in {f, f_classif, mi, mutual_info}

    Returns (mode, k, scorer_fn_or_None):
      mode in {"topk", "list"}
    """
    if not spec:
        return "list", None, None
    s = str(spec).strip()
    if s.lower().startswith("topk:"):
        parts = s.split(":")
        if len(parts) < 2:
            raise ValueError(f"Invalid spec '{spec}'. Expected 'topk:<K>[:<method>]'.")
        try:
            k = int(parts[1])
        except Exception:
            raise ValueError(f"Invalid K in '{spec}'. Expected integer after 'topk:'.")
        method = parts[2].lower() if len(parts) > 2 else "f"
        method = _SCORER_MAP.get(method, "f_classif")

        def _scorer(X, y):
            if method == "f_classif":
                scores, _ = f_classif(X, y)
                return np.nan_to_num(scores, nan=0.0)
            else:
                return np.nan_to_num(mutual_info_classif(X, y, discrete_features=False), nan=0.0)

        return "topk", k, _scorer
    return "list", None, None


def select_topk_numeric_columns(
    df: pd.DataFrame,
    text_cols: List[str],
    label_col: str,
    k: int,
    scorer: Callable,
) -> List[str]:
    """Select top-k numeric columns by a supervised scorer (ANOVA F or MI)."""
    candidates: List[str] = []
    for c in df.columns:
        if c in text_cols or c == label_col:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            candidates.append(c)
    if not candidates or k <= 0:
        return []

    X = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in candidates}).fillna(0.0).to_numpy(dtype=np.float32)
    y = df[label_col].to_numpy()
    if not np.issubdtype(y.dtype, np.number):
        uniq = {v: i for i, v in enumerate(pd.Series(y).astype(str).unique().tolist())}
        y = np.array([uniq[str(v)] for v in y], dtype=np.int64)

    scores = scorer(X, y)
    order = np.argsort(-scores)[:k]
    return [candidates[i] for i in order]


# ---------------------------
# Dataset
# ---------------------------

class TruthSeekerDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        text_columns: List[str],
        label_column: str,
        label2id: Dict[str, int],
        numeric_columns: Optional[List[str]] = None,
        max_length: int = 256,
        numeric_mean: Optional[Dict[str, float]] = None,
        numeric_std: Optional[Dict[str, float]] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_columns = text_columns
        self.numeric_columns = numeric_columns or []
        self.label_column = label_column
        self.label2id = label2id
        self.max_length = max_length

        if self.label_column not in self.df.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in CSV.")

        texts = self.df[self.text_columns[0]].astype(str)
        for c in self.text_columns[1:]:
            texts = texts + " [SEP] " + self.df[c].astype(str)

        self.enc = self.tokenizer(
            texts.tolist(),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        self.labels = torch.tensor([self.label2id[x] for x in self.df[self.label_column].tolist()], dtype=torch.long)

        if self.numeric_columns:
            num_df = self.df[self.numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            if numeric_mean is not None and numeric_std is not None:
                for c in self.numeric_columns:
                    mu = float(numeric_mean.get(c, 0.0))
                    sd = float(numeric_std.get(c, 1.0)) or 1.0  # protect against 0
                    num_df[c] = (num_df[c] - mu) / sd
                self.numeric = torch.tensor(num_df.values, dtype=torch.float32)
        else:
            self.numeric = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        item["labels"] = self.labels[idx]
        if self.numeric is not None:
            item["numeric"] = self.numeric[idx]
        return item


# ---------------------------
# Model with optional numeric fusion
# ---------------------------

class RobertaWithNumeric(nn.Module):
    def __init__(self, model_name: str, num_labels: int, numeric_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(dropout)
        self.numeric_dim = numeric_dim
        hidden = self.config.hidden_size
        if numeric_dim > 0:
            self.proj = nn.Linear(hidden + numeric_dim, hidden)
        else:
            self.proj = nn.Identity()
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask, labels=None, numeric=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        if numeric is not None and self.numeric_dim > 0:
            if numeric.dim() == 1:
                numeric = numeric.unsqueeze(1)
            fused = torch.cat([pooled, numeric], dim=1)
        else:
            fused = pooled
        fused = self.dropout(fused)
        fused = self.proj(fused)
        logits = self.classifier(fused)
        return SequenceClassifierOutput(logits=logits)


# ---------------------------
# Trainer with CE / R-Drop
# ---------------------------

class CustomTrainer(Trainer):
    def __init__(self, loss_type: str = "ce", rdrop_alpha: float = 2.0, label_smoothing: float = 0.0,
                 class_weights: Optional[torch.Tensor] = None, **kwargs):
        # Keep tokenizer arg for your current script; HF warns it's deprecated but it's fine.
        super().__init__(**kwargs)
        self.loss_type = loss_type
        self.rdrop_alpha = rdrop_alpha
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        numeric = inputs.pop("numeric", None)
        logits = model(numeric=numeric, **inputs).logits

        if self.loss_type == "rdrop":
            logits2 = model(numeric=numeric, **inputs).logits
            loss1 = F.cross_entropy(
                logits.view(-1, self.model.config.num_labels),
                labels.view(-1),
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
            )
            loss2 = F.cross_entropy(
                logits2.view(-1, self.model.config.num_labels),
                labels.view(-1),
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
            )
            p = F.log_softmax(logits, dim=-1)
            q = F.log_softmax(logits2, dim=-1)
            p_target = F.softmax(logits, dim=-1).detach()
            q_target = F.softmax(logits2, dim=-1).detach()
            kl = F.kl_div(p, q_target, reduction="batchmean") + F.kl_div(q, p_target, reduction="batchmean")
            loss = (loss1 + loss2) / 2 + self.rdrop_alpha * kl
        else:
            loss = F.cross_entropy(
                logits.view(-1, self.model.config.num_labels),
                labels.view(-1),
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
            )

        return (loss, logits) if return_outputs else loss


# ---------------------------
# Metrics
# ---------------------------

def compute_metrics_fn(eval_pred):
    """Robust metrics across HF versions; trims to equal length if needed."""
    try:
        logits = getattr(eval_pred, "predictions", None)
        labels = getattr(eval_pred, "label_ids", None)
        if logits is None or labels is None:
            logits, labels = eval_pred  # fallback for older HF
    except Exception:
        logits, labels = eval_pred

    logits = np.array(logits)
    labels = np.array(labels)

    if isinstance(logits, (tuple, list)):
        logits = np.array(logits[0])

    if logits.ndim > 1:
        preds = np.argmax(logits, axis=-1)
    else:
        preds = logits

    n = min(len(labels), len(preds))
    if len(labels) != len(preds):
        labels = labels[:n]
        preds = preds[:n]

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "micro_f1": f1_score(labels, preds, average="micro"),
    }


# ---------------------------
# Pretty eval print callback
# ---------------------------

class PrettyEvalCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        loss  = metrics.get("eval_loss",    metrics.get("loss"))
        acc   = metrics.get("eval_accuracy",metrics.get("accuracy"))
        macro = metrics.get("eval_macro_f1",metrics.get("macro_f1"))
        micro = metrics.get("eval_micro_f1",metrics.get("micro_f1"))
        epoch_num = int(state.epoch) if state.epoch is not None else None
        epoch_str = f"epoch {epoch_num}/{int(getattr(args, 'num_train_epochs', 0))}" if epoch_num else "eval"

        def _fmt(x):
            return f"{x:.6f}" if isinstance(x, (int, float)) else str(x)

        print(f"[{epoch_str}] val_loss={_fmt(loss)}  accuracy={_fmt(acc)}  macro_f1={_fmt(macro)}  micro_f1={_fmt(micro)}")

class SaveBestByAccuracyCallback(TrainerCallback):
    def __init__(self, best_dir: Path, tokenizer, label2id: dict, selected_numeric_columns: list):
        self.best_dir = Path(best_dir)
        self.best_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.selected_numeric_columns = selected_numeric_columns
        self.best_acc = float("-inf")
        self.best_epoch = None

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if not metrics:
            return
        acc = metrics.get("eval_accuracy", metrics.get("accuracy"))
        if acc is None:
            return

        if acc > self.best_acc:
            self.best_acc = float(acc)
            self.best_epoch = int(state.epoch) if state.epoch is not None else None
            self.best_dir.mkdir(parents=True, exist_ok=True)

            # 1) ALWAYS save your custom model’s weights
            torch.save(model.state_dict(), self.best_dir / "state_dict.pt")

            # 2) (Nice to have) save the backbone in HF format for inspection
            try:
                if hasattr(model, "bert") and hasattr(model.bert, "save_pretrained"):
                    (self.best_dir / "backbone").mkdir(exist_ok=True)
                    model.bert.save_pretrained(self.best_dir / "backbone")
            except Exception as e:
                print(f"[best] backbone save_pretrained failed: {e}")

            # 3) Save tokenizer (if available)
            try:
                if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
                    self.tokenizer.save_pretrained(self.best_dir)
            except Exception as e:
                print(f"[best] tokenizer save_pretrained failed: {e}")

            # 4) Save helpers for reproducibility
            with open(self.best_dir / "label_mapping.json", "w") as f:
                json.dump(self.label2id, f, indent=2)
            with open(self.best_dir / "selected_numeric_columns.json", "w") as f:
                json.dump({"selected_numeric_columns": self.selected_numeric_columns}, f, indent=2)
            with open(self.best_dir / "best_metrics.json", "w") as f:
                json.dump({"best_eval_accuracy": self.best_acc, "epoch": self.best_epoch}, f, indent=2)

            print(f"[best] Saved new best model (acc={self.best_acc:.6f}) to: {self.best_dir}")


def _load_state_dict_safely(model: nn.Module, state_path: Path):
    ckpt = torch.load(state_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        ckpt = ckpt["state_dict"]
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f"[init] Loaded state_dict from: {state_path}")
    if missing:
        print(f"[init] Missing keys ({len(missing)}): {sorted(missing)[:12]}{' ...' if len(missing)>12 else ''}")
    if unexpected:
        print(f"[init] Unexpected keys ({len(unexpected)}): {sorted(unexpected)[:12]}{' ...' if len(unexpected)>12 else ''}")

def init_from_path(model: nn.Module, path_like: str):
    """
    Rules:
      - If path contains 'state_dict.pt' -> load into full fused model.
      - Else if dir contains 'state_dict.pt' -> load that.
      - Else treat as HF checkpoint dir and load/replace the backbone.
    """
    if not path_like:
        return
    p = Path(path_like)
    if p.is_file() and p.name.endswith(".pt"):
        _load_state_dict_safely(model, p)
        return
    if p.is_dir():
        sd = p / "state_dict.pt"
        if sd.exists():
            _load_state_dict_safely(model, sd)
            return
        # try HF backbone directory (either this dir, or './backbone')
        hf_dir = p / "backbone" if (p / "backbone").exists() else p
        try:
            hf_backbone = AutoModel.from_pretrained(str(hf_dir), config=model.config)
            model.bert = hf_backbone
            print(f"[init] Initialized backbone from HF dir: {hf_dir}")
        except Exception as e:
            raise ValueError(f"Could not initialize from '{path_like}': {e}") from e
    else:
        raise ValueError(f"Invalid --pretrained-weights path: {path_like}")


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--valid-csv", required=True)
    parser.add_argument("--test-csv",  required=True)

    parser.add_argument("--text-columns", required=True, help="space-separated columns to concatenate as text input")
    parser.add_argument("--numeric-columns", default="", help="space-separated numeric feature columns OR 'topk:K[:f|mi]'")
    parser.add_argument("--raw-label-col", required=True)
    parser.add_argument("--label-map-json", default="", help="optional JSON mapping of label->id; if omitted, derived from train")

    parser.add_argument("--model-name", default="roberta-base")
    parser.add_argument("--output-dir", required=True)

    # Training hyperparams
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)

    # Loss/options
    parser.add_argument("--loss-type", choices=["ce", "rdrop"], default="ce")
    parser.add_argument("--rdrop-alpha", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--use-class-weights", action="store_true")

    # Misc
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=0, help=">0 enables early stopping")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--pretrained", default="", help="Path to init weights. Either a directory with state_dict.pt (your saved best) or a HF checkpoint dir. If set, training starts from these weights.")


    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Read CSVs
    def _read(path):
        df = pd.read_csv(path)
        return df

    train_df = _read(args.train_csv)
    valid_df = _read(args.valid_csv)
    test_df  = _read(args.test_csv)

    # Label mapping
    if args.label_map_json and Path(args.label_map_json).exists():
        with open(args.label_map_json) as f:
            label2id = json.load(f)
        before = len(train_df)
        train_df = train_df[train_df[args.raw_label_col].isin(label2id)].copy()
        dropped = before - len(train_df)
        if dropped:
            print(f"[clean] Dropped {dropped} rows with labels not in mapping: ['nan']")
        # FIX 2: apply same label filtering to valid/test
        valid_df = valid_df[valid_df[args.raw_label_col].isin(label2id)].copy()
        test_df  = test_df[test_df[args.raw_label_col].isin(label2id)].copy()
    else:
        uniques = sorted(train_df[args.raw_label_col].dropna().unique().tolist())
        label2id = {lbl: i for i, lbl in enumerate(uniques)}
    id2label = {v: k for k, v in label2id.items()}

    # Columns
    text_columns = [x for x in args.text_columns.strip().split() if x]

    # Numeric columns
    mode, k, scorer = parse_numeric_mode(args.numeric_columns)
    if mode == "topk":
        numeric_columns = select_topk_numeric_columns(train_df, text_columns, args.raw_label_col, k, scorer)
    else:
        numeric_columns = [x for x in args.numeric_columns.strip().split() if x]

    # Save selected numeric columns for downstream use
    with open(out_dir / "selected_numeric_columns.json", "w") as f:
        json.dump({"selected_numeric_columns": numeric_columns}, f, indent=2)

    # ---- Standardize numeric features (train statistics) ----
    numeric_mean = None
    numeric_std = None
    if numeric_columns:
        train_num_df = train_df[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        # population std (ddof=0) is fine; protect zeros by replacing with 1.0
        mean_s = train_num_df.mean()
        std_s = train_num_df.std(ddof=0).replace(0.0, 1.0)
        numeric_mean = {c: float(mean_s[c]) for c in numeric_columns}
        numeric_std = {c: float(std_s[c]) for c in numeric_columns}
        # Persist stats for evaluation scripts/inference

        with open(out_dir / "numeric_standardization.json", "w") as f:
            json.dump(
                {"columns": numeric_columns, "mean": numeric_mean, "std": numeric_std},
                f, indent=2
            )
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # Datasets
    train_ds = TruthSeekerDataset(train_df, tokenizer, text_columns, args.raw_label_col, label2id, numeric_columns, max_length=args.max_length, numeric_mean=numeric_mean, numeric_std=numeric_std)
    valid_ds = TruthSeekerDataset(valid_df, tokenizer, text_columns, args.raw_label_col, label2id, numeric_columns, max_length=args.max_length, numeric_mean=numeric_mean, numeric_std=numeric_std)
    test_ds  = TruthSeekerDataset(test_df,  tokenizer, text_columns, args.raw_label_col, label2id, numeric_columns, max_length=args.max_length, numeric_mean=numeric_mean, numeric_std=numeric_std)

    num_labels = len(label2id)
    numeric_dim = train_ds.numeric.shape[1] if train_ds.numeric is not None else 0

    # Class weights
    if args.use_class_weights:
        counts = np.bincount(train_ds.labels.numpy(), minlength=num_labels)
        weights = 1.0 / np.maximum(counts, 1)
        weights = weights / weights.sum() * num_labels
        class_weights = torch.tensor(weights, dtype=torch.float32)
        print(f"[preflight] class_weights = {class_weights.tolist()}")
    else:
        class_weights = None

    print(f"[preflight] num_train_examples = {len(train_ds)}")
    print(f"[preflight] num_valid_examples = {len(valid_ds)}")
    print(f"[preflight] num_test_examples  = {len(test_ds)}")
    print(f"[preflight] num_labels        = {num_labels}")
    print(f"[preflight] numeric_dim       = {numeric_dim}")

    # Model
    model = RobertaWithNumeric(args.model_name, num_labels=num_labels, numeric_dim=numeric_dim)
    if args.gradient_checkpointing and hasattr(model.bert, "gradient_checkpointing_enable"):
        model.bert.gradient_checkpointing_enable()

        # NEW: Initialize from user-provided pretrained weights (if any)
    if args.pretrained:
        init_from_path(model, args.pretrained)
    # --------- TrainingArguments (version-robust) ----------
    ta = TrainingArguments(output_dir=str(out_dir))  # only required arg; set rest conditionally

    def _set(name, value):
        if hasattr(ta, name):
            setattr(ta, name, value)
            return True
        return False

    _set("num_train_epochs", args.num_epochs)
    if not _set("per_device_train_batch_size", args.train_batch_size):
        _set("per_gpu_train_batch_size", args.train_batch_size)
    if not _set("per_device_eval_batch_size", args.eval_batch_size):
        _set("per_gpu_eval_batch_size", args.eval_batch_size)
    _set("learning_rate", args.learning_rate)
    _set("weight_decay", args.weight_decay)
    if not _set("warmup_ratio", args.warmup_ratio):
        _set("warmup_steps", 0)
    _set("gradient_accumulation_steps", args.gradient_accumulation)
    _set("seed", args.seed)
    _set("fp16", args.fp16)

    # FIX 3: correct ceil for steps_per_epoch with grad accumulation
    world_bs = args.train_batch_size  # (single-GPU assumption)
    steps_per_epoch = max(1, math.ceil(len(train_ds) / (max(1, world_bs) * max(1, args.gradient_accumulation))))
    total_steps = steps_per_epoch * max(1, args.num_epochs)
    if hasattr(ta, "warmup_steps"):
        ta.warmup_steps = int(total_steps * float(args.warmup_ratio))

    # Ensure evaluation runs at epoch boundaries across HF versions
    if hasattr(ta, "evaluation_strategy"):
        ta.evaluation_strategy = "epoch"
    elif hasattr(ta, "eval_strategy"):
        ta.eval_strategy = IntervalStrategy.EPOCH  # FIX 1: was STEPS; now EPOCH
    else:
        if hasattr(ta, "eval_steps"):
            ta.eval_steps = steps_per_epoch

    # Optional logging alignment if available
    _set("logging_steps", 100)

    # Make old-HF eval stable with extra tensors
    if hasattr(ta, "remove_unused_columns"):
        ta.remove_unused_columns = False
    if hasattr(ta, "prediction_loss_only"):
        ta.prediction_loss_only = False

    training_args = ta
    # -------------------------------------------------------

    # Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_fn,
        loss_type=args.loss_type,
        rdrop_alpha=args.rdrop_alpha,
        label_smoothing=args.label_smoothing,
        class_weights=class_weights.cuda() if (class_weights is not None and torch.cuda.is_available()) else class_weights,
    )

    trainer.add_callback(PrettyEvalCallback())

    # Configure EarlyStopping fields before adding the callback
    training_args.metric_for_best_model = "macro_f1"
    training_args.greater_is_better = True
    training_args.load_best_model_at_end = True

    if args.early_stopping_patience and args.early_stopping_patience > 0:
        try:
            trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
        except Exception:
            pass

    best_dir = Path(out_dir) / "best_model"
    trainer.add_callback(
        SaveBestByAccuracyCallback(
            best_dir=best_dir,
            tokenizer=tokenizer,
            label2id=label2id,
            selected_numeric_columns=numeric_columns,
        )
    )

    print("[start] Training...")
    trainer.train()

    print("[start] Final evaluation...")
    v_metrics = trainer.evaluate(valid_ds)
    t_metrics = trainer.evaluate(test_ds)

    with open(out_dir / "valid_metrics.json", "w") as f:
        json.dump({k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in v_metrics.items()}, f, indent=2)
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump({k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in t_metrics.items()}, f, indent=2)

    with open(out_dir / "label_mapping.json", "w") as f:
        json.dump(label2id, f, indent=2)

    print("[done] Wrote metrics to:")
    print(f"       {out_dir/'valid_metrics.json'}")
    print(f"       {out_dir/'test_metrics.json'}")


if __name__ == "__main__":
    main()
