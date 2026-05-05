"""
models/phobert_classifier.py
-----------------------------
PhoBERT (and mBERT) fine-tuning for Vietnamese sentiment classification.

Features:
  - Standard deterministic inference
  - Monte Carlo Dropout inference for uncertainty quantification
  - Checkpoint saving / loading
  - Early stopping
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
from typing import Optional, List, Dict, Tuple


PHOBERT_BASE = "vinai/phobert-base"
MBERT_BASE   = "bert-base-multilingual-cased"


class SentimentClassifier(nn.Module):
    """
    Transformer + linear classification head.
    [CLS] → Dropout(0.1) → Linear(768, num_labels) → softmax
    """

    def __init__(self, model_name: str, num_labels: int = 3, dropout: float = 0.1,
                 class_weights=None):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.num_labels  = num_labels
        # class_weights: tensor [num_labels] for imbalanced datasets (e.g. UIT-VSFC)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, input_ids, attention_mask, labels=None):
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]   # [CLS] token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(weight=self.class_weights)(logits, labels)

        return {"loss": loss, "logits": logits}


class PhoBERTTrainer:
    """
    Full training + evaluation loop for SentimentClassifier.

    Parameters
    ----------
    model_name   : HuggingFace model ID (phobert-base or mbert)
    num_labels   : 3 for VLSP 2016
    lr           : Learning rate
    epochs       : Max training epochs
    batch_size   : Train batch size
    max_length   : Token sequence length
    patience     : Early stopping patience (epochs without val F1 improvement)
    device       : 'cuda' | 'cpu'
    seed         : Random seed
    output_dir   : Where to save best checkpoint
    """

    def __init__(
        self,
        model_name:    str   = PHOBERT_BASE,
        num_labels:    int   = 3,
        lr:            float = 2e-5,
        epochs:        int   = 10,
        batch_size:    int   = 32,
        max_length:    int   = 128,
        patience:      int   = 3,
        device:        str   = None,
        seed:          int   = 42,
        output_dir:    str   = "checkpoints",
        class_weights: list  = None,   # e.g. [0.69, 9.54, 0.69] for UIT-VSFC
    ):
        self.model_name    = model_name
        self.lr            = lr
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.max_length    = max_length
        self.patience      = patience
        self.seed          = seed
        self.output_dir    = output_dir
        self.device        = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._set_seed()
        print(f"[Trainer] Model: {model_name} | Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Build class weight tensor for imbalanced datasets
        cw_tensor = None
        if class_weights is not None:
            cw_tensor = torch.tensor(class_weights, dtype=torch.float)
            print(f"[Trainer] Class weights: {class_weights}")

        self.model = SentimentClassifier(model_name, num_labels,
                                         class_weights=cw_tensor).to(self.device)

    def _set_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    # ── Data ──────────────────────────────────────────────────────────────────
    def _make_loader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=(self.device == "cuda"),
        )

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, train_dataset, val_dataset) -> Dict:
        train_loader = self._make_loader(train_dataset, shuffle=True)
        val_loader   = self._make_loader(val_dataset,   shuffle=False)

        total_steps  = len(train_loader) * self.epochs
        warmup_steps = int(0.10 * total_steps)

        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        best_f1   = 0.0
        no_improve = 0
        history   = []

        for epoch in range(1, self.epochs + 1):
            # ── Train epoch ──────────────────────────────────────────────────
            self.model.train()
            total_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out   = self.model(**batch)
                loss  = out["loss"]
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # ── Val epoch ────────────────────────────────────────────────────
            val_metrics = self.evaluate(val_loader)
            val_f1      = val_metrics["macro_f1"]

            print(
                f"  Epoch {epoch} | loss={avg_loss:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | val_f1={val_f1:.4f}"
            )
            history.append({
                "epoch": epoch, "train_loss": avg_loss, **val_metrics
            })

            # ── Early stopping ───────────────────────────────────────────────
            if val_f1 > best_f1:
                best_f1    = val_f1
                no_improve = 0
                self._save_checkpoint("best_model")
                print(f"  [Checkpoint] Saved best model (F1={best_f1:.4f})")
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"  [EarlyStopping] No improvement for {self.patience} epochs.")
                    break

        # Load best weights
        self._load_checkpoint("best_model")
        return {"history": history, "best_val_f1": best_f1}

    # ── Standard evaluation ───────────────────────────────────────────────────
    def evaluate(self, loader) -> Dict:
        from sklearn.metrics import (
            accuracy_score, precision_recall_fscore_support
        )
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                batch  = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch.pop("labels")
                logits = self.model(**batch)["logits"]
                preds  = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        p, r, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )
        return {
            "accuracy":   float(acc),
            "precision":  float(p),
            "recall":     float(r),
            "macro_f1":   float(f1),
        }

    # ── MC-Dropout inference ──────────────────────────────────────────────────
    def predict_with_uncertainty(
        self,
        loader,
        mc_passes: int = 20,
    ) -> Dict:
        """
        Run K stochastic forward passes with dropout ACTIVE.

        Returns
        -------
        dict with keys:
            predictions  : np.ndarray [N]       — final predicted class
            mean_probs   : np.ndarray [N, C]     — averaged softmax probs
            uncertainty  : np.ndarray [N]        — entropy of mean_probs
            confidence   : np.ndarray [N]        — max of mean_probs
            labels       : np.ndarray [N]        — true labels
        """
        from sklearn.metrics import (
            accuracy_score, precision_recall_fscore_support
        )

        # Keep dropout ACTIVE during inference
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()   # Keep BatchNorm frozen if present

        all_mc_probs = []   # [K, N, C]
        all_labels   = []

        for k in tqdm(range(mc_passes), desc="MC-Dropout passes"):
            pass_probs = []
            with torch.no_grad():
                for batch in loader:
                    batch  = {kk: v.to(self.device) for kk, v in batch.items()}
                    labels = batch.pop("labels")   # always pop to clean batch
                    logits = self.model(**batch)["logits"]
                    probs  = torch.softmax(logits, dim=-1).cpu().numpy()
                    pass_probs.extend(probs)
                    if k == 0:  # FIX Bug4: collect labels only on first pass (clear intent)
                        all_labels.extend(labels.cpu().numpy())
            all_mc_probs.append(pass_probs)

        # Stack: [K, N, C]
        mc_array = np.array(all_mc_probs)   # shape [K, N, C]
        mean_probs = mc_array.mean(axis=0)  # [N, C]

        # Predicted class = argmax of mean
        predictions = mean_probs.argmax(axis=-1)
        labels_arr  = np.array(all_labels)

        # Uncertainty = entropy of mean_probs
        eps = 1e-10
        entropy = -(mean_probs * np.log(mean_probs + eps)).sum(axis=-1)

        # Confidence = max probability
        confidence = mean_probs.max(axis=-1)

        acc = accuracy_score(labels_arr, predictions)
        p, r, f1, _ = precision_recall_fscore_support(
            labels_arr, predictions, average="macro", zero_division=0
        )

        # Restore eval mode
        self.model.eval()

        # FIX Bug3: 'uncertainty' here is ENTROPY (Eq.3, paper Section 3.3).
        # assign_strata() uses 1-max_softmax scale (Eq.1).
        # For stratified eval, use key 'confidence' → compute 1-confidence externally,
        # OR pass probs directly to uncertainty_report() which handles both scales.
        return {
            "predictions": predictions,
            "mean_probs":  mean_probs,
            "uncertainty": entropy,      # entropy scale  (Eq.3)
            "confidence":  confidence,   # max_softmax    (for Eq.1: use 1-confidence)
            "labels":      labels_arr,
            "accuracy":    float(acc),
            "precision":   float(p),
            "recall":      float(r),
            "macro_f1":    float(f1),
        }

    # ── Checkpoint helpers ────────────────────────────────────────────────────
    def _save_checkpoint(self, name: str):
        path = os.path.join(self.output_dir, name)
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))
        self.tokenizer.save_pretrained(path)

    def _load_checkpoint(self, name: str):
        path = os.path.join(self.output_dir, name)
        state = torch.load(os.path.join(path, "model.pt"), map_location=self.device)
        self.model.load_state_dict(state)
        print(f"[Trainer] Loaded checkpoint from {path}")

    def load_best(self):
        self._load_checkpoint("best_model")
