"""
models/baseline_models.py
--------------------------
Classical baseline models for Vietnamese sentiment analysis:
  - SVM with TF-IDF features
  - BiLSTM with Word2Vec embeddings (PyTorch)
"""

import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from typing import List, Dict, Optional


# ══════════════════════════════════════════════════════════════════════════════
# 1. SVM + TF-IDF
# ══════════════════════════════════════════════════════════════════════════════

class SVMClassifier:
    """
    Linear SVM with TF-IDF character + word n-gram features.
    Calibrated via Platt scaling for probability outputs.
    """

    def __init__(
        self,
        max_features: int = 50_000,
        ngram_range: tuple = (1, 3),
        C: float = 1.0,
        seed: int = 42,
    ):
        self.seed = seed
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="word",
                ngram_range=ngram_range,
                max_features=max_features,
                sublinear_tf=True,
                min_df=2,
            )),
            ("clf", CalibratedClassifierCV(
                LinearSVC(C=C, random_state=seed, max_iter=5000),
                cv=3,
            )),
        ])

    def train(self, texts: List[str], labels: List[int]) -> None:
        print(f"[SVM] Training on {len(texts)} samples...")
        self.pipeline.fit(texts, labels)
        print("[SVM] Done.")

    def evaluate(self, texts: List[str], labels: List[int]) -> Dict:
        preds = self.pipeline.predict(texts)
        acc   = accuracy_score(labels, preds)
        p, r, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        return {
            "accuracy": float(acc), "precision": float(p),
            "recall": float(r), "macro_f1": float(f1),
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)
        print(f"[SVM] Saved to {path}")

    @classmethod
    def load(cls, path: str):
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            obj.pipeline = pickle.load(f)
        return obj


# ══════════════════════════════════════════════════════════════════════════════
# 2. BiLSTM + Word2Vec / FastText embeddings
# ══════════════════════════════════════════════════════════════════════════════

class BiLSTMModel(nn.Module):
    """
    Bidirectional LSTM with mean-pool aggregation over time steps.
    Input: token embeddings (pre-trained or random init)
    Output: logits over num_labels classes
    """

    def __init__(
        self,
        vocab_size:   int,
        embed_dim:    int   = 100,
        hidden_size:  int   = 128,
        num_layers:   int   = 2,
        num_labels:   int   = 3,
        dropout:      float = 0.3,
        pretrained_embeddings: Optional[np.ndarray] = None,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(
                torch.tensor(pretrained_embeddings, dtype=torch.float)
            )
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        emb = self.dropout(self.embedding(input_ids))    # [B, L, E]
        out, _ = self.bilstm(emb)                        # [B, L, 2H]

        # Mean pool over non-padding tokens
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        else:
            pooled = out.mean(dim=1)

        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}


class BiLSTMTrainer:
    """
    Training loop for BiLSTM baseline.
    Builds a simple character/word vocabulary and trains from scratch
    (or from pre-trained embeddings if provided).
    """

    def __init__(
        self,
        embed_dim:   int   = 100,
        hidden_size: int   = 128,
        num_layers:  int   = 2,
        lr:          float = 1e-3,
        epochs:      int   = 20,
        batch_size:  int   = 64,
        max_length:  int   = 128,
        patience:    int   = 5,
        device:      str   = None,
        seed:        int   = 42,
        output_dir:  str   = "checkpoints_bilstm",
        pretrained_emb_path: Optional[str] = None,
    ):
        self.embed_dim   = embed_dim
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lr          = lr
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.max_length  = max_length
        self.patience    = patience
        self.seed        = seed
        self.output_dir  = output_dir
        self.pretrained_emb_path = pretrained_emb_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab  = {"<PAD>": 0, "<UNK>": 1}
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"[BiLSTM] Device: {self.device}")

    # ── Vocabulary ────────────────────────────────────────────────────────────
    def build_vocab(self, texts: List[str], min_freq: int = 2):
        from collections import Counter
        counter = Counter()
        for text in texts:
            counter.update(text.split())
        for word, freq in counter.items():
            if freq >= min_freq and word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        print(f"[BiLSTM] Vocabulary size: {len(self.vocab)}")

    def _encode(self, texts: List[str]) -> torch.Tensor:
        ids = []
        for text in texts:
            tokens = text.split()[: self.max_length]
            row = [self.vocab.get(t, 1) for t in tokens]
            # Pad
            row += [0] * (self.max_length - len(row))
            ids.append(row)
        return torch.tensor(ids, dtype=torch.long)

    def _make_loader(self, texts, labels, shuffle=False):
        X = self._encode(texts)
        mask = (X != 0).long()
        y = torch.tensor(labels, dtype=torch.long)
        ds = TensorDataset(X, mask, y)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    # ── Pre-trained embeddings ────────────────────────────────────────────────
    def _load_pretrained_embeddings(self) -> Optional[np.ndarray]:
        if not self.pretrained_emb_path or not os.path.exists(self.pretrained_emb_path):
            return None
        print(f"[BiLSTM] Loading embeddings from {self.pretrained_emb_path}...")
        embeddings = np.random.uniform(-0.1, 0.1, (len(self.vocab), self.embed_dim))
        with open(self.pretrained_emb_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < self.embed_dim + 1:
                    continue
                word = parts[0]
                vec  = np.array(parts[1:self.embed_dim + 1], dtype=float)
                if word in self.vocab:
                    embeddings[self.vocab[word]] = vec
        print(f"[BiLSTM] Embeddings loaded.")
        return embeddings

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, train_texts, train_labels, val_texts, val_labels) -> Dict:
        self.build_vocab(train_texts)
        pre_emb = self._load_pretrained_embeddings()

        self.model = BiLSTMModel(
            vocab_size=len(self.vocab),
            embed_dim=self.embed_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            pretrained_embeddings=pre_emb,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2, factor=0.5
        )

        train_loader = self._make_loader(train_texts, train_labels, shuffle=True)
        val_loader   = self._make_loader(val_texts,   val_labels)

        best_f1    = 0.0
        no_improve = 0
        history    = []

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0
            for X, mask, y in tqdm(train_loader, desc=f"BiLSTM Epoch {epoch}"):
                X, mask, y = X.to(self.device), mask.to(self.device), y.to(self.device)
                loss = self.model(X, mask, y)["loss"]
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            val_metrics = self._evaluate(val_loader)
            val_f1 = val_metrics["macro_f1"]
            scheduler.step(1 - val_f1)

            print(
                f"  Epoch {epoch} | loss={total_loss/len(train_loader):.4f} | "
                f"val_f1={val_f1:.4f}"
            )
            history.append({"epoch": epoch, **val_metrics})

            if val_f1 > best_f1:
                best_f1    = val_f1
                no_improve = 0
                os.makedirs(self.output_dir, exist_ok=True)
                torch.save(self.model.state_dict(),
                           os.path.join(self.output_dir, "bilstm_best.pt"))
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        # Load best
        self.model.load_state_dict(torch.load(
            os.path.join(self.output_dir, "bilstm_best.pt"), map_location=self.device
        ))
        return {"history": history, "best_val_f1": best_f1}

    def _evaluate(self, loader) -> Dict:
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, mask, y in loader:
                X, mask = X.to(self.device), mask.to(self.device)
                preds = self.model(X, mask)["logits"].argmax(-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())
        acc = accuracy_score(all_labels, all_preds)
        p, r, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )
        return {"accuracy": float(acc), "precision": float(p),
                "recall": float(r), "macro_f1": float(f1)}

    def evaluate(self, texts, labels) -> Dict:
        loader = self._make_loader(texts, labels)
        return self._evaluate(loader)
