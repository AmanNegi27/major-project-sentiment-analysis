"""
Model training pipeline for all 3 deep learning models.
Trains LSTM, GRU, and CNN-BiLSTM on financial features,
evaluates each, and saves the best model.
"""
from __future__ import annotations

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import joblib

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    ALL_TICKERS, DATA_START_DATE, DATA_END_DATE, MODEL_DIR,
    TECHNICAL_FEATURES, LOOKBACK_WINDOW, TEST_SPLIT_DAYS,
    EPOCHS, BATCH_SIZE, LEARNING_RATE, EARLY_STOP_PATIENCE,
    HIDDEN_SIZE, NUM_LAYERS, DROPOUT,
)
from data.fetcher import DataFetcher
from data.feature_engineering import FeatureEngineer
from models.lstm_model import LSTMModel
from models.gru_model import GRUModel
from models.cnn_bilstm_model import CNNBiLSTMModel
from models.transformer_model import (
    TransformerEncoder,
    BERTStyleModel,
    RoBERTaStyleModel,
    DistilBERTStyleModel,
    HybridTransformerLSTM,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ModelTrainer:
    """Trains and evaluates all DL models (LSTM, GRU, CNN-BiLSTM, Transformers), selects the best one."""

    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()

    def _create_sequences(
        self, data: np.ndarray, labels: np.ndarray, seq_len: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(seq_len, len(data)):
            X.append(data[i - seq_len : i])
            y.append(labels[i])
        return np.array(X), np.array(y)

    def prepare_data(
        self,
        tickers: Optional[List[str]] = None,
        start: str = DATA_START_DATE,
        end: str = DATA_END_DATE,
    ) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray, int]:
        """Fetch data, engineer features, create sequences, and return train/test loaders."""
        tickers = tickers or ALL_TICKERS
        logger.info("Preparing data for %d tickers...", len(tickers))

        # Fetch and build dataset
        prices = DataFetcher.fetch_prices(tickers, start=start, end=end)
        if prices.empty:
            raise RuntimeError("No price data fetched. Check tickers and date range.")

        dataset = FeatureEngineer.build_dataset(prices, horizon=1, feature_cols=TECHNICAL_FEATURES)

        # Time-based train/test split
        dataset = dataset.sort_values("date").reset_index(drop=True)
        max_date = dataset["date"].max()
        cutoff = max_date - pd.Timedelta(days=TEST_SPLIT_DAYS)

        train_df = dataset[dataset["date"] < cutoff].copy()
        test_df = dataset[dataset["date"] >= cutoff].copy()

        logger.info("Train: %d rows, Test: %d rows", len(train_df), len(test_df))

        feature_cols = [c for c in TECHNICAL_FEATURES if c in dataset.columns]

        # Scale features
        X_train_raw = train_df[feature_cols].values.astype(np.float32)
        X_test_raw = test_df[feature_cols].values.astype(np.float32)
        y_train_raw = train_df["y_class"].values.astype(np.float32)
        y_test_raw = test_df["y_class"].values.astype(np.float32)

        X_train_scaled = self.scaler.fit_transform(X_train_raw)
        X_test_scaled = self.scaler.transform(X_test_raw)

        # Create sequences
        X_train_seq, y_train_seq = self._create_sequences(X_train_scaled, y_train_raw, LOOKBACK_WINDOW)
        X_test_seq, y_test_seq = self._create_sequences(X_test_scaled, y_test_raw, LOOKBACK_WINDOW)

        logger.info("Sequences — Train: %s, Test: %s", X_train_seq.shape, X_test_seq.shape)

        # Create DataLoaders
        train_ds = TensorDataset(
            torch.tensor(X_train_seq, dtype=torch.float32),
            torch.tensor(y_train_seq, dtype=torch.float32),
        )
        test_ds = TensorDataset(
            torch.tensor(X_test_seq, dtype=torch.float32),
            torch.tensor(y_test_seq, dtype=torch.float32),
        )

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        num_features = X_train_seq.shape[2]

        # Save scaler and metadata
        joblib.dump(self.scaler, os.path.join(self.model_dir, "scaler.joblib"))
        meta = {
            "feature_cols": feature_cols,
            "lookback_window": LOOKBACK_WINDOW,
            "tickers": tickers,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "num_features": num_features,
        }
        with open(os.path.join(self.model_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        return train_loader, test_loader, y_test_seq, X_test_seq, num_features

    def _train_single_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        model_name: str,
    ) -> nn.Module:
        """Train a single model with early stopping."""
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.BCELoss()

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(EPOCHS):
            model.train()
            epoch_losses = []

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)
            scheduler.step(avg_loss)

            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
                # Save best checkpoint
                torch.save(model.state_dict(), os.path.join(self.model_dir, f"{model_name}_best.pt"))
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    "[%s] Epoch %d/%d — Loss: %.4f (best: %.4f) — Patience: %d/%d",
                    model_name, epoch + 1, EPOCHS, avg_loss, best_loss,
                    patience_counter, EARLY_STOP_PATIENCE,
                )

            if patience_counter >= EARLY_STOP_PATIENCE:
                logger.info("[%s] Early stopping at epoch %d", model_name, epoch + 1)
                break

        # Load best checkpoint
        best_path = os.path.join(self.model_dir, f"{model_name}_best.pt")
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=self.device))

        return model

    def _evaluate_model(
        self, model: nn.Module, test_loader: DataLoader, y_true: np.ndarray
    ) -> Tuple[Dict, np.ndarray]:
        """Evaluate model and return metrics + predicted probabilities."""
        model.eval()
        all_probs = []

        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(self.device)
                probs = model(X_batch)
                all_probs.extend(probs.cpu().numpy())

        probs = np.array(all_probs[: len(y_true)])
        preds = (probs >= 0.5).astype(int)

        acc = float(accuracy_score(y_true, preds))
        f1 = float(f1_score(y_true, preds, zero_division=0))
        try:
            roc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else None
        except Exception:
            roc = None

        report = classification_report(y_true, preds, digits=3, zero_division=0)
        cm = confusion_matrix(y_true, preds).tolist()

        metrics = {
            "accuracy": acc,
            "f1": f1,
            "roc_auc": roc,
            "report": report,
            "confusion_matrix": cm,
        }
        return metrics, probs

    def train_all(
        self,
        tickers: Optional[List[str]] = None,
    ) -> Dict:
        """Train all 3 models, evaluate, select best, and save everything."""
        train_loader, test_loader, y_test, X_test, num_features = self.prepare_data(tickers)

        # Define models - Original 3 + New Transformer-based models
        model_configs = {
            # Original models
            "lstm": LSTMModel(
                input_size=num_features,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT,
            ),
            "gru": GRUModel(
                input_size=num_features,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT,
            ),
            "cnn_bilstm": CNNBiLSTMModel(
                input_size=num_features,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT,
            ),
            # Transformer-based models
            "transformer": TransformerEncoder(
                input_dim=num_features,
                d_model=64,
                nhead=4,
                num_layers=2,
                dim_feedforward=128,
                dropout=DROPOUT,
            ),
            "bert_style": BERTStyleModel(
                input_dim=num_features,
                hidden_dim=128,
                num_heads=8,
                num_layers=4,
                dropout=DROPOUT,
            ),
            "roberta_style": RoBERTaStyleModel(
                input_dim=num_features,
                hidden_dim=128,
                num_heads=8,
                num_layers=6,
                dropout=DROPOUT,
            ),
            "distilbert_style": DistilBERTStyleModel(
                input_dim=num_features,
                hidden_dim=96,
                num_heads=6,
                num_layers=3,
                dropout=DROPOUT,
            ),
            "hybrid_transformer_lstm": HybridTransformerLSTM(
                input_dim=num_features,
                hidden_dim=64,
                num_heads=4,
                num_transformer_layers=2,
                lstm_layers=2,
                dropout=DROPOUT,
            ),
        }

        all_metrics = {}
        all_probs = {}
        best_model_name = None
        best_accuracy = 0.0

        for name, model in model_configs.items():
            logger.info("=" * 60)
            logger.info("Training %s...", name.upper())
            logger.info("=" * 60)

            trained_model = self._train_single_model(model, train_loader, name)
            metrics, probs = self._evaluate_model(trained_model, test_loader, y_test)

            all_metrics[name] = metrics
            all_probs[name] = probs

            logger.info("[%s] Accuracy: %.4f | F1: %.4f | ROC-AUC: %s",
                        name, metrics["accuracy"], metrics["f1"],
                        f"{metrics['roc_auc']:.4f}" if metrics["roc_auc"] else "N/A")

            if metrics["accuracy"] > best_accuracy:
                best_accuracy = metrics["accuracy"]
                best_model_name = name

        # Save best model info
        logger.info("=" * 60)
        logger.info("BEST MODEL: %s with accuracy %.4f", best_model_name.upper(), best_accuracy)
        logger.info("=" * 60)

        results = {
            "models": all_metrics,
            "best_model": best_model_name,
            "best_accuracy": best_accuracy,
            "trained_at": datetime.now().isoformat(),
            "device": str(self.device),
        }

        # Save metrics JSON
        with open(os.path.join(self.model_dir, "metrics.json"), "w") as f:
            # Remove non-serializable items
            save_metrics = {}
            for name, m in all_metrics.items():
                save_metrics[name] = {k: v for k, v in m.items() if k != "report"}
                save_metrics[name]["report"] = m.get("report", "")
            results_save = {**results, "models": save_metrics}
            json.dump(results_save, f, indent=2)

        # Save best model config for easy reload
        best_config = {
            "model_name": best_model_name,
            "input_size": num_features,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
        }
        with open(os.path.join(self.model_dir, "best_model_config.json"), "w") as f:
            json.dump(best_config, f, indent=2)

        return results


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train all 3 DL models")
    parser.add_argument("--tickers", default=None, help="Comma-separated tickers (default: all)")
    parser.add_argument("--start", default=DATA_START_DATE, help="Start date")
    parser.add_argument("--end", default=DATA_END_DATE, help="End date")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None

    trainer = ModelTrainer()
    results = trainer.train_all(tickers=tickers)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    for name, m in results["models"].items():
        print(f"\n{name.upper()}: Acc={m['accuracy']:.4f} F1={m['f1']:.4f} ROC={m.get('roc_auc', 'N/A')}")
    print(f"\nBest: {results['best_model'].upper()} ({results['best_accuracy']:.4f})")
