"""
Fusion Predictor — combines sentiment analysis with DL model prediction.
Fusion: P_final = 0.4 * P(sentiment) + 0.6 * P(deep_learning)
"""
from __future__ import annotations

import os
import sys
import json
import logging
from datetime import timedelta
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd
import torch
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MODEL_DIR, TECHNICAL_FEATURES, LOOKBACK_WINDOW,
    SENTIMENT_WEIGHT, DL_WEIGHT, HIDDEN_SIZE, NUM_LAYERS, DROPOUT,
)
from data.fetcher import DataFetcher
from data.feature_engineering import FeatureEngineer
from sentiment.vader_analyzer import VaderAnalyzer
from sentiment.finbert_analyzer import FinBERTAnalyzer
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


MODEL_CLASS_MAP = {
    "lstm": LSTMModel,
    "gru": GRUModel,
    "cnn_bilstm": CNNBiLSTMModel,
    "transformer": TransformerEncoder,
    "bert_style": BERTStyleModel,
    "roberta_style": RoBERTaStyleModel,
    "distilbert_style": DistilBERTStyleModel,
    "hybrid_transformer_lstm": HybridTransformerLSTM,
}


class FusionPredictor:
    """
    Loads the best trained DL model and performs fusion prediction
    combining sentiment probability with DL probability.
    """

    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = None
        self.meta = None
        self.best_config = None
        self.vader = VaderAnalyzer()
        self.finbert = FinBERTAnalyzer()
        self._loaded = False

    def load(self) -> bool:
        """Load the best model, scaler, and metadata."""
        try:
            # Load metadata
            meta_path = os.path.join(self.model_dir, "meta.json")
            config_path = os.path.join(self.model_dir, "best_model_config.json")

            if not os.path.exists(meta_path) or not os.path.exists(config_path):
                logger.warning("Model files not found. Train models first.")
                return False

            with open(meta_path) as f:
                self.meta = json.load(f)
            with open(config_path) as f:
                self.best_config = json.load(f)

            # Load scaler
            scaler_path = os.path.join(self.model_dir, "scaler.joblib")
            self.scaler = joblib.load(scaler_path)

            # Load model
            model_name = self.best_config["model_name"]
            model_class = MODEL_CLASS_MAP[model_name]
            
            # Handle different parameter names for different model types
            if model_name in ["transformer", "bert_style", "roberta_style", "distilbert_style", "hybrid_transformer_lstm"]:
                # Transformer models use different parameter names
                if model_name == "transformer":
                    self.model = model_class(
                        input_dim=self.best_config["input_size"],
                        d_model=self.best_config.get("d_model", 64),
                        nhead=self.best_config.get("nhead", 4),
                        num_layers=self.best_config.get("num_layers", 2),
                        dim_feedforward=self.best_config.get("dim_feedforward", 128),
                        dropout=self.best_config.get("dropout", DROPOUT),
                    )
                elif model_name == "bert_style":
                    self.model = model_class(
                        input_dim=self.best_config["input_size"],
                        hidden_dim=self.best_config.get("hidden_dim", 128),
                        num_heads=self.best_config.get("num_heads", 8),
                        num_layers=self.best_config.get("num_layers", 4),
                        dropout=self.best_config.get("dropout", DROPOUT),
                    )
                elif model_name == "roberta_style":
                    self.model = model_class(
                        input_dim=self.best_config["input_size"],
                        hidden_dim=self.best_config.get("hidden_dim", 128),
                        num_heads=self.best_config.get("num_heads", 8),
                        num_layers=self.best_config.get("num_layers", 6),
                        dropout=self.best_config.get("dropout", DROPOUT),
                    )
                elif model_name == "distilbert_style":
                    self.model = model_class(
                        input_dim=self.best_config["input_size"],
                        hidden_dim=self.best_config.get("hidden_dim", 96),
                        num_heads=self.best_config.get("num_heads", 6),
                        num_layers=self.best_config.get("num_layers", 3),
                        dropout=self.best_config.get("dropout", DROPOUT),
                    )
                elif model_name == "hybrid_transformer_lstm":
                    self.model = model_class(
                        input_dim=self.best_config["input_size"],
                        hidden_dim=self.best_config.get("hidden_dim", 64),
                        num_heads=self.best_config.get("num_heads", 4),
                        num_transformer_layers=self.best_config.get("num_transformer_layers", 2),
                        lstm_layers=self.best_config.get("lstm_layers", 2),
                        dropout=self.best_config.get("dropout", DROPOUT),
                    )
            else:
                # RNN-based models use original parameters
                self.model = model_class(
                    input_size=self.best_config["input_size"],
                    hidden_size=self.best_config.get("hidden_size", HIDDEN_SIZE),
                    num_layers=self.best_config.get("num_layers", NUM_LAYERS),
                    dropout=self.best_config.get("dropout", DROPOUT),
                )

            weights_path = os.path.join(self.model_dir, f"{model_name}_best.pt")
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()

            self._loaded = True
            logger.info("Loaded best model: %s", model_name)
            return True

        except Exception as e:
            logger.exception("Failed to load model: %s", e)
            return False

    def _get_dl_prediction(self, ticker: str, earnings_date: str) -> Dict[str, Any]:
        """Get DL model prediction for a ticker around an earnings date."""
        if not self._loaded:
            if not self.load():
                return {"error": "Model not loaded", "probability": 0.5}

        try:
            end_date = pd.to_datetime(earnings_date) + timedelta(days=1)
            start_date = end_date - timedelta(days=LOOKBACK_WINDOW * 3)

            prices = DataFetcher.fetch_single(ticker, start=start_date, end=end_date)
            if prices.empty:
                return {"error": "No price data available", "probability": 0.5}

            # Build features
            df = FeatureEngineer.add_technicals(prices)
            df = FeatureEngineer.add_calendar_features(df)

            feature_cols = [c for c in TECHNICAL_FEATURES if c in df.columns]
            df = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()

            if len(df) < LOOKBACK_WINDOW:
                return {"error": f"Not enough data ({len(df)} < {LOOKBACK_WINDOW})", "probability": 0.5}

            # Scale and create sequence
            X_raw = df.values[-LOOKBACK_WINDOW:].astype(np.float32)
            X_scaled = self.scaler.transform(X_raw)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Predict
            with torch.no_grad():
                prob = self.model(X_tensor).cpu().item()

            return {
                "probability": float(np.clip(prob, 0.0, 1.0)),
                "direction": "UP" if prob >= 0.5 else "DOWN",
                "model_name": self.best_config["model_name"],
                "last_close": float(prices["adj_close"].iloc[-1]),
            }

        except Exception as e:
            logger.exception("DL prediction error: %s", e)
            return {"error": str(e), "probability": 0.5}

    def _get_sentiment_prediction(self, transcript: str) -> Dict[str, Any]:
        """Analyze transcript with VADER + FinBERT and return blended sentiment probability."""
        vader_result = self.vader.analyze(transcript)

        # Get sentences for FinBERT
        sentences = [s["sentence"] for s in vader_result.sentence_scores]
        finbert_result = self.finbert.analyze(sentences)

        # Blend: if FinBERT available, 0.4*VADER + 0.6*FinBERT, else VADER only
        vader_prob = vader_result.probability
        if finbert_result.available:
            finbert_prob = finbert_result.probability
            sentiment_prob = 0.4 * vader_prob + 0.6 * finbert_prob
        else:
            finbert_prob = None
            sentiment_prob = vader_prob

        sentiment_prob = float(np.clip(sentiment_prob, 0.0, 1.0))

        return {
            "sentiment_probability": sentiment_prob,
            "vader_compound_mean": vader_result.compound_mean,
            "vader_probability": vader_prob,
            "finbert_polarity_mean": finbert_result.polarity_mean if finbert_result.available else None,
            "finbert_probability": finbert_prob,
            "finbert_available": finbert_result.available,
            "finbert_error": finbert_result.error,
            "n_sentences": len(vader_result.sentence_scores),
            "pos_ratio": vader_result.pos_ratio,
            "neg_ratio": vader_result.neg_ratio,
            "top_positive_vader": vader_result.top_positive[:3],
            "top_negative_vader": vader_result.top_negative[:3],
            "top_positive_finbert": finbert_result.top_positive[:3] if finbert_result.available else [],
            "top_negative_finbert": finbert_result.top_negative[:3] if finbert_result.available else [],
        }

    def predict(self, ticker: str, transcript: str, earnings_date: str) -> Dict[str, Any]:
        """
        Full fusion prediction:
        P_final = 0.4 * P(sentiment) + 0.6 * P(deep_learning)
        """
        # Get both predictions
        sentiment = self._get_sentiment_prediction(transcript)
        dl_result = self._get_dl_prediction(ticker, earnings_date)

        sentiment_prob = sentiment["sentiment_probability"]
        dl_prob = dl_result["probability"]

        # Fusion formula
        fusion_prob = SENTIMENT_WEIGHT * sentiment_prob + DL_WEIGHT * dl_prob
        fusion_prob = float(np.clip(fusion_prob, 0.0, 1.0))
        fusion_direction = "UP" if fusion_prob >= 0.5 else "DOWN"

        return {
            "ticker": ticker,
            "earnings_date": earnings_date,
            "fusion": {
                "probability": fusion_prob,
                "direction": fusion_direction,
                "sentiment_weight": SENTIMENT_WEIGHT,
                "dl_weight": DL_WEIGHT,
                "formula": f"{SENTIMENT_WEIGHT}*sentiment + {DL_WEIGHT}*dl",
            },
            "sentiment": sentiment,
            "deep_learning": dl_result,
            "components": {
                "sentiment_prob": sentiment_prob,
                "dl_prob": dl_prob,
                "fusion_prob": fusion_prob,
            },
        }

    def get_model_metrics(self) -> Optional[Dict]:
        """Load and return saved training metrics."""
        metrics_path = os.path.join(self.model_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            return None
        with open(metrics_path) as f:
            return json.load(f)

    def get_price_history(self, ticker: str, days: int = 180) -> Optional[pd.DataFrame]:
        """Fetch recent price history for charting."""
        from datetime import datetime
        end = datetime.now()
        start = end - timedelta(days=days)
        prices = DataFetcher.fetch_single(ticker, start=start, end=end)
        if prices.empty:
            return None
        return prices[["date", "open", "high", "low", "close", "adj_close", "volume"]]
