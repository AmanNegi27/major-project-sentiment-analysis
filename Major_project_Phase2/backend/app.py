"""
Flask API for Earnings Call NLP + Deep Learning Stock Prediction.
Serves the React frontend with REST endpoints.
"""
from __future__ import annotations

import os
import sys
import json
import logging
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS

# Ensure backend is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import COMPANIES, ALL_TICKERS, FLASK_HOST, FLASK_PORT, FLASK_DEBUG, MODEL_DIR
from models.predictor import FusionPredictor
from data.fetcher import DataFetcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize predictor (lazy model load)
predictor = FusionPredictor(model_dir=MODEL_DIR)


# ---------------------------------------------------
# API Endpoints
# ---------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


@app.route("/api/companies", methods=["GET"])
def get_companies():
    """Return list of all supported companies with metadata."""
    companies = []
    for ticker, info in COMPANIES.items():
        companies.append({
            "ticker": ticker,
            "name": info["name"],
            "market": info["market"],
            "sector": info["sector"],
        })
    return jsonify({"companies": companies})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Main prediction endpoint.
    Expects JSON: { ticker, transcript, earnings_date }
    Returns fusion prediction with sentiment + DL breakdown.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    ticker = data.get("ticker", "").strip().upper()
    transcript = data.get("transcript", "").strip()
    earnings_date = data.get("earnings_date", "")

    if not ticker:
        return jsonify({"error": "ticker is required"}), 400
    if not transcript:
        return jsonify({"error": "transcript is required"}), 400
    if not earnings_date:
        return jsonify({"error": "earnings_date is required"}), 400

    if ticker not in COMPANIES:
        return jsonify({"error": f"Unsupported ticker: {ticker}. Supported: {ALL_TICKERS}"}), 400

    try:
        result = predictor.predict(ticker, transcript, earnings_date)
        return jsonify(result)
    except Exception as e:
        logger.exception("Prediction error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/model-metrics", methods=["GET"])
def model_metrics():
    """Return training metrics for all 3 DL models."""
    metrics = predictor.get_model_metrics()
    if metrics is None:
        return jsonify({
            "error": "No metrics found. Train models first with: python -m models.trainer",
            "trained": False,
        }), 404
    return jsonify({**metrics, "trained": True})


@app.route("/api/price-history", methods=["GET"])
def price_history():
    """Return historical price data for charting."""
    ticker = request.args.get("ticker", "").strip().upper()
    days = int(request.args.get("days", 180))

    if not ticker:
        return jsonify({"error": "ticker query param is required"}), 400

    prices = predictor.get_price_history(ticker, days=days)
    if prices is None:
        return jsonify({"error": f"No price data for {ticker}"}), 404

    prices["date"] = prices["date"].dt.strftime("%Y-%m-%d")
    return jsonify({
        "ticker": ticker,
        "days": days,
        "data": prices.to_dict(orient="records"),
    })


@app.route("/api/train", methods=["POST"])
def train():
    """Trigger model training (can be long-running)."""
    data = request.get_json() or {}
    tickers_str = data.get("tickers", "")

    if tickers_str:
        tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    else:
        tickers = ALL_TICKERS

    try:
        from models.trainer import ModelTrainer
        trainer = ModelTrainer()
        results = trainer.train_all(tickers=tickers)

        # Reload predictor with new model
        predictor.load()

        return jsonify({
            "status": "success",
            "best_model": results["best_model"],
            "best_accuracy": results["best_accuracy"],
            "models": {
                name: {"accuracy": m["accuracy"], "f1": m["f1"], "roc_auc": m.get("roc_auc")}
                for name, m in results["models"].items()
            },
        })
    except Exception as e:
        logger.exception("Training error: %s", e)
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------
# Run
# ---------------------------------------------------
def preload_models():
    """Preload all models before starting server."""
    # Load FinBERT first (heaviest)
    from sentiment.finbert_analyzer import FinBERTAnalyzer
    analyzer = FinBERTAnalyzer()
    if analyzer.is_available():
        logger.info("FinBERT preloaded successfully")
    else:
        logger.warning("FinBERT preload failed")

    # Load trained model
    if predictor.load():
        logger.info("Deep learning model loaded successfully")
    else:
        logger.warning("No trained model found. Train with: python -m models.trainer")

if __name__ == "__main__":
    # Preload models
    preload_models()

    # Run Flask without auto-reload
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=False,  # Disable debug mode to prevent reloads
        use_reloader=False  # Explicitly disable reloader
    )
