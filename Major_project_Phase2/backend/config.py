"""
Central configuration for the Phase 2 backend.
"""
import os
from datetime import date

# ---------------------------------------------------
# Company Universe
# ---------------------------------------------------
COMPANIES = {
    # Top 10 NASDAQ
    "AAPL":  {"name": "Apple Inc.",                 "market": "NASDAQ", "sector": "Technology"},
    "MSFT":  {"name": "Microsoft Corporation",      "market": "NASDAQ", "sector": "Technology"},
    "AMZN":  {"name": "Amazon.com Inc.",            "market": "NASDAQ", "sector": "Consumer Cyclical"},
    "NVDA":  {"name": "NVIDIA Corporation",         "market": "NASDAQ", "sector": "Technology"},
    "GOOGL": {"name": "Alphabet Inc.",              "market": "NASDAQ", "sector": "Communication Services"},
    "META":  {"name": "Meta Platforms Inc.",         "market": "NASDAQ", "sector": "Communication Services"},
    "TSLA":  {"name": "Tesla Inc.",                 "market": "NASDAQ", "sector": "Consumer Cyclical"},
    "AVGO":  {"name": "Broadcom Inc.",              "market": "NASDAQ", "sector": "Technology"},
    "COST":  {"name": "Costco Wholesale Corp.",     "market": "NASDAQ", "sector": "Consumer Defensive"},
    "NFLX":  {"name": "Netflix Inc.",               "market": "NASDAQ", "sector": "Communication Services"},
    # Dell Technologies
    "DELL":  {"name": "Dell Technologies Inc.",     "market": "NYSE",   "sector": "Technology"},
    # Indian Companies
    "RELIANCE.NS": {"name": "Reliance Industries",  "market": "NSE", "sector": "Energy"},
    "TCS.NS":      {"name": "Tata Consultancy Services", "market": "NSE", "sector": "Technology"},
    "INFY.NS":     {"name": "Infosys Limited",      "market": "NSE", "sector": "Technology"},
    "HDFCBANK.NS": {"name": "HDFC Bank Limited",    "market": "NSE", "sector": "Financial Services"},
    "WIPRO.NS":    {"name": "Wipro Limited",        "market": "NSE", "sector": "Technology"},
}

ALL_TICKERS = list(COMPANIES.keys())

# ---------------------------------------------------
# Data Configuration
# ---------------------------------------------------
DATA_START_DATE = "2015-01-01"
DATA_END_DATE = date.today().isoformat()
PREDICTION_HORIZON = 1      # predict next-day direction
LOOKBACK_WINDOW = 60        # sequence length for DL models
TEST_SPLIT_DAYS = 180       # last 6 months for testing

# ---------------------------------------------------
# Model Configuration
# ---------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

DL_MODELS = ["lstm", "gru", "cnn_bilstm"]
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EARLY_STOP_PATIENCE = 15
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3

# ---------------------------------------------------
# Fusion Weights
# ---------------------------------------------------
SENTIMENT_WEIGHT = 0.4
DL_WEIGHT = 0.6

# ---------------------------------------------------
# Feature Columns (from feature engineering)
# ---------------------------------------------------
TECHNICAL_FEATURES = [
    "ret_1d", "ret_3d", "ret_5d", "ret_10d",
    "vol_10d", "vol_21d",
    "sma_ratio_5_10", "sma_ratio_5_20",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_pos", "bb_bw",
    "atr_14", "adx_14",
    "mom_10", "mom_20",
    "zscore_20", "vol_z_20",
    "obv_norm",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
    "vwap_ratio",
    "stoch_k", "stoch_d",
]

# ---------------------------------------------------
# Flask Configuration
# ---------------------------------------------------
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True
