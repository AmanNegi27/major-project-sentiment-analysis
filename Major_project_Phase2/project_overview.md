# Project Overview

## Sentiment Analysis of Corporate Earning Calls for Stock Price Direction Prediction

### Phase II - Deep Learning Models

---

## Executive Summary

This project implements an advanced stock price direction prediction system that combines Natural Language Processing (NLP) sentiment analysis of corporate earnings calls with deep learning models trained on historical financial data. The system uses a fusion approach to provide probabilistic predictions of whether a stock will move UP or DOWN following an earnings announcement.

---

## Project Objectives

1. **Primary Goal**: Predict stock price direction (UP/DOWN) with accuracy exceeding random baseline (50%)
2. **NLP Integration**: Extract sentiment signals from earnings call transcripts
3. **Deep Learning**: Train multiple neural network architectures on technical indicators
4. **Fusion Approach**: Combine sentiment and technical analysis for robust predictions
5. **User Interface**: Provide an intuitive, professional-grade trading dashboard

---

## Technical Architecture

### Backend Stack
| Component | Technology | Purpose |
|-----------|------------|---------|
| Web Framework | Flask | REST API server |
| ML Framework | PyTorch | Deep learning models |
| NLP - Rule Based | NLTK VADER | Sentiment scoring |
| NLP - Transformer | FinBERT | Financial sentiment |
| Data Source | yfinance | Historical prices |
| Feature Engineering | pandas, numpy | Technical indicators |

### Frontend Stack
| Component | Technology | Purpose |
|-----------|------------|---------|
| Styling | TailwindCSS | Dark blue theme |
| Charts | Chart.js | Price visualization |
| Interactivity | Vanilla JS | Dynamic updates |

---

## Deep Learning Models

### Phase II introduces 8 deep learning architectures:

#### Recurrent Neural Networks (RNN)
1. **LSTM** (Long Short-Term Memory)
   - Captures long-term dependencies in time series
   - Multi-layer with attention mechanism
   - Baseline accuracy: ~51%

2. **GRU** (Gated Recurrent Unit)
   - Simplified LSTM variant
   - Faster training, similar performance
   - Baseline accuracy: ~49.6%

3. **CNN-BiLSTM** (Hybrid)
   - CNN for local pattern extraction
   - Bidirectional LSTM for sequence modeling
   - Baseline accuracy: ~49.5%

#### Transformer-Based Models (NEW in Phase II)
4. **Transformer Encoder**
   - Pure attention-based architecture
   - Positional encoding for sequence order
   - Expected accuracy: ~52%

5. **BERT-Style Model**
   - Pre-norm architecture
   - Multi-head self-attention (8 heads)
   - 4 encoder layers
   - Expected accuracy: ~53%

6. **RoBERTa-Style Model**
   - Post-norm architecture
   - Learnable positional embeddings
   - 6 encoder layers
   - Expected accuracy: ~54%

7. **DistilBERT-Style Model**
   - Lightweight transformer
   - 3 encoder layers
   - Faster inference
   - Expected accuracy: ~52%

8. **Hybrid Transformer-LSTM**
   - Combines transformer attention with LSTM memory
   - Best of both architectures
   - Expected accuracy: ~55%

---

## Sentiment Analysis

### VADER (Valence Aware Dictionary and sEntiment Reasoner)
- Rule-based sentiment analysis
- Optimized for social media and financial text
- Outputs: compound, positive, negative, neutral scores
- Fast inference, no GPU required

### FinBERT
- BERT model fine-tuned on financial communications
- Domain-specific understanding of financial language
- Outputs: positive, negative, neutral probabilities
- Requires model download (~500MB)

---

## Fusion Formula

The system combines sentiment and deep learning predictions using a weighted average:

```
P(final) = 0.4 × P(sentiment) + 0.6 × P(deep_learning)
```

**Rationale**:
- Deep learning models capture technical patterns (60% weight)
- Sentiment provides qualitative context (40% weight)
- Weights determined through empirical testing

---

## Feature Engineering

### 28 Technical Indicators

| Category | Features |
|----------|----------|
| **Returns** | Daily return, Log return, 5-day return |
| **Moving Averages** | SMA(5,10,20,50), EMA(12,26) |
| **Momentum** | RSI(14), MACD, Stochastic %K/%D, ROC |
| **Volatility** | Bollinger Bands, ATR(14), Historical volatility |
| **Volume** | OBV, VWAP, Volume ratio, Volume SMA |
| **Trend** | ADX, Parabolic SAR, Price position |
| **Calendar** | Day of week, Month, Quarter |

### Data Pipeline
1. Fetch 10+ years of OHLCV data from yfinance
2. Calculate technical indicators per ticker
3. Create binary labels (1 if next day close > today, else 0)
4. Normalize features using StandardScaler
5. Create sequences of 60 days for model input
6. Split: 90% train, 10% test (time-based)

---

## Supported Companies

### NASDAQ (10 companies)
| Ticker | Company | Sector |
|--------|---------|--------|
| AAPL | Apple Inc. | Technology |
| MSFT | Microsoft Corporation | Technology |
| AMZN | Amazon.com Inc. | Consumer |
| NVDA | NVIDIA Corporation | Semiconductors |
| GOOGL | Alphabet Inc. | Technology |
| META | Meta Platforms Inc. | Technology |
| TSLA | Tesla Inc. | Automotive |
| AVGO | Broadcom Inc. | Semiconductors |
| COST | Costco Wholesale | Retail |
| NFLX | Netflix Inc. | Entertainment |

### NYSE (1 company)
| Ticker | Company | Sector |
|--------|---------|--------|
| DELL | Dell Technologies | Technology |

### NSE - India (5 companies)
| Ticker | Company | Sector |
|--------|---------|--------|
| RELIANCE.NS | Reliance Industries | Conglomerate |
| TCS.NS | Tata Consultancy Services | IT Services |
| INFY.NS | Infosys Limited | IT Services |
| HDFCBANK.NS | HDFC Bank Limited | Banking |
| WIPRO.NS | Wipro Limited | IT Services |

---

## User Interface Features

### Dashboard Components
1. **Header**: Project title with live indicator
2. **Market Overview**: 5 top stocks with real-time prices
3. **Price Chart**: Interactive 4-month chart with zoom
4. **Analysis Panel**: Company selection, date picker, transcript input
5. **Results Panel**: Prediction with confidence and breakdown
6. **Project Details**: Collapsible section with model metrics
7. **Footer**: Credits to Homi Shivanshu Purbey

### Design Principles
- **Color Scheme**: Dark navy blue (#0a1628 to #1e3a6e)
- **Accent Color**: Electric blue (#2563eb, #00d4ff)
- **Typography**: Inter (UI), JetBrains Mono (data)
- **Interactions**: Hover effects, smooth transitions
- **Responsiveness**: Desktop and tablet optimized

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /api/health` | Health check and status |
| `GET /api/companies` | List of supported companies |
| `POST /api/analyze` | Analyze transcript and predict |
| `GET /api/model-metrics` | Model performance metrics |
| `GET /api/price-history` | Historical price data |
| `POST /api/train` | Trigger model retraining |

---

## Performance Metrics

### Model Evaluation
- **Accuracy**: Percentage of correct predictions
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under receiver operating characteristic curve

### Target Performance
| Metric | Target | Achieved |
|--------|--------|----------|
| Best Model Accuracy | >55% | TBD |
| API Response Time | <5s | <3s |
| Chart Load Time | <2s | <1s |

---

## Future Enhancements

1. **Real-time Data**: WebSocket for live price updates
2. **More Companies**: Expand to 50+ tickers
3. **News Integration**: Add news sentiment analysis
4. **Portfolio Mode**: Multi-stock analysis
5. **Backtesting**: Historical performance simulation
6. **Mobile App**: React Native version

---

## Project Structure

```
Major_Project_Phase2/
├── backend/
│   ├── app.py                    # Flask API
│   ├── app_minimal.py            # Simplified API
│   ├── config.py                 # Configuration
│   ├── requirements.txt          # Dependencies
│   ├── sentiment/
│   │   ├── vader_analyzer.py     # VADER
│   │   └── finbert_analyzer.py   # FinBERT
│   ├── models/
│   │   ├── lstm_model.py         # LSTM
│   │   ├── gru_model.py          # GRU
│   │   ├── cnn_bilstm_model.py   # CNN-BiLSTM
│   │   ├── transformer_model.py  # Transformers
│   │   ├── trainer.py            # Training
│   │   └── predictor.py          # Inference
│   ├── data/
│   │   ├── fetcher.py            # Data fetching
│   │   └── feature_engineering.py # Features
│   └── saved_models/             # Trained models
├── frontend/
│   └── index.html                # Dashboard
├── README.md                     # Quick start
├── SRS.md                        # Requirements
└── project_overview.md           # This file
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- pip package manager
- Internet connection

### Installation
```bash
# Backend
cd backend
pip install -r requirements.txt
python -m models.trainer  # Train models
python app_minimal.py     # Start API

# Frontend
cd frontend
start index.html          # Open in browser
```

### Usage
1. Open dashboard in browser
2. Click on a stock ticker to view chart
3. Select company from dropdown
4. Paste earnings call transcript
5. Click "Analyze & Predict"
6. View prediction results

---

## Credits

**Developed by**: Homi Shivanshu Purbey

**Phase II Enhancements**:
- Transformer-based deep learning models
- Futuristic dark blue dashboard
- Real-time stock price display
- Interactive price charts
- Comprehensive documentation

---

**Project**: Sentiment Analysis of Corporate Earning Calls for Stock Price Direction Prediction  
**Phase**: II - Deep Learning Models  
**Status**: Active Development
