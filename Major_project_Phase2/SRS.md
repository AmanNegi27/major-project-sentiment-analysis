# Software Requirements Specification (SRS)

## Sentiment Analysis of Corporate Earning Calls for Stock Price Direction Prediction - Phase II

**Version:** 2.0  
**Date:** February 2026  
**Author:** Homi Shivanshu Purbey

---

## 1. Introduction

### 1.1 Purpose
This document specifies the software requirements for the Stock Price Direction Prediction System (Phase II), which uses Natural Language Processing (NLP) sentiment analysis on corporate earnings calls combined with deep learning models to predict stock price movements.

### 1.2 Scope
The system analyzes earnings call transcripts using VADER and FinBERT sentiment models, combines this with technical indicator-based deep learning predictions, and provides a fusion probability for stock direction (UP/DOWN).

### 1.3 Definitions and Acronyms
| Term | Definition |
|------|------------|
| NLP | Natural Language Processing |
| VADER | Valence Aware Dictionary and sEntiment Reasoner |
| FinBERT | Financial BERT - domain-specific transformer model |
| LSTM | Long Short-Term Memory neural network |
| GRU | Gated Recurrent Unit neural network |
| CNN | Convolutional Neural Network |
| BERT | Bidirectional Encoder Representations from Transformers |
| RoBERTa | Robustly Optimized BERT Pretraining Approach |
| API | Application Programming Interface |

---

## 2. Overall Description

### 2.1 Product Perspective
This is Phase II of the project, building upon Phase I with:
- Addition of Transformer-based deep learning models
- Modern React-style frontend dashboard
- Real-time stock price display
- Interactive price charts
- Improved model accuracy through ensemble methods

### 2.2 Product Functions
1. **Sentiment Analysis**: Analyze earnings call transcripts using VADER and FinBERT
2. **Deep Learning Prediction**: Predict stock direction using 8 different models
3. **Fusion Prediction**: Combine sentiment and DL predictions using weighted formula
4. **Real-time Display**: Show live stock prices with change indicators
5. **Interactive Charts**: Display 4-month price history for selected stocks
6. **Model Comparison**: Display accuracy metrics for all trained models

### 2.3 User Classes and Characteristics
| User Class | Description |
|------------|-------------|
| Financial Analysts | Professionals analyzing earnings calls for investment decisions |
| Traders | Active traders seeking directional signals |
| Researchers | Academic users studying NLP in finance |
| Students | Learning about ML/DL applications in finance |

### 2.4 Operating Environment
- **Backend**: Python 3.10+, Flask, PyTorch
- **Frontend**: HTML5, TailwindCSS, Chart.js
- **Database**: File-based (JSON, joblib)
- **Deployment**: Local server (localhost:5000 for API, localhost:3000 for frontend)

---

## 3. Specific Requirements

### 3.1 Functional Requirements

#### FR-001: Sentiment Analysis
- **Description**: System shall analyze text transcripts for sentiment
- **Input**: Earnings call transcript (text)
- **Output**: Sentiment scores (compound, positive, negative, neutral ratios)
- **Models**: VADER (rule-based), FinBERT (transformer-based)

#### FR-002: Deep Learning Prediction
- **Description**: System shall predict stock direction using trained models
- **Input**: Historical price data, technical indicators
- **Output**: Probability of UP direction (0-1)
- **Models**: 
  - LSTM (Long Short-Term Memory)
  - GRU (Gated Recurrent Unit)
  - CNN-BiLSTM (Hybrid)
  - Transformer Encoder
  - BERT-Style Model
  - RoBERTa-Style Model
  - DistilBERT-Style Model
  - Hybrid Transformer-LSTM

#### FR-003: Fusion Prediction
- **Description**: System shall combine sentiment and DL predictions
- **Formula**: `P(final) = 0.4 × P(sentiment) + 0.6 × P(deep_learning)`
- **Output**: Final direction (UP/DOWN) with confidence percentage

#### FR-004: Company Selection
- **Description**: System shall support 16 companies across 3 markets
- **Markets**: NASDAQ (10), NYSE (1), NSE (5)
- **Companies**: AAPL, MSFT, AMZN, NVDA, GOOGL, META, TSLA, AVGO, COST, NFLX, DELL, RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, WIPRO.NS

#### FR-005: Price Chart Display
- **Description**: System shall display interactive price charts
- **Timeframes**: 1 month, 4 months, 1 year
- **Data**: Adjusted close prices from yfinance

#### FR-006: Live Market Overview
- **Description**: System shall display real-time prices for top 5 stocks
- **Indicators**: Price, percentage change, up/down arrows
- **Update**: On page load and manual refresh

### 3.2 Non-Functional Requirements

#### NFR-001: Performance
- API response time < 5 seconds for sentiment analysis
- Model inference time < 2 seconds
- Page load time < 3 seconds

#### NFR-002: Usability
- Intuitive dark blue dashboard interface
- Hover effects on interactive elements
- Responsive design for desktop and tablet

#### NFR-003: Reliability
- Graceful error handling for API failures
- Fallback to mock data when API unavailable
- Model persistence across server restarts

#### NFR-004: Scalability
- Support for additional companies via configuration
- Modular model architecture for easy additions

---

## 4. System Architecture

### 4.1 High-Level Architecture
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│   Flask API     │────▶│   ML Models     │
│   (HTML/JS)     │◀────│   (Python)      │◀────│   (PyTorch)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │   Data Layer    │
                        │ (yfinance/NLTK) │
                        └─────────────────┘
```

### 4.2 Component Diagram
```
Backend/
├── app.py              # Flask API server
├── config.py           # Configuration settings
├── sentiment/
│   ├── vader_analyzer.py    # VADER sentiment
│   └── finbert_analyzer.py  # FinBERT sentiment
├── models/
│   ├── lstm_model.py        # LSTM architecture
│   ├── gru_model.py         # GRU architecture
│   ├── cnn_bilstm_model.py  # CNN-BiLSTM hybrid
│   ├── transformer_model.py # Transformer models
│   ├── trainer.py           # Training pipeline
│   └── predictor.py         # Inference engine
└── data/
    ├── fetcher.py           # Price data fetching
    └── feature_engineering.py # Technical indicators
```

---

## 5. Data Requirements

### 5.1 Input Data
| Data Type | Source | Format |
|-----------|--------|--------|
| Stock Prices | yfinance API | OHLCV DataFrame |
| Earnings Transcripts | User Input | Plain Text |
| Company Info | Configuration | JSON |

### 5.2 Technical Indicators (28 Features)
- Price-based: Returns, Log Returns, Volatility
- Moving Averages: SMA (5, 10, 20, 50), EMA
- Momentum: RSI, MACD, Stochastic, ROC
- Volatility: Bollinger Bands, ATR
- Volume: OBV, VWAP, Volume Ratio
- Trend: ADX, Parabolic SAR
- Calendar: Day of Week, Month, Quarter

### 5.3 Output Data
| Output | Format | Range |
|--------|--------|-------|
| Direction | String | "UP" / "DOWN" |
| Probability | Float | 0.0 - 1.0 |
| Sentiment Scores | JSON | Various metrics |

---

## 6. Interface Requirements

### 6.1 API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/companies` | GET | List companies |
| `/api/analyze` | POST | Analyze transcript |
| `/api/model-metrics` | GET | Model performance |
| `/api/price-history` | GET | Historical prices |

### 6.2 User Interface
- **Header**: Project title, live indicator
- **Market Overview**: 5 ticker cards with prices
- **Price Chart**: Interactive 4-month chart
- **Analysis Panel**: Company select, date, transcript input
- **Results Panel**: Prediction with breakdown
- **Project Details**: Collapsible dropdown with model info
- **Footer**: Credits and project info

---

## 7. Quality Attributes

### 7.1 Accuracy Targets
| Model | Target Accuracy |
|-------|-----------------|
| Baseline (Random) | 50% |
| LSTM/GRU | 51-52% |
| Transformer Models | 53-55% |
| Best Model | >55% |

### 7.2 Code Quality
- PEP 8 compliant Python code
- Type hints for function signatures
- Docstrings for classes and methods
- Modular architecture

---

## 8. Constraints and Assumptions

### 8.1 Constraints
- Limited to 16 pre-configured companies
- Requires internet for price data fetching
- FinBERT requires ~500MB model download
- Training requires significant compute time

### 8.2 Assumptions
- Users have basic understanding of stock markets
- Earnings transcripts are in English
- Historical data available for all companies
- Stable internet connection available

---

## 9. Appendices

### 9.1 Glossary
- **Earnings Call**: Quarterly conference call where company discusses financial results
- **Sentiment**: Emotional tone of text (positive, negative, neutral)
- **Technical Indicators**: Mathematical calculations based on price/volume
- **Fusion**: Combining multiple prediction sources

### 9.2 References
- VADER: Hutto, C.J. & Gilbert, E.E. (2014)
- FinBERT: Araci, D. (2019)
- LSTM: Hochreiter & Schmidhuber (1997)
- Transformer: Vaswani et al. (2017)

---

**Document End**
