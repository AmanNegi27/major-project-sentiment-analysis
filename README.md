# Sentiment Analysis of Corporate Earning Calls for Stock Price Direction Prediction

## Phase II - Deep Learning Models


---

## Overview
This project uses **NLP sentiment analysis** (VADER + FinBERT) on earnings call transcripts combined with **8 deep learning models** (including Transformer-based architectures) trained on 10+ years of financial data to predict whether a stock's price will go **UP** or **DOWN** after an earnings call.

### Companies Covered (16)
- **NASDAQ (10)**: AAPL, MSFT, AMZN, NVDA, GOOGL, META, TSLA, AVGO, COST, NFLX
- **NYSE (1)**: DELL
- **NSE India (5)**: RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, WIPRO.NS

### Fusion Formula
```
P(final) = 0.4 × P(sentiment) + 0.6 × P(deep_learning_best_model)
```

### Deep Learning Models (8)

#### RNN-Based Models
1. **LSTM** — Long Short-Term Memory with attention
2. **GRU** — Gated Recurrent Unit with attention
3. **CNN-BiLSTM** — 1D Convolutional + Bidirectional LSTM hybrid

#### Transformer-Based Models (NEW in Phase II)
4. **Transformer Encoder** — Pure attention-based architecture
5. **BERT-Style** — Pre-norm multi-head attention (8 heads, 4 layers)
6. **RoBERTa-Style** — Post-norm with learnable positional embeddings (6 layers)
7. **DistilBERT-Style** — Lightweight transformer (3 layers)
8. **Hybrid Transformer-LSTM** — Combines attention with LSTM memory

The system automatically selects the best-performing model based on accuracy.

### Sentiment Analysis
- **VADER** — Rule-based sentiment optimized for financial text
- **FinBERT** — Transformer model fine-tuned on financial communications

---

## Tech Stack
| Layer     | Technology                              |
|-----------|-----------------------------------------|
| Frontend  | HTML5, TailwindCSS, Chart.js            |
| Backend   | Flask, Flask-CORS                        |
| NLP       | NLTK (VADER), HuggingFace Transformers (FinBERT) |
| DL Models | PyTorch (LSTM, GRU, CNN, Transformers)  |
| Data      | yfinance, pandas, numpy                 |

## Dashboard Features
- **Dark Blue Futuristic Theme** — Professional trading dashboard design
- **Live Market Overview** — Real-time prices for top 5 stocks with up/down indicators
- **Interactive Price Charts** — 4-month historical data with zoom controls
- **Prediction Results** — Clear UP/DOWN direction with confidence percentage
- **Model Comparison** — Performance metrics for all 8 deep learning models
- **Collapsible Project Details** — Technical information in dropdown section

---

## Setup & Run

### Backend
```bash
cd backend
pip install -r requirements.txt
# Train models (first time)
python -m models.trainer
# Start API server
python app.py
```
The Flask API runs on `http://localhost:5000`.

### Frontend
```bash
cd frontend
npm install
npm start
```
The React app runs on `http://localhost:3000`.

---

## API Endpoints

| Method | Endpoint              | Description                                |
|--------|-----------------------|--------------------------------------------|
| GET    | `/api/companies`      | List all supported companies               |
| POST   | `/api/analyze`        | Analyze transcript + predict direction     |
| GET    | `/api/model-metrics`  | Get training metrics for all 3 DL models   |
| POST   | `/api/train`          | Trigger model training for a company       |
| GET    | `/api/price-history`  | Get historical price data for a company    |

---

## Project Structure
```
Major_Project_Phase2/
├── backend/
│   ├── app.py                          # Flask API (full)
│   ├── app_minimal.py                  # Flask API (VADER only)
│   ├── config.py                       # Configuration & company list
│   ├── requirements.txt
│   ├── sentiment/
│   │   ├── vader_analyzer.py           # VADER sentiment
│   │   └── finbert_analyzer.py         # FinBERT sentiment
│   ├── data/
│   │   ├── fetcher.py                  # yfinance data fetcher
│   │   └── feature_engineering.py      # 28 technical indicators
│   ├── models/
│   │   ├── lstm_model.py               # LSTM architecture
│   │   ├── gru_model.py                # GRU architecture
│   │   ├── cnn_bilstm_model.py         # CNN-BiLSTM hybrid
│   │   ├── transformer_model.py        # Transformer models (NEW)
│   │   ├── trainer.py                  # Train all 8 models
│   │   └── predictor.py                # Inference + fusion
│   ├── utils/
│   │   └── preprocess.py               # Text preprocessing
│   └── saved_models/                   # Trained model weights
├── frontend/
│   └── index.html                      # Dashboard (standalone)
├── README.md                           # This file
├── SRS.md                              # Software Requirements
└── project_overview.md                 # Detailed overview
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | Quick start guide |
| [SRS.md](SRS.md) | Software Requirements Specification |
| [project_overview.md](project_overview.md) | Detailed project overview |

---

## Screenshots

### Dashboard
- Dark blue futuristic theme
- Live stock ticker cards with price changes
- Interactive 4-month price charts
- Prediction results with confidence scores

### Features
- Hover effects on all interactive elements
- Collapsible project details section
- Real-time market indicators
- Professional trading interface design

---

## Credits

**Developed by:** Homi Shivanshu Purbey

**Project:** Sentiment Analysis of Corporate Earning Calls for Stock Price Direction Prediction  
**Phase:** II - Deep Learning Models
