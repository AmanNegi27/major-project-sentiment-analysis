# StockSense AI - Run Guide

## 🚀 Quick Start

### Backend (Flask API)
```bash
cd backend
pip install -r requirements.txt
python app_minimal.py
```
API runs on: http://localhost:5000

### Test the API
```bash
# Health check
curl http://localhost:5000/api/health

# Run demo
python demo.py
```

### Frontend (React - requires Node.js)
```bash
cd frontend
npm install
npm start
```
Frontend runs on: http://localhost:3000

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | API status check |
| `/api/companies` | GET | List of 16 supported companies |
| `/api/analyze` | POST | Analyze earnings call transcript |
| `/api/model-metrics` | GET | Trained model performance |
| `/api/price-history` | GET | Historical price data |
| `/api/train` | POST | Trigger model training |

## 🧠 Model Performance

Trained on 10+ years of data for 16 companies:

- **LSTM**: 51.0% accuracy (BEST)
- **GRU**: 49.6% accuracy
- **CNN-BiLSTM**: 49.5% accuracy

## ⚡ Fusion Formula

```
Final Score = 0.4 × P(VADER Sentiment) + 0.6 × P(Deep Learning)
```

## 📝 Sample API Request

```json
POST /api/analyze
{
  "ticker": "AAPL",
  "transcript": "We had an excellent quarter with strong revenue growth...",
  "earnings_date": "2024-01-25"
}
```

## 🎯 Demo Results

The system successfully predicts stock direction:

- **AAPL**: UP (57.3% confidence)
- **TSLA**: UP (54.0% confidence) 
- **MSFT**: UP (60.1% confidence)

## 🔧 Components

### Sentiment Analysis
- **VADER**: Rule-based sentiment analysis
- **FinBERT**: Financial BERT model (optional, heavy)

### Deep Learning Models
- **LSTM**: Long Short-Term Memory with attention
- **GRU**: Gated Recurrent Unit with attention
- **CNN-BiLSTM**: Hybrid CNN + Bidirectional LSTM

### Features (28 technical indicators)
- RSI, MACD, Bollinger Bands, ATR, ADX
- Stochastic, OBV, VWAP, Momentum
- Calendar features, Z-scores

## 📁 Project Structure

```
Major_Project_Phase2/
├── backend/
│   ├── app_minimal.py      # Main API server
│   ├── config.py          # Company configurations
│   ├── sentiment/          # VADER + FinBERT
│   ├── models/            # DL models + trainer
│   ├── data/              # Fetcher + feature engineering
│   └── saved_models/      # Trained models
├── frontend/
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── pages/         # Dashboard
│   │   └── utils/         # API helpers
│   └── package.json
└── README.md
```

## 🎨 Frontend Features

- **Company Selector**: Searchable dropdown with market filters
- **Transcript Input**: Textarea with character counter
- **Prediction Results**: Animated UP/DOWN with probability
- **Sentiment Breakdown**: VADER scores, top sentences
- **Model Comparison**: Bar/Radar charts for DL models
- **Price Charts**: Interactive historical prices

## ⚠️ Notes

- FinBERT model download may cause initial delays
- System works with VADER-only for faster predictions
- Predictions are for educational purposes only
- Best model (LSTM) automatically selected

## 🎯 Next Steps

1. Install Node.js for frontend
2. Run `npm install && npm start` in frontend folder
3. Access full web UI at http://localhost:3000
4. Try different earnings call transcripts
5. Experiment with various companies

## 📞 Support

The system combines NLP sentiment analysis with deep learning to provide
earnings call-based stock direction predictions using a fusion approach.
