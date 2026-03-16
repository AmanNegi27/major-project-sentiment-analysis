import axios from "axios";

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:5000";

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000,
  headers: { "Content-Type": "application/json" },
});

export const getCompanies = () => api.get("/api/companies");

export const analyzePrediction = (ticker, transcript, earningsDate) =>
  api.post("/api/analyze", {
    ticker,
    transcript,
    earnings_date: earningsDate,
  });

export const getModelMetrics = () => api.get("/api/model-metrics");

export const getPriceHistory = (ticker, days = 180) =>
  api.get("/api/price-history", { params: { ticker, days } });

export const triggerTraining = (tickers = "") =>
  api.post("/api/train", { tickers });

export default api;
