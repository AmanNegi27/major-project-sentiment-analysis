import React, { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import toast from "react-hot-toast";
import { Activity, TrendingUp, Brain, Cpu, Sparkles } from "lucide-react";

import CompanySelector from "../components/CompanySelector";
import TranscriptInput from "../components/TranscriptInput";
import PredictionResult from "../components/PredictionResult";
import SentimentBreakdown from "../components/SentimentBreakdown";
import ModelComparison from "../components/ModelComparison";
import PriceChart from "../components/PriceChart";
import LoadingSpinner from "../components/LoadingSpinner";
import { analyzePrediction } from "../utils/api";

function StatCard({ icon: Icon, label, value, color }) {
  return (
    <div className="glass rounded-xl p-3 flex items-center gap-3">
      <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${color}`}>
        <Icon className="w-4 h-4 text-white" />
      </div>
      <div>
        <p className="text-[10px] text-slate-500 uppercase tracking-wider">{label}</p>
        <p className="text-sm font-bold text-white">{value}</p>
      </div>
    </div>
  );
}

function Dashboard() {
  const [selectedTicker, setSelectedTicker] = useState("");
  const [transcript, setTranscript] = useState("");
  const [earningsDate, setEarningsDate] = useState(
    new Date().toISOString().split("T")[0]
  );
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("prediction");

  const handleAnalyze = useCallback(async () => {
    if (!selectedTicker || !transcript.trim() || !earningsDate) {
      toast.error("Please fill in all fields");
      return;
    }

    setLoading(true);
    setResult(null);
    const loadingToast = toast.loading("Analyzing transcript & running predictions...");

    try {
      const response = await analyzePrediction(
        selectedTicker,
        transcript,
        earningsDate
      );
      setResult(response.data);
      setActiveTab("prediction");
      toast.success(
        `Prediction complete: ${response.data.fusion?.direction || "N/A"}`,
        { id: loadingToast }
      );
    } catch (err) {
      const msg =
        err.response?.data?.error || err.message || "Analysis failed";
      toast.error(msg, { id: loadingToast });
    } finally {
      setLoading(false);
    }
  }, [selectedTicker, transcript, earningsDate]);

  const tabs = [
    { id: "prediction", label: "Prediction", icon: Sparkles },
    { id: "sentiment", label: "Sentiment", icon: Brain },
    { id: "models", label: "DL Models", icon: Cpu },
    { id: "chart", label: "Price Chart", icon: TrendingUp },
  ];

  return (
    <div className="space-y-6">
      {/* Hero Section */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center py-4"
      >
        <h2 className="text-2xl md:text-3xl font-extrabold gradient-text">
          Earnings Call Stock Predictor
        </h2>
        <p className="text-sm text-slate-400 mt-2 max-w-xl mx-auto">
          Paste an earnings call transcript, select a company, and get an AI-powered
          prediction combining NLP sentiment with deep learning models.
        </p>
      </motion.div>

      {/* Quick Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard icon={Activity} label="Companies" value="16" color="bg-blue-500/20" />
        <StatCard icon={Brain} label="NLP Models" value="VADER + FinBERT" color="bg-purple-500/20" />
        <StatCard icon={Cpu} label="DL Models" value="LSTM / GRU / CNN" color="bg-cyan-500/20" />
        <StatCard icon={Sparkles} label="Fusion" value="0.4S + 0.6D" color="bg-amber-500/20" />
      </div>

      {/* Input Section */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="glass rounded-2xl p-5 border border-slate-700/30"
      >
        <div className="space-y-4">
          <CompanySelector
            selectedTicker={selectedTicker}
            onSelect={setSelectedTicker}
          />
          <TranscriptInput
            transcript={transcript}
            setTranscript={setTranscript}
            earningsDate={earningsDate}
            setEarningsDate={setEarningsDate}
            onAnalyze={handleAnalyze}
            loading={loading}
            selectedTicker={selectedTicker}
          />
        </div>
      </motion.div>

      {/* Loading State */}
      <AnimatePresence>
        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <LoadingSpinner
              message="Running NLP + Deep Learning Analysis..."
              submessage={`Analyzing ${selectedTicker} earnings call with VADER, FinBERT, and the best DL model`}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Results Section */}
      {result && !loading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="space-y-4"
        >
          {/* Tabs */}
          <div className="flex gap-1 p-1 glass rounded-xl">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-xs font-semibold transition-all duration-200 ${
                  activeTab === tab.id
                    ? "bg-gradient-to-r from-blue-600/30 to-purple-600/30 text-white border border-blue-500/30"
                    : "text-slate-500 hover:text-slate-300 hover:bg-slate-800/50"
                }`}
              >
                <tab.icon className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">{tab.label}</span>
              </button>
            ))}
          </div>

          {/* Tab Content */}
          <AnimatePresence mode="wait">
            {activeTab === "prediction" && (
              <motion.div
                key="prediction"
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 10 }}
                transition={{ duration: 0.2 }}
              >
                <PredictionResult result={result} />
              </motion.div>
            )}

            {activeTab === "sentiment" && (
              <motion.div
                key="sentiment"
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 10 }}
                transition={{ duration: 0.2 }}
              >
                <SentimentBreakdown sentiment={result?.sentiment} />
              </motion.div>
            )}

            {activeTab === "models" && (
              <motion.div
                key="models"
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 10 }}
                transition={{ duration: 0.2 }}
              >
                <ModelComparison />
              </motion.div>
            )}

            {activeTab === "chart" && (
              <motion.div
                key="chart"
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 10 }}
                transition={{ duration: 0.2 }}
              >
                <PriceChart ticker={selectedTicker} />
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      )}

      {/* Always-visible Model Comparison and Price Chart (before first analysis) */}
      {!result && !loading && (
        <div className="space-y-4">
          {selectedTicker && <PriceChart ticker={selectedTicker} />}
          <ModelComparison />
        </div>
      )}

      {/* Footer */}
      <div className="text-center py-6 border-t border-slate-800/50">
        <p className="text-[10px] text-slate-600">
          StockSense AI Phase 2 — VADER + FinBERT Sentiment | LSTM + GRU + CNN-BiLSTM Deep Learning |
          Fusion: 0.4 × Sentiment + 0.6 × DL
        </p>
        <p className="text-[10px] text-slate-700 mt-1">
          Predictions are for educational purposes only. Not financial advice.
        </p>
      </div>
    </div>
  );
}

export default Dashboard;
