import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend } from "recharts";
import { Trophy, Cpu, RefreshCw } from "lucide-react";
import { getModelMetrics } from "../utils/api";

const MODEL_LABELS = {
  lstm: "LSTM",
  gru: "GRU",
  cnn_bilstm: "CNN-BiLSTM",
};

const MODEL_COLORS = {
  lstm: "#3b82f6",
  gru: "#8b5cf6",
  cnn_bilstm: "#06b6d4",
};

function ModelComparison() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchMetrics = () => {
    setLoading(true);
    setError(null);
    getModelMetrics()
      .then((res) => setMetrics(res.data))
      .catch((err) => setError(err.response?.data?.error || "Failed to load metrics"))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchMetrics();
  }, []);

  if (loading) {
    return (
      <div className="glass rounded-2xl p-6 border border-slate-700/30">
        <div className="flex items-center gap-2 mb-4">
          <Cpu className="w-5 h-5 text-blue-400" />
          <h3 className="text-base font-bold text-white">Model Comparison</h3>
        </div>
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-16 shimmer rounded-lg" />
          ))}
        </div>
      </div>
    );
  }

  if (error || !metrics?.trained) {
    return (
      <div className="glass rounded-2xl p-6 border border-slate-700/30">
        <div className="flex items-center gap-2 mb-4">
          <Cpu className="w-5 h-5 text-blue-400" />
          <h3 className="text-base font-bold text-white">Model Comparison</h3>
          <button onClick={fetchMetrics} className="ml-auto p-1.5 rounded-lg hover:bg-slate-700/50 transition-colors">
            <RefreshCw className="w-3.5 h-3.5 text-slate-400" />
          </button>
        </div>
        <div className="text-center py-8">
          <Cpu className="w-10 h-10 text-slate-600 mx-auto mb-3" />
          <p className="text-sm text-slate-400">No trained models found</p>
          <p className="text-xs text-slate-600 mt-1">
            Run <code className="px-1.5 py-0.5 rounded bg-slate-800 text-blue-400">python -m models.trainer</code> to train
          </p>
        </div>
      </div>
    );
  }

  const models = metrics.models || {};
  const bestModel = metrics.best_model;

  // Bar chart data
  const barData = Object.entries(models).map(([name, m]) => ({
    name: MODEL_LABELS[name] || name,
    Accuracy: parseFloat((m.accuracy * 100).toFixed(1)),
    F1: parseFloat((m.f1 * 100).toFixed(1)),
    "ROC-AUC": m.roc_auc ? parseFloat((m.roc_auc * 100).toFixed(1)) : 0,
  }));

  // Radar chart data
  const radarData = [
    { metric: "Accuracy", ...Object.fromEntries(Object.entries(models).map(([n, m]) => [MODEL_LABELS[n], m.accuracy * 100])) },
    { metric: "F1 Score", ...Object.fromEntries(Object.entries(models).map(([n, m]) => [MODEL_LABELS[n], m.f1 * 100])) },
    { metric: "ROC-AUC", ...Object.fromEntries(Object.entries(models).map(([n, m]) => [MODEL_LABELS[n], (m.roc_auc || 0) * 100])) },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.2 }}
      className="glass rounded-2xl p-5 border border-slate-700/30"
    >
      <div className="flex items-center gap-2 mb-4">
        <Cpu className="w-5 h-5 text-blue-400" />
        <h3 className="text-base font-bold text-white">Deep Learning Model Comparison</h3>
        <button onClick={fetchMetrics} className="ml-auto p-1.5 rounded-lg hover:bg-slate-700/50 transition-colors">
          <RefreshCw className="w-3.5 h-3.5 text-slate-400" />
        </button>
      </div>

      {/* Model Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-5">
        {Object.entries(models).map(([name, m]) => {
          const isBest = name === bestModel;
          return (
            <div
              key={name}
              className={`rounded-xl p-4 border transition-all ${
                isBest
                  ? "border-yellow-500/40 bg-yellow-500/5 glow-blue"
                  : "border-slate-700/30 bg-slate-800/30"
              }`}
            >
              <div className="flex items-center gap-2 mb-3">
                {isBest && <Trophy className="w-4 h-4 text-yellow-400" />}
                <span className="text-sm font-bold text-white">{MODEL_LABELS[name] || name}</span>
                {isBest && (
                  <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-yellow-500/20 border border-yellow-500/30 text-yellow-300 ml-auto">
                    BEST
                  </span>
                )}
              </div>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-[10px] text-slate-500">Accuracy</span>
                  <span className="text-xs font-bold text-green-400">{(m.accuracy * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full rounded-full" style={{ width: `${m.accuracy * 100}%`, backgroundColor: MODEL_COLORS[name] }} />
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-[10px] text-slate-500">F1 Score</span>
                  <span className="text-xs font-bold text-blue-400">{(m.f1 * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-[10px] text-slate-500">ROC-AUC</span>
                  <span className="text-xs font-bold text-purple-400">
                    {m.roc_auc ? `${(m.roc_auc * 100).toFixed(1)}%` : "N/A"}
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Bar Chart */}
        <div className="rounded-xl bg-slate-800/30 p-4">
          <p className="text-xs font-semibold text-slate-400 mb-3 uppercase tracking-wider">Performance Comparison</p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={barData} barGap={4}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 11 }} />
              <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} domain={[0, 100]} />
              <Tooltip
                contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: "8px", fontSize: "12px" }}
                labelStyle={{ color: "#e2e8f0" }}
              />
              <Bar dataKey="Accuracy" fill="#22c55e" radius={[4, 4, 0, 0]} />
              <Bar dataKey="F1" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              <Bar dataKey="ROC-AUC" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Radar Chart */}
        <div className="rounded-xl bg-slate-800/30 p-4">
          <p className="text-xs font-semibold text-slate-400 mb-3 uppercase tracking-wider">Model Radar</p>
          <ResponsiveContainer width="100%" height={220}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#334155" />
              <PolarAngleAxis dataKey="metric" tick={{ fill: "#94a3b8", fontSize: 10 }} />
              <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: "#64748b", fontSize: 9 }} />
              {Object.entries(MODEL_LABELS).map(([key, label]) => (
                <Radar key={key} name={label} dataKey={label} stroke={MODEL_COLORS[key]} fill={MODEL_COLORS[key]} fillOpacity={0.15} strokeWidth={2} />
              ))}
              <Legend wrapperStyle={{ fontSize: "11px" }} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </motion.div>
  );
}

export default ModelComparison;
