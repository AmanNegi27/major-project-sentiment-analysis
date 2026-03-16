import React from "react";
import { motion } from "framer-motion";
import { TrendingUp, TrendingDown, Sparkles, Brain, Cpu, Shuffle } from "lucide-react";

function ProbBadge({ label, value, icon: Icon, color }) {
  const isUp = value >= 0.5;
  const barWidth = Math.round(value * 100);
  return (
    <div className="glass rounded-xl p-4 flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <Icon className={`w-4 h-4 ${color}`} />
        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">{label}</span>
      </div>
      <div className="flex items-end justify-between">
        <div>
          <span className={`text-3xl font-bold ${isUp ? "text-green-400" : "text-red-400"}`}>
            {(value * 100).toFixed(1)}%
          </span>
          <div className="flex items-center gap-1.5 mt-1">
            {isUp ? (
              <TrendingUp className="w-3.5 h-3.5 text-green-400" />
            ) : (
              <TrendingDown className="w-3.5 h-3.5 text-red-400" />
            )}
            <span className={`text-xs font-semibold ${isUp ? "text-green-400" : "text-red-400"}`}>
              {isUp ? "BULLISH" : "BEARISH"}
            </span>
          </div>
        </div>
      </div>
      <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${barWidth}%` }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className={`h-full rounded-full ${isUp ? "bg-gradient-to-r from-green-500 to-emerald-400" : "bg-gradient-to-r from-red-500 to-orange-400"}`}
        />
      </div>
    </div>
  );
}

function PredictionResult({ result }) {
  if (!result) return null;

  const { fusion, sentiment, deep_learning, components } = result;
  const isUp = fusion.probability >= 0.5;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="space-y-4"
    >
      {/* Main Fusion Result */}
      <div className={`rounded-2xl p-6 border ${isUp ? "border-green-500/30 glow-green" : "border-red-500/30 glow-red"} glass`}>
        <div className="flex items-center gap-2 mb-4">
          <Sparkles className={`w-5 h-5 ${isUp ? "text-green-400" : "text-red-400"}`} />
          <h3 className="text-lg font-bold text-white">Fusion Prediction</h3>
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-slate-700/50 text-slate-400 ml-auto">
            {fusion.formula}
          </span>
        </div>

        <div className="flex items-center justify-center gap-6 py-4">
          <div className="text-center">
            {isUp ? (
              <TrendingUp className="w-16 h-16 text-green-400 mx-auto animate-float" />
            ) : (
              <TrendingDown className="w-16 h-16 text-red-400 mx-auto animate-float" />
            )}
            <p className={`text-4xl font-extrabold mt-2 ${isUp ? "text-green-400" : "text-red-400"}`}>
              {fusion.direction}
            </p>
            <p className={`text-5xl font-black mt-1 ${isUp ? "text-green-300" : "text-red-300"}`}>
              {(fusion.probability * 100).toFixed(1)}%
            </p>
            <p className="text-xs text-slate-500 mt-2">Confidence Score</p>
          </div>
        </div>

        {deep_learning?.last_close && (
          <p className="text-center text-xs text-slate-500">
            Last Close: <span className="text-slate-300 font-medium">${deep_learning.last_close.toFixed(2)}</span>
            {deep_learning.model_name && (
              <span className="ml-2">| Best Model: <span className="text-blue-400 font-medium">{deep_learning.model_name.toUpperCase()}</span></span>
            )}
          </p>
        )}
      </div>

      {/* Component Probabilities */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <ProbBadge
          label="Sentiment Score"
          value={components.sentiment_prob}
          icon={Brain}
          color="text-purple-400"
        />
        <ProbBadge
          label="Deep Learning"
          value={components.dl_prob}
          icon={Cpu}
          color="text-blue-400"
        />
        <ProbBadge
          label="Fusion Score"
          value={components.fusion_prob}
          icon={Shuffle}
          color="text-cyan-400"
        />
      </div>

      {/* Breakdown Table */}
      <div className="glass rounded-xl overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-700/50">
              <th className="text-left px-4 py-3 text-xs font-semibold text-slate-400 uppercase">Component</th>
              <th className="text-right px-4 py-3 text-xs font-semibold text-slate-400 uppercase">Probability</th>
              <th className="text-right px-4 py-3 text-xs font-semibold text-slate-400 uppercase">Weight</th>
              <th className="text-right px-4 py-3 text-xs font-semibold text-slate-400 uppercase">Contribution</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-slate-800/50">
              <td className="px-4 py-3 flex items-center gap-2">
                <Brain className="w-3.5 h-3.5 text-purple-400" />
                <span className="text-slate-300">Sentiment</span>
              </td>
              <td className="px-4 py-3 text-right font-mono text-slate-300">{(components.sentiment_prob * 100).toFixed(2)}%</td>
              <td className="px-4 py-3 text-right font-mono text-slate-500">{(fusion.sentiment_weight * 100).toFixed(0)}%</td>
              <td className="px-4 py-3 text-right font-mono text-purple-400">{(components.sentiment_prob * fusion.sentiment_weight * 100).toFixed(2)}%</td>
            </tr>
            <tr>
              <td className="px-4 py-3 flex items-center gap-2">
                <Cpu className="w-3.5 h-3.5 text-blue-400" />
                <span className="text-slate-300">Deep Learning</span>
              </td>
              <td className="px-4 py-3 text-right font-mono text-slate-300">{(components.dl_prob * 100).toFixed(2)}%</td>
              <td className="px-4 py-3 text-right font-mono text-slate-500">{(fusion.dl_weight * 100).toFixed(0)}%</td>
              <td className="px-4 py-3 text-right font-mono text-blue-400">{(components.dl_prob * fusion.dl_weight * 100).toFixed(2)}%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </motion.div>
  );
}

export default PredictionResult;
