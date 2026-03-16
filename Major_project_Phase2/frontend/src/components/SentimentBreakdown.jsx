import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Brain, ThumbsUp, ThumbsDown, MessageSquare, ChevronDown, AlertTriangle } from "lucide-react";

function SentenceCard({ item, type }) {
  const isPositive = type === "positive";
  return (
    <div className={`px-3 py-2.5 rounded-lg border ${isPositive ? "border-green-500/20 bg-green-500/5" : "border-red-500/20 bg-red-500/5"}`}>
      <p className="text-xs text-slate-300 leading-relaxed line-clamp-2">{item.sentence}</p>
      <div className="flex gap-3 mt-1.5">
        {item.compound !== undefined && (
          <span className="text-[10px] text-slate-500">
            VADER: <span className={item.compound > 0 ? "text-green-400" : "text-red-400"}>{item.compound?.toFixed(3)}</span>
          </span>
        )}
        {item.polarity !== undefined && (
          <span className="text-[10px] text-slate-500">
            FinBERT: <span className={item.polarity > 0 ? "text-green-400" : "text-red-400"}>{item.polarity?.toFixed(3)}</span>
          </span>
        )}
      </div>
    </div>
  );
}

function MetricPill({ label, value, suffix = "", color = "text-slate-300" }) {
  return (
    <div className="glass rounded-lg px-3 py-2 text-center">
      <p className="text-[10px] text-slate-500 uppercase tracking-wider">{label}</p>
      <p className={`text-sm font-bold mt-0.5 ${color}`}>
        {typeof value === "number" ? value.toFixed(3) : value}{suffix}
      </p>
    </div>
  );
}

function SentimentBreakdown({ sentiment }) {
  const [showSentences, setShowSentences] = useState(false);

  if (!sentiment) return null;

  const {
    vader_compound_mean,
    vader_probability,
    finbert_polarity_mean,
    finbert_probability,
    finbert_available,
    finbert_error,
    sentiment_probability,
    n_sentences,
    pos_ratio,
    neg_ratio,
    top_positive_vader = [],
    top_negative_vader = [],
    top_positive_finbert = [],
    top_negative_finbert = [],
  } = sentiment;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.1 }}
      className="glass rounded-2xl p-5 border border-slate-700/30"
    >
      <div className="flex items-center gap-2 mb-4">
        <Brain className="w-5 h-5 text-purple-400" />
        <h3 className="text-base font-bold text-white">Sentiment Analysis</h3>
        <span className="text-[10px] px-2 py-0.5 rounded-full bg-purple-500/10 border border-purple-500/30 text-purple-300 ml-auto">
          {n_sentences} sentences
        </span>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-4">
        <MetricPill
          label="VADER Mean"
          value={vader_compound_mean}
          color={vader_compound_mean > 0 ? "text-green-400" : "text-red-400"}
        />
        <MetricPill
          label="VADER P(Up)"
          value={vader_probability}
          color={vader_probability >= 0.5 ? "text-green-400" : "text-red-400"}
        />
        {finbert_available ? (
          <>
            <MetricPill
              label="FinBERT Polarity"
              value={finbert_polarity_mean}
              color={finbert_polarity_mean > 0 ? "text-green-400" : "text-red-400"}
            />
            <MetricPill
              label="FinBERT P(Up)"
              value={finbert_probability}
              color={finbert_probability >= 0.5 ? "text-green-400" : "text-red-400"}
            />
          </>
        ) : (
          <div className="col-span-2 glass rounded-lg px-3 py-2 flex items-center gap-2">
            <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
            <span className="text-[10px] text-amber-400">
              FinBERT unavailable{finbert_error ? `: ${finbert_error}` : ""}
            </span>
          </div>
        )}
      </div>

      {/* Blended Score */}
      <div className="rounded-xl bg-gradient-to-r from-purple-500/10 to-blue-500/10 border border-purple-500/20 p-3 mb-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-[10px] text-slate-400 uppercase tracking-wider">Blended Sentiment Probability</p>
            <p className="text-xs text-slate-500 mt-0.5">
              {finbert_available ? "0.4 × VADER + 0.6 × FinBERT" : "VADER only"}
            </p>
          </div>
          <span className={`text-2xl font-bold ${sentiment_probability >= 0.5 ? "text-green-400" : "text-red-400"}`}>
            {(sentiment_probability * 100).toFixed(1)}%
          </span>
        </div>
        <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden mt-2">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${sentiment_probability * 100}%` }}
            transition={{ duration: 0.6 }}
            className={`h-full rounded-full ${sentiment_probability >= 0.5 ? "bg-gradient-to-r from-green-500 to-emerald-400" : "bg-gradient-to-r from-red-500 to-orange-400"}`}
          />
        </div>
      </div>

      {/* Sentiment Distribution */}
      <div className="flex gap-2 mb-4">
        <div className="flex-1 rounded-lg bg-green-500/10 border border-green-500/20 p-2 text-center">
          <ThumbsUp className="w-3.5 h-3.5 text-green-400 mx-auto" />
          <p className="text-xs font-bold text-green-400 mt-1">{(pos_ratio * 100).toFixed(1)}%</p>
          <p className="text-[9px] text-slate-500">Positive</p>
        </div>
        <div className="flex-1 rounded-lg bg-slate-500/10 border border-slate-500/20 p-2 text-center">
          <MessageSquare className="w-3.5 h-3.5 text-slate-400 mx-auto" />
          <p className="text-xs font-bold text-slate-400 mt-1">{((1 - pos_ratio - neg_ratio) * 100).toFixed(1)}%</p>
          <p className="text-[9px] text-slate-500">Neutral</p>
        </div>
        <div className="flex-1 rounded-lg bg-red-500/10 border border-red-500/20 p-2 text-center">
          <ThumbsDown className="w-3.5 h-3.5 text-red-400 mx-auto" />
          <p className="text-xs font-bold text-red-400 mt-1">{(neg_ratio * 100).toFixed(1)}%</p>
          <p className="text-[9px] text-slate-500">Negative</p>
        </div>
      </div>

      {/* Top Sentences Toggle */}
      <button
        onClick={() => setShowSentences(!showSentences)}
        className="w-full flex items-center justify-between px-3 py-2 rounded-lg hover:bg-slate-800/50 transition-colors"
      >
        <span className="text-xs font-semibold text-slate-400">Top Sentences</span>
        <ChevronDown className={`w-4 h-4 text-slate-500 transition-transform ${showSentences ? "rotate-180" : ""}`} />
      </button>

      <AnimatePresence>
        {showSentences && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-3">
              <div>
                <p className="text-[10px] font-semibold text-green-400 uppercase mb-2 flex items-center gap-1">
                  <ThumbsUp className="w-3 h-3" /> Most Positive
                </p>
                <div className="space-y-1.5">
                  {(top_positive_vader.length > 0 ? top_positive_vader : top_positive_finbert).map((s, i) => (
                    <SentenceCard key={i} item={s} type="positive" />
                  ))}
                </div>
              </div>
              <div>
                <p className="text-[10px] font-semibold text-red-400 uppercase mb-2 flex items-center gap-1">
                  <ThumbsDown className="w-3 h-3" /> Most Negative
                </p>
                <div className="space-y-1.5">
                  {(top_negative_vader.length > 0 ? top_negative_vader : top_negative_finbert).map((s, i) => (
                    <SentenceCard key={i} item={s} type="negative" />
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

export default SentimentBreakdown;
