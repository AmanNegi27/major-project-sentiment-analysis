import React, { useState } from "react";
import { FileText, Calendar, Zap, AlertCircle } from "lucide-react";

function TranscriptInput({ transcript, setTranscript, earningsDate, setEarningsDate, onAnalyze, loading, selectedTicker }) {
  const [charCount, setCharCount] = useState(0);

  const handleChange = (e) => {
    const val = e.target.value;
    setTranscript(val);
    setCharCount(val.length);
  };

  const canSubmit = selectedTicker && transcript.trim().length > 50 && earningsDate && !loading;

  return (
    <div className="space-y-4">
      {/* Transcript Input */}
      <div>
        <label className="block text-xs font-semibold text-slate-400 mb-2 uppercase tracking-wider">
          Earnings Call Transcript
        </label>
        <div className="relative">
          <textarea
            value={transcript}
            onChange={handleChange}
            placeholder="Paste the full earnings call transcript here...&#10;&#10;The NLP engine will analyze sentiment using VADER and FinBERT to extract bullish/bearish signals from the management's commentary."
            rows={10}
            className="w-full glass rounded-xl px-4 py-3.5 text-sm text-white placeholder-slate-600 border border-slate-700/50 focus:border-blue-500/40 focus:outline-none resize-none transition-all duration-200 hover:border-slate-600/50"
          />
          <div className="absolute bottom-3 right-3 flex items-center gap-3">
            <span className={`text-[10px] ${charCount > 100 ? "text-green-400" : "text-slate-600"}`}>
              {charCount.toLocaleString()} chars
            </span>
            {charCount > 0 && charCount < 50 && (
              <span className="text-[10px] text-amber-400 flex items-center gap-1">
                <AlertCircle className="w-3 h-3" />
                Min 50 chars
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Date + Analyze Row */}
      <div className="flex flex-col sm:flex-row gap-3">
        <div className="flex-1">
          <label className="block text-xs font-semibold text-slate-400 mb-2 uppercase tracking-wider">
            Earnings Call Date
          </label>
          <div className="relative">
            <Calendar className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
            <input
              type="date"
              value={earningsDate}
              onChange={(e) => setEarningsDate(e.target.value)}
              className="w-full glass rounded-xl pl-10 pr-4 py-3.5 text-sm text-white border border-slate-700/50 focus:border-blue-500/40 focus:outline-none transition-all duration-200 hover:border-slate-600/50"
            />
          </div>
        </div>

        <div className="flex items-end">
          <button
            onClick={onAnalyze}
            disabled={!canSubmit}
            className={`w-full sm:w-auto px-8 py-3.5 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all duration-300 ${
              canSubmit
                ? "bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 hover:scale-[1.02] active:scale-[0.98]"
                : "bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700/50"
            }`}
          >
            <Zap className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
            {loading ? "Analyzing..." : "Analyze & Predict"}
          </button>
        </div>
      </div>

      {/* Helper hints */}
      {!selectedTicker && (
        <p className="text-xs text-amber-400/80 flex items-center gap-1.5">
          <AlertCircle className="w-3 h-3" />
          Select a company above to continue
        </p>
      )}
    </div>
  );
}

export default TranscriptInput;
