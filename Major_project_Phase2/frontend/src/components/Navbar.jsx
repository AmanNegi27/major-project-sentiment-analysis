import React from "react";
import { TrendingUp, Brain, BarChart3 } from "lucide-react";

function Navbar() {
  return (
    <nav className="glass sticky top-0 z-50 border-b border-slate-700/50">
      <div className="container mx-auto px-4 max-w-7xl">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                <TrendingUp className="w-5 h-5 text-white" />
              </div>
              <div className="absolute -top-1 -right-1 w-3 h-3 rounded-full bg-green-400 pulse-ring" />
            </div>
            <div>
              <h1 className="text-lg font-bold gradient-text">StockSense AI</h1>
              <p className="text-[10px] text-slate-400 -mt-0.5">
                Earnings Call NLP + Deep Learning
              </p>
            </div>
          </div>

          <div className="flex items-center gap-6">
            <div className="hidden md:flex items-center gap-2 text-xs text-slate-400">
              <Brain className="w-3.5 h-3.5 text-purple-400" />
              <span>VADER + FinBERT</span>
            </div>
            <div className="hidden md:flex items-center gap-2 text-xs text-slate-400">
              <BarChart3 className="w-3.5 h-3.5 text-blue-400" />
              <span>LSTM / GRU / CNN-BiLSTM</span>
            </div>
            <div className="px-3 py-1.5 rounded-full bg-gradient-to-r from-blue-600/20 to-purple-600/20 border border-blue-500/30">
              <span className="text-xs font-semibold text-blue-300">Phase 2</span>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
