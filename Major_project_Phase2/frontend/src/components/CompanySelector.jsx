import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Building2, Search, Globe, ChevronDown } from "lucide-react";
import { getCompanies } from "../utils/api";

const MARKET_COLORS = {
  NASDAQ: "text-blue-400 bg-blue-400/10 border-blue-400/30",
  NYSE: "text-emerald-400 bg-emerald-400/10 border-emerald-400/30",
  NSE: "text-orange-400 bg-orange-400/10 border-orange-400/30",
};

function CompanySelector({ selectedTicker, onSelect }) {
  const [companies, setCompanies] = useState([]);
  const [search, setSearch] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const [filter, setFilter] = useState("ALL");

  useEffect(() => {
    getCompanies()
      .then((res) => setCompanies(res.data.companies || []))
      .catch(() => {
        setCompanies([
          { ticker: "AAPL", name: "Apple Inc.", market: "NASDAQ", sector: "Technology" },
          { ticker: "MSFT", name: "Microsoft Corporation", market: "NASDAQ", sector: "Technology" },
          { ticker: "AMZN", name: "Amazon.com Inc.", market: "NASDAQ", sector: "Consumer Cyclical" },
          { ticker: "NVDA", name: "NVIDIA Corporation", market: "NASDAQ", sector: "Technology" },
          { ticker: "GOOGL", name: "Alphabet Inc.", market: "NASDAQ", sector: "Communication Services" },
          { ticker: "META", name: "Meta Platforms Inc.", market: "NASDAQ", sector: "Communication Services" },
          { ticker: "TSLA", name: "Tesla Inc.", market: "NASDAQ", sector: "Consumer Cyclical" },
          { ticker: "AVGO", name: "Broadcom Inc.", market: "NASDAQ", sector: "Technology" },
          { ticker: "COST", name: "Costco Wholesale Corp.", market: "NASDAQ", sector: "Consumer Defensive" },
          { ticker: "NFLX", name: "Netflix Inc.", market: "NASDAQ", sector: "Communication Services" },
          { ticker: "DELL", name: "Dell Technologies Inc.", market: "NYSE", sector: "Technology" },
          { ticker: "RELIANCE.NS", name: "Reliance Industries", market: "NSE", sector: "Energy" },
          { ticker: "TCS.NS", name: "Tata Consultancy Services", market: "NSE", sector: "Technology" },
          { ticker: "INFY.NS", name: "Infosys Limited", market: "NSE", sector: "Technology" },
          { ticker: "HDFCBANK.NS", name: "HDFC Bank Limited", market: "NSE", sector: "Financial Services" },
          { ticker: "WIPRO.NS", name: "Wipro Limited", market: "NSE", sector: "Technology" },
        ]);
      });
  }, []);

  const selected = companies.find((c) => c.ticker === selectedTicker);

  const filtered = companies.filter((c) => {
    const matchSearch =
      c.ticker.toLowerCase().includes(search.toLowerCase()) ||
      c.name.toLowerCase().includes(search.toLowerCase());
    const matchFilter = filter === "ALL" || c.market === filter;
    return matchSearch && matchFilter;
  });

  const markets = ["ALL", ...new Set(companies.map((c) => c.market))];

  return (
    <div className="relative">
      <label className="block text-xs font-semibold text-slate-400 mb-2 uppercase tracking-wider">
        Select Company
      </label>

      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full glass rounded-xl px-4 py-3.5 flex items-center justify-between hover:border-blue-500/40 transition-all duration-200 group"
      >
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-blue-500/20 to-purple-500/20 flex items-center justify-center border border-blue-500/20">
            <Building2 className="w-4 h-4 text-blue-400" />
          </div>
          {selected ? (
            <div className="text-left">
              <div className="flex items-center gap-2">
                <span className="font-semibold text-white text-sm">{selected.ticker}</span>
                <span className={`text-[10px] px-1.5 py-0.5 rounded-full border ${MARKET_COLORS[selected.market] || "text-slate-400"}`}>
                  {selected.market}
                </span>
              </div>
              <p className="text-xs text-slate-400">{selected.name}</p>
            </div>
          ) : (
            <span className="text-slate-500 text-sm">Choose a company...</span>
          )}
        </div>
        <ChevronDown className={`w-4 h-4 text-slate-400 transition-transform duration-200 ${isOpen ? "rotate-180" : ""}`} />
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -8, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8, scale: 0.98 }}
            transition={{ duration: 0.15 }}
            className="absolute z-50 w-full mt-2 glass rounded-xl overflow-hidden shadow-2xl shadow-black/50"
          >
            <div className="p-3 border-b border-slate-700/50">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-500" />
                <input
                  type="text"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search ticker or name..."
                  className="w-full bg-slate-800/50 rounded-lg pl-9 pr-3 py-2 text-sm text-white placeholder-slate-500 border border-slate-700/50 focus:border-blue-500/50 focus:outline-none transition-colors"
                  autoFocus
                />
              </div>

              <div className="flex gap-1.5 mt-2">
                {markets.map((m) => (
                  <button
                    key={m}
                    onClick={() => setFilter(m)}
                    className={`text-[10px] px-2.5 py-1 rounded-full border transition-all ${
                      filter === m
                        ? "bg-blue-500/20 border-blue-500/40 text-blue-300"
                        : "border-slate-700/50 text-slate-500 hover:text-slate-300"
                    }`}
                  >
                    {m === "ALL" ? (
                      <span className="flex items-center gap-1">
                        <Globe className="w-2.5 h-2.5" /> All
                      </span>
                    ) : (
                      m
                    )}
                  </button>
                ))}
              </div>
            </div>

            <div className="max-h-64 overflow-y-auto">
              {filtered.map((c) => (
                <button
                  key={c.ticker}
                  onClick={() => {
                    onSelect(c.ticker);
                    setIsOpen(false);
                    setSearch("");
                  }}
                  className={`w-full px-4 py-2.5 flex items-center gap-3 hover:bg-slate-700/30 transition-colors text-left ${
                    c.ticker === selectedTicker ? "bg-blue-500/10 border-l-2 border-blue-500" : ""
                  }`}
                >
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-sm text-white">{c.ticker}</span>
                      <span className={`text-[9px] px-1.5 py-0.5 rounded-full border ${MARKET_COLORS[c.market] || "text-slate-400"}`}>
                        {c.market}
                      </span>
                    </div>
                    <p className="text-xs text-slate-500">{c.name}</p>
                  </div>
                  <span className="text-[10px] text-slate-600">{c.sector}</span>
                </button>
              ))}
              {filtered.length === 0 && (
                <p className="px-4 py-6 text-center text-xs text-slate-500">No companies found</p>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {isOpen && (
        <div className="fixed inset-0 z-40" onClick={() => setIsOpen(false)} />
      )}
    </div>
  );
}

export default CompanySelector;
