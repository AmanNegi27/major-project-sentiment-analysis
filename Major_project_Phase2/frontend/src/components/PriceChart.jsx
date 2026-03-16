import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { LineChart as LineChartIcon, RefreshCw } from "lucide-react";
import { getPriceHistory } from "../utils/api";

const TIME_RANGES = [
  { label: "1M", days: 30 },
  { label: "3M", days: 90 },
  { label: "6M", days: 180 },
  { label: "1Y", days: 365 },
  { label: "2Y", days: 730 },
];

function PriceChart({ ticker }) {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [range, setRange] = useState(180);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!ticker) return;
    setLoading(true);
    setError(null);
    getPriceHistory(ticker, range)
      .then((res) => {
        const priceData = (res.data.data || []).map((d) => ({
          ...d,
          date: d.date,
          price: parseFloat(d.adj_close?.toFixed(2) || d.close?.toFixed(2) || 0),
        }));
        setData(priceData);
      })
      .catch((err) => setError(err.response?.data?.error || "Failed to load prices"))
      .finally(() => setLoading(false));
  }, [ticker, range]);

  if (!ticker) return null;

  const priceChange = data.length > 1 ? data[data.length - 1].price - data[0].price : 0;
  const priceChangePct = data.length > 1 ? ((priceChange / data[0].price) * 100).toFixed(2) : 0;
  const isPositive = priceChange >= 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.15 }}
      className="glass rounded-2xl p-5 border border-slate-700/30"
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <LineChartIcon className="w-5 h-5 text-cyan-400" />
          <h3 className="text-base font-bold text-white">{ticker} Price History</h3>
          {data.length > 0 && (
            <div className="flex items-center gap-2 ml-3">
              <span className="text-lg font-bold text-white">${data[data.length - 1]?.price}</span>
              <span className={`text-xs font-semibold px-1.5 py-0.5 rounded ${isPositive ? "bg-green-500/10 text-green-400" : "bg-red-500/10 text-red-400"}`}>
                {isPositive ? "+" : ""}{priceChangePct}%
              </span>
            </div>
          )}
        </div>

        <div className="flex items-center gap-1">
          {TIME_RANGES.map((r) => (
            <button
              key={r.days}
              onClick={() => setRange(r.days)}
              className={`text-[10px] px-2.5 py-1 rounded-lg transition-all ${
                range === r.days
                  ? "bg-blue-500/20 border border-blue-500/40 text-blue-300"
                  : "text-slate-500 hover:text-slate-300 hover:bg-slate-800/50"
              }`}
            >
              {r.label}
            </button>
          ))}
          <button
            onClick={() => { setLoading(true); getPriceHistory(ticker, range).then(res => setData((res.data.data || []).map(d => ({ ...d, date: d.date, price: parseFloat(d.adj_close?.toFixed(2) || 0) })))).finally(() => setLoading(false)); }}
            className="p-1.5 rounded-lg hover:bg-slate-700/50 transition-colors ml-1"
          >
            <RefreshCw className={`w-3.5 h-3.5 text-slate-400 ${loading ? "animate-spin" : ""}`} />
          </button>
        </div>
      </div>

      {error ? (
        <div className="text-center py-8">
          <p className="text-sm text-red-400">{error}</p>
        </div>
      ) : loading ? (
        <div className="h-[250px] shimmer rounded-lg" />
      ) : (
        <ResponsiveContainer width="100%" height={250}>
          <AreaChart data={data}>
            <defs>
              <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={isPositive ? "#22c55e" : "#ef4444"} stopOpacity={0.3} />
                <stop offset="95%" stopColor={isPositive ? "#22c55e" : "#ef4444"} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="date"
              tick={{ fill: "#64748b", fontSize: 10 }}
              tickFormatter={(val) => {
                const d = new Date(val);
                return `${d.getMonth() + 1}/${d.getDate()}`;
              }}
              interval="preserveStartEnd"
            />
            <YAxis
              tick={{ fill: "#64748b", fontSize: 10 }}
              domain={["auto", "auto"]}
              tickFormatter={(val) => `$${val}`}
            />
            <Tooltip
              contentStyle={{
                background: "#1e293b",
                border: "1px solid #334155",
                borderRadius: "8px",
                fontSize: "12px",
              }}
              labelStyle={{ color: "#e2e8f0" }}
              formatter={(value) => [`$${value}`, "Price"]}
            />
            <Area
              type="monotone"
              dataKey="price"
              stroke={isPositive ? "#22c55e" : "#ef4444"}
              fill="url(#priceGradient)"
              strokeWidth={2}
            />
          </AreaChart>
        </ResponsiveContainer>
      )}
    </motion.div>
  );
}

export default PriceChart;
