import React from "react";
import { motion } from "framer-motion";
import { Loader2 } from "lucide-react";

function LoadingSpinner({ message = "Analyzing...", submessage = "" }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="flex flex-col items-center justify-center py-16 gap-4"
    >
      <div className="relative">
        <div className="w-16 h-16 rounded-full border-2 border-blue-500/20" />
        <div className="absolute inset-0 w-16 h-16 rounded-full border-2 border-transparent border-t-blue-500 animate-spin" />
        <div className="absolute inset-2 w-12 h-12 rounded-full border-2 border-transparent border-t-purple-500 animate-spin" style={{ animationDuration: "1.5s", animationDirection: "reverse" }} />
        <Loader2 className="absolute inset-0 m-auto w-5 h-5 text-blue-400 animate-pulse" />
      </div>
      <div className="text-center">
        <p className="text-sm font-medium text-slate-300">{message}</p>
        {submessage && (
          <p className="text-xs text-slate-500 mt-1">{submessage}</p>
        )}
      </div>
    </motion.div>
  );
}

export default LoadingSpinner;
