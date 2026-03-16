import React from "react";
import { Toaster } from "react-hot-toast";
import Dashboard from "./pages/Dashboard";
import Navbar from "./components/Navbar";

function App() {
  return (
    <div className="min-h-screen animated-gradient">
      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: "#1e293b",
            color: "#e2e8f0",
            border: "1px solid rgba(148,163,184,0.15)",
          },
        }}
      />
      <Navbar />
      <main className="container mx-auto px-4 py-6 max-w-7xl">
        <Dashboard />
      </main>
    </div>
  );
}

export default App;
