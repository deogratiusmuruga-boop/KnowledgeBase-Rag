import React, { useState } from "react";

import HomeScreen from "./screens/HomeScreen";
import HealthScreen from "./screens/HealthScreen";

function App() {
  const [screen, setScreen] = useState("home");

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#f8fafc",
        fontFamily: "'Pretendard', sans-serif",
        paddingBottom: "90px",
      }}
    >
      {/* Main Application Content */}

      {screen === "home" && <HomeScreen />}

      {screen === "health" && <HealthScreen />}

      {/* Bottom Navigation */}

      <nav
        style={{
          position: "fixed",
          bottom: 0,
          left: 0,
          right: 0,
          height: "72px",
          background: "#ffffff",
          borderTop: "1px solid #e2e8f0",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          gap: "12px",
          zIndex: 1000,
          boxShadow: "0 -4px 20px rgba(15, 23, 42, 0.06)",
        }}
      >
        <button
          onClick={() => setScreen("home")}
          style={{
            minWidth: "140px",
            padding: "12px 20px",
            borderRadius: "14px",
            border:
              screen === "home"
                ? "1px solid #2563eb"
                : "1px solid #e2e8f0",
            background:
              screen === "home" ? "#eff6ff" : "#ffffff",
            color:
              screen === "home" ? "#1d4ed8" : "#475569",
            fontWeight: "700",
            cursor: "pointer",
            fontSize: "0.95rem",
          }}
        >
          🏠 Home
        </button>

        <button
          onClick={() => setScreen("health")}
          style={{
            minWidth: "140px",
            padding: "12px 20px",
            borderRadius: "14px",
            border:
              screen === "health"
                ? "1px solid #2563eb"
                : "1px solid #e2e8f0",
            background:
              screen === "health" ? "#eff6ff" : "#ffffff",
            color:
              screen === "health" ? "#1d4ed8" : "#475569",
            fontWeight: "700",
            cursor: "pointer",
            fontSize: "0.95rem",
          }}
        >
          ❤️ Health
        </button>
      </nav>
    </div>
  );
}

export default App;