import React, { useState } from "react";
import PredictionForm from "./components/PredictionForm.jsx";
import ResultCard from "./components/ResultCard.jsx";
import FeatureImportanceChart from "./components/FeatureImportanceChart.jsx";

export default function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  return (
    <div style={styles.page}>
      <header style={styles.header}>
        <div style={styles.headerInner}>
          <div>
            <h1 style={styles.title}>Telecom Churn Predictor</h1>
            <p style={styles.subtitle}>End-to-end ML pipeline — Logistic Regression on Telco Customer Churn dataset</p>
          </div>
          <div style={styles.badge}>Live Model</div>
        </div>
      </header>

      <main style={styles.main}>
        <div style={styles.left}>
          <PredictionForm onResult={setResult} loading={loading} setLoading={setLoading} />
        </div>

        <div style={styles.right}>
          {result ? (
            <ResultCard result={result} />
          ) : (
            <div style={styles.placeholder}>
              <p style={styles.placeholderText}>Fill in the form and click<br /><strong>Predict Churn</strong> to see the result.</p>
            </div>
          )}
          <FeatureImportanceChart />
        </div>
      </main>
    </div>
  );
}

const styles = {
  page: { minHeight: "100vh", display: "flex", flexDirection: "column" },
  header: { background: "#1e293b", borderBottom: "1px solid #334155", padding: "16px 0" },
  headerInner: { maxWidth: 1200, margin: "0 auto", padding: "0 24px", display: "flex", justifyContent: "space-between", alignItems: "center" },
  title: { fontSize: 22, fontWeight: 700, color: "#f1f5f9" },
  subtitle: { fontSize: 13, color: "#94a3b8", marginTop: 2 },
  badge: { background: "#22c55e22", color: "#22c55e", border: "1px solid #22c55e55", borderRadius: 20, padding: "4px 12px", fontSize: 12, fontWeight: 600 },
  main: { flex: 1, maxWidth: 1200, margin: "0 auto", padding: 24, width: "100%", display: "grid", gridTemplateColumns: "1fr 380px", gap: 24, alignItems: "start" },
  left: { minWidth: 0 },
  right: { display: "flex", flexDirection: "column", gap: 20, position: "sticky", top: 24 },
  placeholder: { background: "#1e293b", borderRadius: 12, padding: 40, border: "2px dashed #334155", display: "flex", alignItems: "center", justifyContent: "center", minHeight: 160 },
  placeholderText: { color: "#64748b", textAlign: "center", lineHeight: 1.8, fontSize: 14 },
};
