import React, { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

export default function FeatureImportanceChart() {
  const [data, setData] = useState([]);
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {
    fetch("/api/model-info")
      .then((r) => r.json())
      .then(setModelInfo)
      .catch(() => {});
  }, []);

  // Parse feature importances from model-info CSV via a separate endpoint we'll add
  // For now, display model metadata
  if (!modelInfo) return null;

  const { best_model, test_metrics } = modelInfo;
  const metrics = [
    { name: "ROC-AUC",   value: test_metrics?.roc_auc },
    { name: "F1",        value: test_metrics?.f1 },
    { name: "Precision", value: test_metrics?.precision },
    { name: "Recall",    value: test_metrics?.recall },
  ].filter((m) => m.value != null);

  const COLORS = ["#6366f1", "#818cf8", "#a5b4fc", "#c7d2fe"];

  return (
    <div style={styles.card}>
      <h3 style={styles.title}>Model Performance</h3>
      <p style={styles.sub}>Best model: <strong style={{ color: "var(--brand-light)" }}>{best_model}</strong></p>

      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={metrics} layout="vertical" margin={{ left: 10, right: 20 }}>
          <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 11, fill: "#94a3b8" }} />
          <YAxis type="category" dataKey="name" tick={{ fontSize: 12, fill: "#e2e8f0" }} width={70} />
          <Tooltip
            formatter={(v) => v.toFixed(4)}
            contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 6 }}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {metrics.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

const styles = {
  card: { background: "var(--surface)", borderRadius: 12, padding: 20, border: "1px solid var(--border)" },
  title: { fontSize: 13, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--brand-light)", marginBottom: 4 },
  sub: { fontSize: 13, color: "var(--muted)", marginBottom: 16 },
};
