import React from "react";

const BAND_CONFIG = {
  low:    { color: "var(--success)", emoji: "🟢", label: "Low Risk" },
  medium: { color: "var(--warn)",    emoji: "🟡", label: "Medium Risk" },
  high:   { color: "var(--danger)",  emoji: "🔴", label: "High Risk" },
};

export default function ResultCard({ result }) {
  if (!result) return null;

  const { churn_probability, prediction, risk_band } = result;
  const band = BAND_CONFIG[risk_band] ?? BAND_CONFIG.medium;
  const pct = (churn_probability * 100).toFixed(1);

  return (
    <div style={{ ...styles.card, borderColor: band.color }}>
      <div style={styles.header}>
        <span style={styles.emoji}>{band.emoji}</span>
        <div>
          <p style={styles.bandLabel}>{band.label}</p>
          <p style={{ ...styles.prediction, color: prediction === "Yes" ? "var(--danger)" : "var(--success)" }}>
            Churn: <strong>{prediction}</strong>
          </p>
        </div>
      </div>

      <div style={styles.probRow}>
        <span style={styles.probLabel}>Churn Probability</span>
        <span style={{ ...styles.probValue, color: band.color }}>{pct}%</span>
      </div>

      <div style={styles.barTrack}>
        <div style={{ ...styles.barFill, width: `${pct}%`, background: band.color }} />
      </div>

      <p style={styles.hint}>
        {risk_band === "low"    && "This customer is unlikely to churn. Focus on upsell opportunities."}
        {risk_band === "medium" && "Moderate risk. Consider a proactive retention offer."}
        {risk_band === "high"   && "High churn risk! Immediate retention action recommended."}
      </p>
    </div>
  );
}

const styles = {
  card: { background: "var(--surface)", borderRadius: 12, padding: 24, border: "2px solid", display: "flex", flexDirection: "column", gap: 16 },
  header: { display: "flex", alignItems: "center", gap: 16 },
  emoji: { fontSize: 40 },
  bandLabel: { fontSize: 18, fontWeight: 700 },
  prediction: { fontSize: 14, marginTop: 2 },
  probRow: { display: "flex", justifyContent: "space-between", alignItems: "baseline" },
  probLabel: { fontSize: 13, color: "var(--muted)" },
  probValue: { fontSize: 28, fontWeight: 700 },
  barTrack: { height: 8, background: "var(--border)", borderRadius: 4, overflow: "hidden" },
  barFill: { height: "100%", borderRadius: 4, transition: "width 0.5s ease" },
  hint: { fontSize: 13, color: "var(--muted)", lineHeight: 1.5 },
};
