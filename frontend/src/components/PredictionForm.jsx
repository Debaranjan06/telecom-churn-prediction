import React, { useState } from "react";

const DEFAULTS = {
  gender: "Male",
  SeniorCitizen: "0",
  Partner: "Yes",
  Dependents: "No",
  tenure: 12,
  PhoneService: "Yes",
  MultipleLines: "No",
  InternetService: "Fiber optic",
  OnlineSecurity: "No",
  OnlineBackup: "No",
  DeviceProtection: "No",
  TechSupport: "No",
  StreamingTV: "No",
  StreamingMovies: "No",
  Contract: "Month-to-month",
  PaperlessBilling: "Yes",
  PaymentMethod: "Electronic check",
  MonthlyCharges: 70.0,
  TotalCharges: 840.0,
};

const SELECT = ({ label, name, options, value, onChange }) => (
  <div style={styles.field}>
    <label style={styles.label}>{label}</label>
    <select name={name} value={value} onChange={onChange} style={styles.select}>
      {options.map((o) => <option key={o} value={o}>{o}</option>)}
    </select>
  </div>
);

const NUM = ({ label, name, value, min, max, step, onChange }) => (
  <div style={styles.field}>
    <label style={styles.label}>{label}</label>
    <input
      type="number" name={name} value={value}
      min={min} max={max} step={step}
      onChange={onChange} style={styles.input}
    />
  </div>
);

export default function PredictionForm({ onResult, loading, setLoading }) {
  const [form, setForm] = useState(DEFAULTS);
  const [error, setError] = useState(null);

  const handle = (e) => {
    const { name, value, type } = e.target;
    setForm((f) => ({ ...f, [name]: type === "number" ? parseFloat(value) : value }));
  };

  const submit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const payload = { ...form, tenure: parseInt(form.tenure) };
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(JSON.stringify(err.detail ?? err));
      }
      onResult(await res.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={submit} style={styles.form}>
      <Section title="Profile">
        <SELECT label="Gender" name="gender" options={["Male", "Female"]} value={form.gender} onChange={handle} />
        <SELECT label="Senior Citizen" name="SeniorCitizen" options={["0", "1"]} value={form.SeniorCitizen} onChange={handle} />
        <SELECT label="Partner" name="Partner" options={["Yes", "No"]} value={form.Partner} onChange={handle} />
        <SELECT label="Dependents" name="Dependents" options={["Yes", "No"]} value={form.Dependents} onChange={handle} />
        <NUM label="Tenure (months)" name="tenure" value={form.tenure} min={0} max={72} step={1} onChange={handle} />
      </Section>

      <Section title="Services">
        <SELECT label="Phone Service" name="PhoneService" options={["Yes", "No"]} value={form.PhoneService} onChange={handle} />
        <SELECT label="Multiple Lines" name="MultipleLines" options={["Yes", "No", "No phone service"]} value={form.MultipleLines} onChange={handle} />
        <SELECT label="Internet Service" name="InternetService" options={["DSL", "Fiber optic", "No"]} value={form.InternetService} onChange={handle} />
        <SELECT label="Online Security" name="OnlineSecurity" options={["Yes", "No", "No internet service"]} value={form.OnlineSecurity} onChange={handle} />
        <SELECT label="Online Backup" name="OnlineBackup" options={["Yes", "No", "No internet service"]} value={form.OnlineBackup} onChange={handle} />
        <SELECT label="Device Protection" name="DeviceProtection" options={["Yes", "No", "No internet service"]} value={form.DeviceProtection} onChange={handle} />
        <SELECT label="Tech Support" name="TechSupport" options={["Yes", "No", "No internet service"]} value={form.TechSupport} onChange={handle} />
        <SELECT label="Streaming TV" name="StreamingTV" options={["Yes", "No", "No internet service"]} value={form.StreamingTV} onChange={handle} />
        <SELECT label="Streaming Movies" name="StreamingMovies" options={["Yes", "No", "No internet service"]} value={form.StreamingMovies} onChange={handle} />
      </Section>

      <Section title="Billing">
        <SELECT label="Contract" name="Contract" options={["Month-to-month", "One year", "Two year"]} value={form.Contract} onChange={handle} />
        <SELECT label="Paperless Billing" name="PaperlessBilling" options={["Yes", "No"]} value={form.PaperlessBilling} onChange={handle} />
        <SELECT label="Payment Method" name="PaymentMethod"
          options={["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]}
          value={form.PaymentMethod} onChange={handle} />
        <NUM label="Monthly Charges ($)" name="MonthlyCharges" value={form.MonthlyCharges} min={0} step={0.01} onChange={handle} />
        <NUM label="Total Charges ($)" name="TotalCharges" value={form.TotalCharges} min={0} step={0.01} onChange={handle} />
      </Section>

      {error && <p style={styles.error}>{error}</p>}

      <button type="submit" disabled={loading} style={styles.btn}>
        {loading ? "Predicting..." : "Predict Churn"}
      </button>
    </form>
  );
}

function Section({ title, children }) {
  return (
    <div style={styles.section}>
      <h3 style={styles.sectionTitle}>{title}</h3>
      <div style={styles.grid}>{children}</div>
    </div>
  );
}

const styles = {
  form: { display: "flex", flexDirection: "column", gap: 24 },
  section: { background: "var(--surface)", borderRadius: 12, padding: 20, border: "1px solid var(--border)" },
  sectionTitle: { fontSize: 13, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--brand-light)", marginBottom: 16 },
  grid: { display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: 12 },
  field: { display: "flex", flexDirection: "column", gap: 4 },
  label: { fontSize: 12, color: "var(--muted)", fontWeight: 500 },
  select: { background: "#0f172a", color: "#e2e8f0", border: "1px solid var(--border)", borderRadius: 6, padding: "6px 8px", fontSize: 14 },
  input: { background: "#0f172a", color: "#e2e8f0", border: "1px solid var(--border)", borderRadius: 6, padding: "6px 8px", fontSize: 14 },
  btn: { background: "var(--brand)", color: "#fff", border: "none", borderRadius: 8, padding: "12px 0", fontSize: 15, fontWeight: 600, cursor: "pointer", transition: "opacity 0.15s" },
  error: { color: "var(--danger)", fontSize: 13, background: "#1e0a0a", padding: 10, borderRadius: 6, border: "1px solid var(--danger)" },
};
