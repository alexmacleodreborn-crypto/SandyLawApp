import streamlit as st
import numpy as np
import pandas as pd
import math

# =====================================================
# SandyOS Core — INLINE (Cloud-Safe)
# =====================================================

def normalize(values):
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return 0.0
    lo, hi = min(xs), max(xs)
    if hi == lo:
        return 0.5
    return sum((x - lo) / (hi - lo) for x in xs) / len(xs)

def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))

def sandy_core(confinement, entropy, tau_history, theta_rp):
    Z = clamp(normalize(confinement))
    Sigma = max(0.0, normalize(entropy))
    tau_rate = (1 - Z) * Sigma

    # RP probability (logistic on slope)
    if len(tau_history) < 2:
        rp_prob = 0.0
    else:
        slope = tau_rate - tau_history[-1]
        rp_prob = 1 / (1 + math.exp(-15 * (slope - theta_rp)))
        rp_prob = clamp(rp_prob)

    confidence = clamp(min(1.0, (len(confinement) + len(entropy)) / 6))

    return Z, Sigma, tau_rate, rp_prob, confidence

# =====================================================
# Streamlit UI
# =====================================================

st.set_page_config(page_title="SandyOS", layout="wide")
st.title("🧭 SandyOS — Effective Time & Reaction Point Monitor")
st.caption("dτₛₗ/dt = (1 − Z) Σ")

# Sidebar
st.sidebar.header("⚙️ Controls")
theta_rp = st.sidebar.slider("RP Threshold (Θᴿᴾ)", 0.0, 1.0, 0.05, 0.01)
history_len = st.sidebar.slider("History Length", 3, 50, 10)

mode = st.sidebar.radio("Input Mode", ["Manual", "CSV"])

# Inputs
if mode == "Manual":
    st.subheader("🔒 Confinement (Z proxies)")
    confinement = [
        st.slider("Confinement 1", 0.0, 1.0, 0.85),
        st.slider("Confinement 2", 0.0, 1.0, 0.90),
        st.slider("Confinement 3", 0.0, 1.0, 0.80),
    ]

    st.subheader("🌊 Entropy Export (Σ proxies)")
    entropy = [
        st.slider("Entropy 1", 0.0, 1.0, 0.20),
        st.slider("Entropy 2", 0.0, 1.0, 0.35),
        st.slider("Entropy 3", 0.0, 1.0, 0.10),
    ]

    tau_history = list(np.linspace(0.01, 0.03, history_len))

else:
    st.subheader("📋 Paste CSV Data")
st.caption("Columns must include confinement_* and entropy_*")

csv_text = st.text_area(
    "Paste CSV here",
    height=250,
    placeholder=(
        "time,confinement_crust,confinement_pressure,entropy_gas,entropy_seismic\n"
        "t1,0.92,0.88,0.12,0.05\n"
        "t2,0.91,0.87,0.18,0.09\n"
    ),
)

if not csv_text.strip():
    st.stop()

from io import StringIO
df = pd.read_csv(StringIO(csv_text))
# Run SandyOS
Z, Sigma, tau_rate, rp_prob, confidence = sandy_core(
    confinement, entropy, tau_history, theta_rp
)

# Display
st.divider()
st.subheader("📊 SandyOS Output")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Z (Trap Strength)", f"{Z:.3f}")
c2.metric("Σ (Entropy Export)", f"{Sigma:.3f}")
c3.metric("dτₛₗ/dt", f"{tau_rate:.4f}")
c4.metric("RP Probability", f"{rp_prob:.1%}")
c5.metric("Confidence", f"{confidence:.1%}")

# Status
if rp_prob > 0.75:
    st.error("🚨 Reaction Point likely — TRANSITION phase")
elif rp_prob > 0.4:
    st.warning("⚠️ RP watch — system softening")
else:
    st.success("✅ Stable / trapped regime")

# Plot (Streamlit native)
st.subheader("📈 Effective Evolution Rate")
chart_df = pd.DataFrame({
    "dτₛₗ/dt": tau_history + [tau_rate],
    "RP Threshold": [theta_rp] * (len(tau_history) + 1),
})
st.line_chart(chart_df)

# Explainability
with st.expander("🧠 Explainability"):
    st.markdown("""
    • **Z** — how trapped the system is  
    • **Σ** — how much disorder escapes  
    • **dτₛₗ/dt** — how fast change is allowed  
    • **RP probability** — *warning before event*
    """)