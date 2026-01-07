import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sandyos_core.core import SandyCore
from sandyos_core.state import SystemState

# -----------------------------
# App config
# -----------------------------
st.set_page_config(
    page_title="SandyOS",
    layout="wide",
)

st.title("🧭 SandyOS — Effective Time & Reaction Point Monitor")
st.caption("Implements Sandy’s Law:  dτₛₗ/dt = (1 − Z) Σ")

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("⚙️ SandyOS Controls")

theta_rp = st.sidebar.slider(
    "RP Threshold (Θᴿᴾ)",
    min_value=0.0,
    max_value=1.0,
    value=0.05,
    step=0.01,
)

history_len = st.sidebar.slider(
    "History Length",
    min_value=3,
    max_value=50,
    value=10,
)

mode = st.sidebar.radio(
    "Input Mode",
    ["Manual Sliders", "CSV Upload"],
)

# -----------------------------
# Input handling
# -----------------------------
if mode == "Manual Sliders":
    st.subheader("🔒 Confinement Proxies (Z inputs)")
    z1 = st.slider("Proxy 1 (e.g. pressure / opacity)", 0.0, 1.0, 0.85)
    z2 = st.slider("Proxy 2 (e.g. rigidity / curvature)", 0.0, 1.0, 0.90)
    z3 = st.slider("Proxy 3 (optional)", 0.0, 1.0, 0.80)

    confinement = [z1, z2, z3]

    st.subheader("🌊 Entropy Export Proxies (Σ inputs)")
    s1 = st.slider("Proxy 1 (e.g. heat / radiation)", 0.0, 1.0, 0.20)
    s2 = st.slider("Proxy 2 (e.g. flux / throughput)", 0.0, 1.0, 0.35)
    s3 = st.slider("Proxy 3 (optional)", 0.0, 1.0, 0.10)

    entropy = [s1, s2, s3]

    tau_history = list(np.linspace(0.01, 0.03, history_len))

else:
    st.subheader("📂 Upload CSV")
    st.caption("CSV columns: confinement_*, entropy_*")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is None:
        st.stop()

    df = pd.read_csv(file)

    confinement = df.filter(like="confinement").iloc[-1].tolist()
    entropy = df.filter(like="entropy").iloc[-1].tolist()
    tau_history = df.get("tau_rate", []).tail(history_len).tolist()

# -----------------------------
# Run SandyOS Core
# -----------------------------
core = SandyCore(theta_rp=theta_rp)

state = SystemState(
    confinement_proxies=confinement,
    entropy_flux_proxies=entropy,
)

output = core.evaluate(
    state,
    history=tau_history,
)

# -----------------------------
# Display metrics
# -----------------------------
st.divider()
st.subheader("📊 SandyOS Core Output")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Z (Trap Strength)", f"{output.Z:.3f}")
col2.metric("Σ (Entropy Export)", f"{output.Sigma:.3f}")
col3.metric("dτₛₗ/dt", f"{output.tau_rate:.4f}")
col4.metric("RP Probability", f"{output.rp_probability:.2%}")
col5.metric("Confidence", f"{output.confidence:.2%}")

# -----------------------------
# RP interpretation
# -----------------------------
if output.rp_probability > 0.75:
    st.error("🚨 Reaction Point Likely — system entering TRANSITION")
elif output.rp_probability > 0.4:
    st.warning("⚠️ RP Watch — confinement weakening")
else:
    st.success("✅ Stable — trapped regime")

# -----------------------------
# Plot τ-rate history
# -----------------------------
st.divider()
st.subheader("📈 Effective Evolution Rate")

tau_series = tau_history + [output.tau_rate]

fig, ax = plt.subplots()
ax.plot(tau_series, marker="o")
ax.axhline(theta_rp, linestyle="--", label="RP Threshold")
ax.set_xlabel("Time step")
ax.set_ylabel("dτₛₗ/dt")
ax.legend()

st.pyplot(fig)

# -----------------------------
# Explainability panel
# -----------------------------
with st.expander("🧠 Explainability"):
    st.write("**Why this matters:**")
    st.write(
        """
        • Z measures how trapped the system is  
        • Σ measures how much disorder is escaping  
        • dτₛₗ/dt tells you how fast the system is *allowed* to evolve  
        • RP probability rises *before* eruptions, collapses, or regime shifts
        """
    )