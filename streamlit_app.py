import streamlit as st
import numpy as np
import pandas as pd
import math
import requests
from io import StringIO

# =====================================================
# SESSION STATE (Play / Pause)
# =====================================================

if "play" not in st.session_state:
    st.session_state.play = False
if "step" not in st.session_state:
    st.session_state.step = 0

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

    if len(tau_history) < 2:
        rp_prob = 0.0
    else:
        slope = tau_rate - tau_history[-1]
        rp_prob = 1 / (1 + math.exp(-15 * (slope - theta_rp)))
        rp_prob = clamp(rp_prob)

    confidence = clamp(min(1.0, (len(confinement) + len(entropy)) / 6))
    return Z, Sigma, tau_rate, rp_prob, confidence

# =====================================================
# USGS HANS Volcano API (Context Layer)
# =====================================================

@st.cache_data(ttl=3600)
def fetch_usgs_volcanoes():
    url = "https://volcanoes.usgs.gov/hans-public/api/volcano/"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()

# =====================================================
# Streamlit UI
# =====================================================

st.set_page_config(page_title="SandyOS", layout="wide")
st.title("🧭 SandyOS — Effective Time & Reaction Point Monitor")
st.caption("dτₛₗ/dt = (1 − Z) Σ")

# =====================================================
# SIDEBAR CONTROLS
# =====================================================

st.sidebar.header("⚙️ Controls")
theta_rp_user = st.sidebar.slider("RP Threshold (Θᴿᴾ)", 0.0, 1.0, 0.05, 0.01)
history_len = st.sidebar.slider("History Length", 3, 50, 10)
mode = st.sidebar.radio("Input Mode", ["Manual", "Paste CSV"])

# -----------------------------------------------------
# Play / Pause Controls (CSV only)
# -----------------------------------------------------

if mode == "Paste CSV":
    st.sidebar.divider()
    st.sidebar.subheader("▶️ Time Evolution")

    col1, col2 = st.sidebar.columns(2)
    if col1.button("▶️ Play" if not st.session_state.play else "⏸ Pause"):
        st.session_state.play = not st.session_state.play

    if col2.button("⏮ Reset"):
        st.session_state.step = 0
        st.session_state.play = False

# -----------------------------------------------------
# Volcano Context (USGS HANS)
# -----------------------------------------------------

st.sidebar.divider()
st.sidebar.subheader("🌋 Volcano Context (USGS HANS)")

theta_rp = theta_rp_user
volcano = None
selected_volcano = None

try:
    volcanoes = fetch_usgs_volcanoes()
    volcano_lookup = {
        v.get("volcanoName"): v
        for v in volcanoes
        if v.get("volcanoName")
    }

    selected_volcano = st.sidebar.selectbox(
        "Select volcano",
        sorted(volcano_lookup.keys())
    )

    volcano = volcano_lookup[selected_volcano]

    st.sidebar.write(f"Type: {volcano.get('volcanoType', 'n/a')}")
    st.sidebar.write(f"Region: {volcano.get('region', 'n/a')}")
    st.sidebar.write(f"Elevation: {volcano.get('elevation', 'n/a')} m")

    vtype = (volcano.get("volcanoType") or "").lower()
    if "shield" in vtype:
        theta_rp = min(theta_rp, 0.04)
    elif "strato" in vtype:
        theta_rp = max(theta_rp, 0.06)

except Exception:
    st.sidebar.warning("USGS volcano list unavailable")

# =====================================================
# INPUT HANDLING
# =====================================================

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
    )

    if not csv_text.strip():
        st.stop()

    df = pd.read_csv(StringIO(csv_text))
    conf_cols = df.filter(like="confinement")
    ent_cols = df.filter(like="entropy")

    if conf_cols.empty or ent_cols.empty:
        st.error("CSV must contain confinement_* and entropy_* columns")
        st.stop()

    max_step = len(df) - 1
    st.session_state.step = min(st.session_state.step, max_step)

    confinement = conf_cols.iloc[st.session_state.step].tolist()
    entropy = ent_cols.iloc[st.session_state.step].tolist()

    tau_history = []
    for i in range(max(0, st.session_state.step - history_len), st.session_state.step):
        z_i = normalize(conf_cols.iloc[i].tolist())
        s_i = normalize(ent_cols.iloc[i].tolist())
        tau_history.append((1 - z_i) * s_i)

# =====================================================
# RUN SANDYOS
# =====================================================

Z, Sigma, tau_rate, rp_prob, confidence = sandy_core(
    confinement, entropy, tau_history, theta_rp
)

# Auto-advance when playing
if mode == "Paste CSV" and st.session_state.play and st.session_state.step < max_step:
    st.session_state.step += 1
    st.experimental_rerun()

# =====================================================
# OUTPUT
# =====================================================

st.divider()
st.subheader("📊 SandyOS Output")

if mode == "Paste CSV":
    st.caption(f"⏱ Time step {st.session_state.step + 1} / {len(df)}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Z (Trap Strength)", f"{Z:.3f}")
c2.metric("Σ (Entropy Export)", f"{Sigma:.3f}")
c3.metric("dτₛₗ/dt", f"{tau_rate:.4f}")
c4.metric("RP Probability", f"{rp_prob:.1%}")
c5.metric("Confidence", f"{confidence:.1%}")

if rp_prob > 0.75:
    st.error("🚨 Reaction Point likely — TRANSITION phase")
elif rp_prob > 0.4:
    st.warning("⚠️ RP watch — system softening")
else:
    st.success("✅ Stable / trapped regime")

# =====================================================
# PLOT (RESTORED)
# =====================================================

st.subheader("📈 Effective Evolution Rate")

chart_df = pd.DataFrame({
    "dτₛₗ/dt": tau_history + [tau_rate],
    "RP Threshold": [theta_rp] * (len(tau_history) + 1),
})

st.line_chart(chart_df)

# =====================================================
# EXPLAINABILITY
# =====================================================

with st.expander("🧠 Explainability"):
    st.markdown("""
    • **Z** — how trapped the system is  
    • **Σ** — how much disorder escapes  
    • **dτₛₗ/dt** — how fast change is allowed  
    • **RP probability** — warning before event  
    • Volcano context via **USGS HANS API**
    """)