import streamlit as st
import numpy as np
import pandas as pd
import math
import requests
from io import StringIO
import time

# =====================================================
# SESSION STATE
# =====================================================

if "play" not in st.session_state:
    st.session_state.play = False
if "step" not in st.session_state:
    st.session_state.step = 0
if "max_step" not in st.session_state:
    st.session_state.max_step = 0
if "sq_step" not in st.session_state:
    st.session_state.sq_step = 0

# =====================================================
# SandyOS Core
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
        if len(tau_history) >= 3:
            slope = tau_rate - float(np.mean(tau_history[-3:]))
        else:
            slope = tau_rate - tau_history[-1]
        rp_prob = 1 / (1 + math.exp(-15 * (slope - theta_rp)))
        rp_prob = clamp(rp_prob)

    confidence = clamp(min(1.0, (len(confinement) + len(entropy)) / 6))
    return Z, Sigma, tau_rate, rp_prob, confidence

# =====================================================
# Volcano Square Engine (NEW)
# =====================================================

def square_step(df, t, grid=5, bgZ=0.90, bgS=0.10):
    Z = np.full((grid, grid), bgZ)
    S = np.full((grid, grid), bgS)

    rows = df[df["time"] == t]
    for _, r in rows.iterrows():
        i, j = int(r["i"]) - 1, int(r["j"]) - 1
        Z[i, j] = r["confinement"]
        S[i, j] = r["entropy"]

    tau = (1 - Z) * S
    return Z, S, tau

def square_rp(tau, thresh=0.20):
    hot = tau >= thresh
    visited = np.zeros_like(hot, bool)

    def dfs(i, j):
        stack = [(i, j)]
        size = 0
        while stack:
            x, y = stack.pop()
            if (
                x < 0 or x >= 5 or y < 0 or y >= 5
                or visited[x, y] or not hot[x, y]
            ):
                continue
            visited[x, y] = True
            size += 1
            stack.extend([(x+1,y),(x-1,y),(x,y+1),(x,y-1)])
        return size

    Amax = 0
    for i in range(5):
        for j in range(5):
            if hot[i, j] and not visited[i, j]:
                Amax = max(Amax, dfs(i, j))

    return hot, Amax

# =====================================================
# UI
# =====================================================

st.set_page_config(page_title="SandyOS", layout="wide")
st.title("🧭 SandyOS — Effective Time & Reaction Point Monitor")
st.caption("dτₛₗ/dt = (1 − Z) Σ")

st.sidebar.header("⚙️ Controls")
theta_rp = st.sidebar.slider("RP Threshold Θᴿᴾ", 0.0, 1.0, 0.05, 0.01)
history_len = st.sidebar.slider("History Length", 3, 50, 10)
mode = st.sidebar.radio("Input Mode", ["Manual", "Paste CSV", "Square"])

# =====================================================
# INPUT MODES
# =====================================================

df = None

# ---------------- MANUAL ----------------
if mode == "Manual":
    st.subheader("Manual Input")

    confinement = [
        st.slider("Confinement 1", 0.0, 1.0, 0.85),
        st.slider("Confinement 2", 0.0, 1.0, 0.90),
        st.slider("Confinement 3", 0.0, 1.0, 0.80),
    ]

    entropy = [
        st.slider("Entropy 1", 0.0, 1.0, 0.20),
        st.slider("Entropy 2", 0.0, 1.0, 0.35),
        st.slider("Entropy 3", 0.0, 1.0, 0.10),
    ]

    tau_history = list(np.linspace(0.01, 0.03, history_len))
    Z, Sigma, tau_rate, rp_prob, conf = sandy_core(
        confinement, entropy, tau_history, theta_rp
    )

# ---------------- CSV ----------------
elif mode == "Paste CSV":
    st.subheader("Paste Time CSV")

    csv_text = st.text_area("Paste CSV", height=220)
    if not csv_text.strip():
        st.stop()

    df = pd.read_csv(StringIO(csv_text))
    conf_cols = df.filter(like="confinement")
    ent_cols = df.filter(like="entropy")

    st.session_state.max_step = len(df) - 1
    st.session_state.step = min(st.session_state.step, st.session_state.max_step)

    confinement = conf_cols.iloc[st.session_state.step].tolist()
    entropy = ent_cols.iloc[st.session_state.step].tolist()

    tau_history = []
    for i in range(max(0, st.session_state.step - history_len), st.session_state.step):
        z_i = normalize(conf_cols.iloc[i].tolist())
        s_i = normalize(ent_cols.iloc[i].tolist())
        tau_history.append((1 - z_i) * s_i)

    Z, Sigma, tau_rate, rp_prob, conf = sandy_core(
        confinement, entropy, tau_history, theta_rp
    )

# ---------------- SQUARE ----------------
else:
    st.subheader("🟥 Volcano Square (5×5)")

    sq_text = st.text_area(
        "Paste Square CSV",
        height=250,
        placeholder="time,i,j,confinement,entropy"
    )
    if not sq_text.strip():
        st.stop()

    sq_df = pd.read_csv(StringIO(sq_text))
    times = sorted(sq_df["time"].unique())

    st.session_state.sq_step = min(st.session_state.sq_step, len(times)-1)
    t = times[st.session_state.sq_step]

    Zg, Sg, tau_g = square_step(sq_df, t)
    hot, Amax = square_rp(tau_g)

    st.subheader(f"Square State — {t}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Max τ̇", f"{tau_g.max():.3f}")
    c2.metric("Hot region Aₘₐₓ", Amax)
    c3.metric("Square RP", "YES" if Amax >= 3 else "NO")

    st.write("τ̇ field")
    st.dataframe(pd.DataFrame(tau_g).style.background_gradient("inferno"))

    st.write("Hot cells (1 = hot)")
    st.dataframe(pd.DataFrame(hot.astype(int)))

    col1, col2 = st.columns(2)
    if col1.button("▶ Next"):
        st.session_state.sq_step += 1
        st.rerun()
    if col2.button("⏮ Reset"):
        st.session_state.sq_step = 0
        st.rerun()

    st.stop()

# =====================================================
# OUTPUT (Manual / CSV)
# =====================================================

st.divider()
st.subheader("SandyOS Output")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Z", f"{Z:.3f}")
c2.metric("Σ", f"{Sigma:.3f}")
c3.metric("τ̇", f"{tau_rate:.4f}")
c4.metric("RP", f"{rp_prob:.1%}")
c5.metric("Confidence", f"{conf:.1%}")

if rp_prob > 0.75:
    st.error("🚨 TRANSITION")
elif rp_prob > 0.4:
    st.warning("⚠️ RP Watch")
else:
    st.success("✅ Stable")

st.subheader("Evolution")
chart_df = pd.DataFrame({
    "τ̇": tau_history + [tau_rate],
    "Θᴿᴾ": [theta_rp] * (len(tau_history) + 1)
})
st.line_chart(chart_df)