import streamlit as st
import numpy as np
import pandas as pd
import math
from io import StringIO
import time

# =====================================================
# SESSION STATE
# =====================================================

# Scalar (time-series)
if "step" not in st.session_state:
    st.session_state.step = 0
if "max_step" not in st.session_state:
    st.session_state.max_step = 0

# Square (spatial)
if "sq_step" not in st.session_state:
    st.session_state.sq_step = 0
if "sq_play" not in st.session_state:
    st.session_state.sq_play = False
if "sq_persist" not in st.session_state:
    st.session_state.sq_persist = 0

# =====================================================
# CORE MATH (SCALAR)
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

def sandy_scalar(confinement, entropy, tau_history, theta_rp):
    Z = clamp(normalize(confinement))
    Sigma = clamp(normalize(entropy))
    tau_rate = (1 - Z) * Sigma

    if len(tau_history) < 2:
        rp_prob = 0.0
    else:
        base = np.mean(tau_history[-3:]) if len(tau_history) >= 3 else tau_history[-1]
        slope = tau_rate - base
        rp_prob = clamp(1 / (1 + math.exp(-15 * (slope - theta_rp))))

    confidence = clamp(min(1.0, (len(confinement) + len(entropy)) / 6))
    return Z, Sigma, tau_rate, rp_prob, confidence

# =====================================================
# CORE MATH (SQUARE)
# =====================================================

def square_step(df, t, grid=5, bgZ=0.90, bgS=0.10):
    Z = np.full((grid, grid), bgZ)
    S = np.full((grid, grid), bgS)

    rows = df[df["time"] == t]
    for _, r in rows.iterrows():
        i, j = int(r["i"]) - 1, int(r["j"]) - 1
        Z[i, j] = float(r["confinement"])
        S[i, j] = float(r["entropy"])

    tau = (1 - Z) * S
    return tau

def square_connected_region(tau, thresh=0.20, grid=5):
    hot = tau >= thresh
    visited = np.zeros_like(hot, bool)

    def dfs(i, j):
        stack = [(i, j)]
        size = 0
        while stack:
            x, y = stack.pop()
            if (
                x < 0 or x >= grid or y < 0 or y >= grid
                or visited[x, y] or not hot[x, y]
            ):
                continue
            visited[x, y] = True
            size += 1
            stack.extend([(x+1,y),(x-1,y),(x,y+1),(x,y-1)])
        return size

    Amax = 0
    for i in range(grid):
        for j in range(grid):
            if hot[i, j] and not visited[i, j]:
                Amax = max(Amax, dfs(i, j))

    return hot, Amax

# =====================================================
# UI
# =====================================================

st.set_page_config(page_title="SandyOS", layout="wide")
st.title("🧭 SandyOS")
st.caption("Reaction Points from entropy under confinement")

st.sidebar.header("⚙️ Mode")
mode = st.sidebar.radio("Input Mode", ["Manual", "Paste CSV", "Square"])

# =====================================================
# SIDEBAR CONTROLS (OPTION A)
# =====================================================

if mode in ["Manual", "Paste CSV"]:
    st.sidebar.subheader("Scalar Controls")
    theta_rp = st.sidebar.slider("RP Threshold Θᴿᴾ", 0.0, 1.0, 0.05, 0.01)
    history_len = st.sidebar.slider("History Length", 3, 50, 10)
else:
    theta_rp = 0.05
    history_len = 10

if mode == "Square":
    st.sidebar.divider()
    st.sidebar.subheader("Square Controls")

    square_tau_thresh = st.sidebar.slider(
        "Square τ̇ threshold",
        0.05, 0.50, 0.20, 0.01
    )
    square_persist_n = st.sidebar.slider(
        "Persistence (steps)",
        1, 5, 2
    )

    play_speed_ms = st.sidebar.slider(
        "Play speed (ms)",
        200, 2000, 600, 100
    )

# =====================================================
# MANUAL MODE
# =====================================================

if mode == "Manual":
    st.subheader("Manual Scalar Input")

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
    Z, Sigma, tau_rate, rp_prob, conf = sandy_scalar(
        confinement, entropy, tau_history, theta_rp
    )

# =====================================================
# CSV MODE
# =====================================================

elif mode == "Paste CSV":
    st.subheader("Paste Scalar CSV")

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

    Z, Sigma, tau_rate, rp_prob, conf = sandy_scalar(
        confinement, entropy, tau_history, theta_rp
    )

# =====================================================
# SQUARE MODE
# =====================================================

else:
    st.subheader("🟥 Square Mode (Spatial RP)")
    st.info("Square mode uses spatial percolation + persistence. Scalar controls do not apply.")

    sq_text = st.text_area(
        "Paste Square CSV",
        height=260,
        placeholder="time,i,j,confinement,entropy"
    )
    if not sq_text.strip():
        st.stop()

    sq_df = pd.read_csv(StringIO(sq_text))
    times = sorted(sq_df["time"].unique())

    c1, c2 = st.sidebar.columns(2)
    if c1.button("▶️ Play" if not st.session_state.sq_play else "⏸ Pause"):
        st.session_state.sq_play = not st.session_state.sq_play
    if c2.button("⏮ Reset"):
        st.session_state.sq_step = 0
        st.session_state.sq_persist = 0
        st.session_state.sq_play = False

    st.session_state.sq_step = min(st.session_state.sq_step, len(times) - 1)
    t = times[st.session_state.sq_step]

    tau = square_step(sq_df, t)
    hot, Amax = square_connected_region(tau, square_tau_thresh)

    if Amax >= 3:
        st.session_state.sq_persist += 1
    else:
        st.session_state.sq_persist = 0

    square_rp = st.session_state.sq_persist >= square_persist_n

    st.subheader(f"Square State — {t}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Max τ̇", f"{tau.max():.3f}")
    m2.metric("Hot region Aₘₐₓ", Amax)
    m3.metric("Persistence", f"{st.session_state.sq_persist}/{square_persist_n}")
    m4.metric("Square RP", "YES" if square_rp else "NO")

    st.write("τ̇ field")
    st.dataframe(pd.DataFrame(tau))

    st.write("Hot cells")
    st.dataframe(pd.DataFrame(hot.astype(int)))

    if st.session_state.sq_play and st.session_state.sq_step < len(times) - 1:
        time.sleep(play_speed_ms / 1000.0)
        st.session_state.sq_step += 1
        st.rerun()

    st.stop()

# =====================================================
# SCALAR OUTPUT
# =====================================================

st.divider()
st.subheader("Scalar Output")

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