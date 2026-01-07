import streamlit as st
import numpy as np
import pandas as pd
import math
from io import StringIO
import time

# =====================================================
# SESSION STATE
# =====================================================

if "scalar_step" not in st.session_state:
    st.session_state.scalar_step = 0
if "scalar_play" not in st.session_state:
    st.session_state.scalar_play = False

if "square_step" not in st.session_state:
    st.session_state.square_step = 0
if "square_play" not in st.session_state:
    st.session_state.square_play = False
if "square_persist" not in st.session_state:
    st.session_state.square_persist = 0

# =====================================================
# CORE FUNCTIONS
# =====================================================

def normalize(vals):
    v = np.array(vals, dtype=float)
    if v.max() == v.min():
        return 0.5
    return np.mean((v - v.min()) / (v.max() - v.min()))

def sandy_scalar(conf, ent, hist, theta):
    Z = normalize(conf)
    S = normalize(ent)
    tau = (1 - Z) * S

    if len(hist) < 2:
        rp = 0.0
    else:
        base = np.mean(hist[-3:]) if len(hist) >= 3 else hist[-1]
        slope = tau - base
        rp = 1 / (1 + math.exp(-15 * (slope - theta)))

    return Z, S, tau, rp

# ---------------- Square ----------------

def square_tau(df, t, grid=5):
    Z = np.full((grid, grid), 0.9)
    S = np.full((grid, grid), 0.1)

    for _, r in df[df.time == t].iterrows():
        i, j = int(r.i) - 1, int(r.j) - 1
        Z[i, j] = r.confinement
        S[i, j] = r.entropy

    return (1 - Z) * S

def square_rp(tau, thresh):
    hot = tau >= thresh
    visited = np.zeros_like(hot)

    def dfs(i, j):
        stack = [(i, j)]
        s = 0
        while stack:
            x, y = stack.pop()
            if (
                x < 0 or y < 0 or x >= 5 or y >= 5
                or visited[x, y] or not hot[x, y]
            ):
                continue
            visited[x, y] = 1
            s += 1
            stack += [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
        return s

    Amax = 0
    for i in range(5):
        for j in range(5):
            if hot[i,j] and not visited[i,j]:
                Amax = max(Amax, dfs(i,j))
    return hot, Amax

# =====================================================
# UI
# =====================================================

st.set_page_config("SandyOS", layout="wide")
st.title("🧭 SandyOS — Scalar & Square RP Monitor")

mode = st.sidebar.radio("Mode", ["Scalar CSV", "Square 5×5"])

# =====================================================
# SCALAR MODE
# =====================================================

if mode == "Scalar CSV":
    st.sidebar.subheader("Scalar Controls")
    theta = st.sidebar.slider("RP Threshold Θᴿᴾ", 0.0, 1.0, 0.05, 0.01)
    speed = st.sidebar.slider("Play speed (ms)", 200, 2000, 600, 100)

    st.subheader("Paste Scalar CSV")

    csv_text = st.text_area(
        "CSV",
        height=220,
        placeholder="""time,confinement_crust,confinement_pressure,entropy_gas,entropy_seismic
t0,0.97,0.96,0.03,0.02
t1,0.96,0.95,0.06,0.05
t2,0.94,0.93,0.10,0.08
t3,0.91,0.89,0.16,0.14
t4,0.87,0.85,0.23,0.20"""
    )

    if not csv_text.strip():
        st.stop()

    df = pd.read_csv(StringIO(csv_text))
    max_step = len(df) - 1
    st.session_state.scalar_step = min(st.session_state.scalar_step, max_step)

    c1, c2 = st.sidebar.columns(2)
    if c1.button("▶️ Play" if not st.session_state.scalar_play else "⏸ Pause"):
        st.session_state.scalar_play = not st.session_state.scalar_play
    if c2.button("⏮ Reset"):
        st.session_state.scalar_step = 0
        st.session_state.scalar_play = False

    conf_cols = df.filter(like="confinement")
    ent_cols = df.filter(like="entropy")

    tau_hist = []
    for i in range(st.session_state.scalar_step):
        z = normalize(conf_cols.iloc[i])
        s = normalize(ent_cols.iloc[i])
        tau_hist.append((1 - z) * s)

    conf = conf_cols.iloc[st.session_state.scalar_step]
    ent = ent_cols.iloc[st.session_state.scalar_step]

    Z, S, tau, rp = sandy_scalar(conf, ent, tau_hist, theta)

    st.subheader(f"Step {st.session_state.scalar_step+1}/{len(df)}")

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Z", f"{Z:.3f}")
    m2.metric("Σ", f"{S:.3f}")
    m3.metric("τ̇", f"{tau:.4f}")
    m4.metric("RP", f"{rp:.1%}")

    chart = pd.DataFrame({
        "τ̇": tau_hist + [tau],
        "Θᴿᴾ": [theta] * (len(tau_hist)+1)
    })
    st.line_chart(chart)

    if st.session_state.scalar_play and st.session_state.scalar_step < max_step:
        time.sleep(speed/1000)
        st.session_state.scalar_step += 1
        st.rerun()

# =====================================================
# SQUARE MODE
# =====================================================

else:
    st.sidebar.subheader("Square Controls")
    thresh = st.sidebar.slider("τ̇ threshold", 0.05, 0.5, 0.2, 0.01)
    persist_n = st.sidebar.slider("Persistence", 1, 5, 2)
    speed = st.sidebar.slider("Play speed (ms)", 200, 2000, 600, 100)

    st.subheader("Paste Square CSV")

    sq_text = st.text_area(
        "CSV",
        height=260,
        placeholder="""time,i,j,confinement,entropy
t0,3,3,0.95,0.05
t1,3,3,0.92,0.08
t1,3,4,0.93,0.07
t2,3,3,0.90,0.12
t2,3,4,0.90,0.11
t2,4,3,0.91,0.10
t3,3,3,0.88,0.18
t3,3,4,0.87,0.17
t3,4,3,0.86,0.16"""
    )

    if not sq_text.strip():
        st.stop()

    sq = pd.read_csv(StringIO(sq_text))
    times = sorted(sq.time.unique())
    st.session_state.square_step = min(st.session_state.square_step, len(times)-1)
    t = times[st.session_state.square_step]

    c1,c2 = st.sidebar.columns(2)
    if c1.button("▶️ Play" if not st.session_state.square_play else "⏸ Pause"):
        st.session_state.square_play = not st.session_state.square_play
    if c2.button("⏮ Reset"):
        st.session_state.square_step = 0
        st.session_state.square_persist = 0
        st.session_state.square_play = False

    tau = square_tau(sq, t)
    hot, Amax = square_rp(tau, thresh)

    if Amax >= 3:
        st.session_state.square_persist += 1
    else:
        st.session_state.square_persist = 0

    rp = st.session_state.square_persist >= persist_n

    st.subheader(f"Square time {t}")

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Max τ̇", f"{tau.max():.3f}")
    m2.metric("Hot Aₘₐₓ", Amax)
    m3.metric("Persistence", f"{st.session_state.square_persist}/{persist_n}")
    m4.metric("Square RP", "YES" if rp else "NO")

    st.dataframe(pd.DataFrame(tau))
    st.dataframe(pd.DataFrame(hot.astype(int)))

    if st.session_state.square_play and st.session_state.square_step < len(times)-1:
        time.sleep(speed/1000)
        st.session_state.square_step += 1
        st.rerun()