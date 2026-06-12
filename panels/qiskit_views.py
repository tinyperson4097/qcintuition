"""Qiskit views — per-qubit phase disks and global Q-sphere."""
from __future__ import annotations
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
import streamlit as st

from qiskit.quantum_info import Statevector, partial_trace
from qiskit.visualization import plot_state_qsphere

from quantum_core import QuantumSystem


def _np_state(sv: sp.Matrix) -> np.ndarray:
    return np.array([complex(sp.N(x)) for x in sv], dtype=complex)


# ── Q-sphere ──────────────────────────────────────────────────────────────────

def qsphere_panel():
    st.subheader("Q-sphere")
    sys: QuantumSystem = st.session_state.system
    arr = _np_state(sys.active_state())
    # App convention (q0 = most-significant bit) means the binary index string
    # already reads |q0 q1 …⟩ left-to-right, matching the qsphere labels.
    fig = plot_state_qsphere(Statevector(arr), figsize=(3.2, 3.2))
    fig.patch.set_alpha(0)
    st.pyplot(fig, width="content")
    plt.close(fig)
    st.caption("One point per basis state — size ∝ probability, color = phase, "
               "latitude = number of 1s. Labels read |q0 q1 …⟩.")


# ── phase disks ───────────────────────────────────────────────────────────────

def phase_disk_panel():
    st.subheader("Phase Disks")
    sys: QuantumSystem = st.session_state.system
    n   = sys.num_qubits
    qsv = Statevector(_np_state(sys.active_state()))

    fig, axes = plt.subplots(1, n, figsize=(1.3 * n, 1.6))
    if n == 1:
        axes = [axes]
    fig.patch.set_alpha(0)
    for q in range(n):
        # app qubit q = qiskit qubit n-1-q (qiskit is little-endian)
        keep = n - 1 - q
        rho  = partial_trace(qsv, [j for j in range(n) if j != keep]).data
        _draw_disk(axes[q], rho, f"q{q}")
    st.pyplot(fig, width="content")
    plt.close(fig)
    st.caption("Fill area = P(|1⟩) · needle = relative phase of |1⟩ (CCW from east, "
               "hidden when no coherence) · green arc = purity, full ring = pure, "
               "half ring = maximally entangled.")


def _draw_disk(ax, rho: np.ndarray, label: str):
    p1     = float(np.clip(rho[1, 1].real, 0.0, 1.0))
    coh    = complex(rho[1, 0])
    purity = float(np.clip((rho @ rho).trace().real, 0.0, 1.0))

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)

    # purity ring: background track + arc clockwise from 12 o'clock
    ax.add_patch(Wedge((0, 0), 1.15, 0, 360, width=0.13,
                       facecolor="#888888", alpha=0.25, edgecolor="none"))
    ax.add_patch(Wedge((0, 0), 1.15, 90 - purity * 360, 90, width=0.13,
                       facecolor="#52c452", edgecolor="none"))

    # disk outline + probability fill (area-proportional → radius √p)
    ax.add_patch(Circle((0, 0), 0.9, facecolor="none",
                        edgecolor="#aaaaaa", lw=1.0))
    if p1 > 1e-9:
        ax.add_patch(Circle((0, 0), 0.9 * np.sqrt(p1),
                            facecolor="#4a90d9", edgecolor="none", alpha=0.85))

    # phase needle
    if abs(coh) > 1e-9:
        phi = float(np.angle(coh))
        ax.plot([0, 0.9 * np.cos(phi)], [0, 0.9 * np.sin(phi)],
                color="#ffa500", lw=2.2, solid_capstyle="round")

    ax.set_title(label, fontsize=10, color="#999999")
    ax.text(0, -1.32, f"P(1)={p1:.2f}", ha="center", va="top",
            fontsize=8, color="#999999")
