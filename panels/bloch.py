"""Bloch sphere panel — one go.Figure per qubit."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from quantum_core import QuantumSystem, state_to_bloch, bloch_to_angles, set_state_from_bloch
from gates import BLOCH_EFFECT
from format_utils import bump_sys, sync_needed, fmt_angle_latex, show_latex

_BASIS_OPTIONS = ["comp", "hadamard", "circular"]
_BASIS_LABELS  = {"comp": "Computational |0⟩|1⟩",
                  "hadamard": "Hadamard |+⟩|−⟩",
                  "circular": "Circular |i⟩|−i⟩"}


def bloch_panel():
    st.subheader("Bloch Sphere")

    basis = st.selectbox("Basis", _BASIS_OPTIONS, format_func=_BASIS_LABELS.get,
                         key="basis_bs")
    if basis != st.session_state.basis:
        st.session_state.basis = basis
        st.session_state["_basis_sync"] = True
        bump_sys()
        st.rerun()

    sys: QuantumSystem = st.session_state.system
    n         = sys.num_qubits
    active_sv = sys.active_state()
    bvecs     = [state_to_bloch(active_sv, q, n) for q in range(n)]

    # ── figures ───────────────────────────────────────────────────────────────
    for q in range(n):
        if n > 1:
            st.markdown(f"**Qubit {q}**")
        fig = _bloch_fig(*bvecs[q])
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # ── angle display / sliders ───────────────────────────────────────────────
    st.markdown("---")

    if n == 1 and not sys.gate_history:
        theta, phi = bloch_to_angles(*bvecs[0])
        # Reset sliders when system changes externally (e.g. Reset button)
        if sync_needed("bs"):
            st.session_state["bs_theta"] = float(theta)
            st.session_state["bs_phi"]   = float(phi)
        st.markdown("**Adjust angles**")
        new_theta = st.slider("θ", 0.0, float(np.pi),  float(theta), 0.001,
                              format="%.3f", key="bs_theta")
        new_phi   = st.slider("φ", 0.0, float(2*np.pi), float(phi), 0.001,
                              format="%.3f", key="bs_phi")
        if not (np.isclose(new_theta, theta, atol=1e-3) and
                np.isclose(new_phi,   phi,   atol=1e-3)):
            st.session_state.system = set_state_from_bloch(sys, 0, new_theta, new_phi)
            bump_sys()
            st.rerun()

        show_latex(rf"\theta = {fmt_angle_latex(theta)}, \quad \varphi = {fmt_angle_latex(phi)}")
    else:
        for q in range(n):
            theta, phi = bloch_to_angles(*bvecs[q])
            r = float(np.sqrt(sum(v**2 for v in bvecs[q])))
            mixed = f" *(mixed, |r|={r:.2f})*" if not np.isclose(r, 1.0, atol=0.02) else ""
            st.markdown(f"**Qubit {q}**{mixed}")
            show_latex(rf"\theta = {fmt_angle_latex(theta)},\quad \varphi = {fmt_angle_latex(phi)}")
        if sys.gate_history:
            st.caption("Clear history (Reset) to use sliders.")

    # ── gate effects ──────────────────────────────────────────────────────────
    if sys.gate_history:
        st.markdown("---")
        st.markdown("**Gate effects**")
        cur = sys.current_step
        for i, op in enumerate(sys.gate_history):
            desc = BLOCH_EFFECT.get(op.name, f"{op.name} — unitary transform")
            lbl  = op.label_latex()
            if i == cur:
                st.markdown(f"**→ Step {i+1}:**")
                show_latex(lbl)
                st.caption(desc)
            else:
                st.markdown(f"Step {i+1}:")
                show_latex(lbl)
                st.caption(desc)


# ── single Bloch sphere ───────────────────────────────────────────────────────

def _bloch_fig(rx: float, ry: float, rz: float) -> go.Figure:
    fig = go.Figure()

    # Sphere shell
    u  = np.linspace(0, 2*np.pi, 40)
    v  = np.linspace(0, np.pi,   20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(40), np.cos(v))
    fig.add_trace(go.Surface(x=xs, y=ys, z=zs, opacity=0.08,
                             colorscale=[[0,"#4a90d9"],[1,"#4a90d9"]],
                             showscale=False, hoverinfo="skip",
                             lighting=dict(ambient=0.9)))

    # Three great circles (equator + two meridians)
    t = np.linspace(0, 2*np.pi, 80)
    for xs_, ys_, zs_ in [
        (np.cos(t), np.sin(t), np.zeros(80)),   # equator
        (np.cos(t), np.zeros(80), np.sin(t)),   # xz meridian
        (np.zeros(80), np.cos(t), np.sin(t)),   # yz meridian
    ]:
        fig.add_trace(go.Scatter3d(x=xs_, y=ys_, z=zs_, mode="lines",
                                   line=dict(color="rgba(180,180,180,0.3)", width=1),
                                   hoverinfo="skip"))

    # Axes
    for ax, col, lbl in [([1.2,0,0],"#e05252","x"),
                          ([0,1.2,0],"#52c452","y"),
                          ([0,0,1.2],"#5252e0","z")]:
        fig.add_trace(go.Scatter3d(
            x=[0,ax[0]], y=[0,ax[1]], z=[0,ax[2]],
            mode="lines+text", line=dict(color=col, width=3),
            text=["", lbl], textfont=dict(size=10, color=col), hoverinfo="skip"))

    # Pole labels
    for zv, txt in [(1.4,"|0⟩"), (-1.4,"|1⟩")]:
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[zv], mode="text", text=[txt],
                                   textfont=dict(size=12, color="rgba(220,220,220,0.85)"),
                                   hoverinfo="skip"))

    # State vector
    r       = float(np.sqrt(rx**2 + ry**2 + rz**2))
    tip_col = "#ffa500" if np.isclose(r, 1.0, atol=0.03) else "#ffff44"
    fig.add_trace(go.Scatter3d(
        x=[0,rx], y=[0,ry], z=[0,rz],
        mode="lines+markers",
        line=dict(color=tip_col, width=6),
        marker=dict(size=[0,8], color=tip_col),
        hovertemplate=f"x={rx:.2f}, y={ry:.2f}, z={rz:.2f}<extra></extra>"))

    fig.update_layout(
        height=300, margin=dict(l=0,r=0,t=0,b=0),
        paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False,
                       showgrid=False, zeroline=False, range=[-1.5,1.5]),
            yaxis=dict(showbackground=False, showticklabels=False,
                       showgrid=False, zeroline=False, range=[-1.5,1.5]),
            zaxis=dict(showbackground=False, showticklabels=False,
                       showgrid=False, zeroline=False, range=[-1.5,1.5]),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            bgcolor="rgba(0,0,0,0)",
        ))
    return fig
