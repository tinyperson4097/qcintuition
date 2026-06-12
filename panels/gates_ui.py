"""Gate application section."""
import sympy as sp
import streamlit as st
from quantum_core import (QuantumSystem, apply_gate, apply_custom_gate,
                          measure_qubit, measurement_probs)
from gates import ALL_GATE_NAMES, BLOCH_EFFECT
from format_utils import bump_sys, parse_expr, parse_matrix_code

_TWO_Q   = {"CNOT", "CZ", "SWAP"}
_THREE_Q = {"CCNOT"}

# Gates that take an angle parameter, with the symbol shown on the input.
_ANGLE_SYMBOL = {"P": "φ", "Rx": "θ", "Ry": "θ", "Rz": "θ"}

_CUSTOM_TEMPLATE = "sp.Matrix([\n    [1, 0],\n    [0, 1],\n])"


def apply_gates_section():
    sys: QuantumSystem = st.session_state.system
    n   = sys.num_qubits

    col_gate, col_qubits, col_phi, col_btn, col_undo = st.columns([1.2, 1.8, 0.9, 0.7, 0.7])

    with col_gate:
        gate = st.selectbox("Gate", ALL_GATE_NAMES, key="gate_sel")

    n_t  = 2 if gate in _TWO_Q else (3 if gate in _THREE_Q else 1)
    opts = list(range(n))

    with col_qubits:
        if n_t == 1:
            q = st.selectbox("Qubit", opts, key="gate_q0")
            qidx: tuple = (q,)
        elif n_t == 2:
            if n < 2:
                st.warning("Need ≥ 2 qubits"); return
            c0  = st.selectbox("Control", opts, key="gate_q0")
            c1  = st.selectbox("Target",  [q for q in opts if q != c0], key="gate_q1")
            qidx = (c0, c1)
        else:
            if n < 3:
                st.warning("Need ≥ 3 qubits"); return
            c0  = st.selectbox("Ctrl 1", opts, key="gate_q0")
            r1  = [q for q in opts if q != c0]
            c1  = st.selectbox("Ctrl 2", r1,   key="gate_q1")
            r2  = [q for q in opts if q not in (c0, c1)]
            c2  = st.selectbox("Target", r2,   key="gate_q2")
            qidx = (c0, c1, c2)

    with col_phi:
        phi = sp.Integer(0)
        angle_ok = True
        if gate in _ANGLE_SYMBOL:
            sym = _ANGLE_SYMBOL[gate]
            angle_text = st.text_input(f"{sym} (rad)", value="pi/4", key="gate_phi")
            try:
                phi = parse_expr(angle_text)   # exact: pi, pi/4, sqrt(2), …
            except ValueError:
                st.error(f"Invalid {sym} expression")
                angle_ok = False
        elif gate == "Custom":
            st.caption("matrix below ↓")
        else:
            st.caption("angle: P / R gates")

    with col_btn:
        st.markdown("<div style='padding-top:1.7rem'></div>", unsafe_allow_html=True)
        if st.button("Apply →", key="gate_apply", disabled=not angle_ok):
            try:
                if gate == "Custom":
                    code = st.session_state.get("custom_gate_code", _CUSTOM_TEMPLATE)
                    M = parse_matrix_code(code)
                    new_sys = apply_custom_gate(sys, M, qidx)
                else:
                    new_sys = apply_gate(sys, gate, qidx, phi=phi)
                new_sys.current_step = len(new_sys.gate_history)
                st.session_state.system = new_sys
                bump_sys()
                st.rerun()
            except Exception as e:
                st.error(str(e))

    with col_undo:
        st.markdown("<div style='padding-top:1.7rem'></div>", unsafe_allow_html=True)
        if st.button("↩ Undo", key="gate_undo", disabled=not sys.gate_history):
            # State before the last gate is the second-to-last intermediate
            # state — no matrix inversion needed.
            new_sv   = sys.intermediate_states()[-2]
            new_hist = sys.gate_history[:-1]
            st.session_state.system = QuantumSystem(
                n, new_sv, new_hist, len(new_hist),
                initial_state=sys.initial_state)
            bump_sys()
            st.rerun()

    if gate == "Custom":
        st.text_area("Custom matrix", value=_CUSTOM_TEMPLATE, height=120,
                     key="custom_gate_code", label_visibility="collapsed")
        st.caption("2×2 unitary applied to the selected qubit. SymPy syntax — "
                   "exact forms like `sp.pi`, `sp.sqrt(2)`, `sp.I`, `sp.exp(sp.I*sp.pi/4)`. "
                   "`np` is also available. Full-system matrices: use the "
                   "Linear Algebra panel's code editor.")

    effect = BLOCH_EFFECT.get(gate, "")
    if effect:
        st.caption(f"**{gate}:** {effect}")

    _measure_row(st.session_state.system)


def _measure_row(sys: QuantumSystem):
    """Projective measurement controls: live Born probabilities + collapse."""
    col_q, col_p, col_btn, col_out = st.columns([1.2, 2.7, 0.7, 1.4])
    with col_q:
        mq = st.selectbox("Measure qubit", list(range(sys.num_qubits)),
                          key="meas_q")
    p0, p1 = measurement_probs(sys, mq)
    with col_p:
        st.markdown("<div style='padding-top:1.7rem'></div>",
                    unsafe_allow_html=True)
        st.latex(rf"P(0) = {sp.latex(p0)},\quad P(1) = {sp.latex(p1)}")
    with col_btn:
        st.markdown("<div style='padding-top:1.7rem'></div>",
                    unsafe_allow_html=True)
        if st.button("Measure", key="meas_btn"):
            new_sys, outcome, _ = measure_qubit(sys, mq)
            new_sys.current_step = len(new_sys.gate_history)
            st.session_state.system = new_sys
            st.session_state["last_meas"] = (mq, outcome)
            bump_sys()
            st.rerun()
    with col_out:
        st.markdown("<div style='padding-top:1.7rem'></div>",
                    unsafe_allow_html=True)
        last = st.session_state.get("last_meas")
        if last is not None:
            st.markdown(f"last: qubit {last[0]} → **{last[1]}**")
