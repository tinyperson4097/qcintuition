"""Guided example walkthroughs (sidebar) — drive the simulator panels."""
from __future__ import annotations
import sympy as sp
import streamlit as st

from quantum_core import (QuantumSystem, apply_gate, apply_custom_gate,
                          measure_qubit)
from format_utils import bump_sys, parse_expr

_H_LATEX = r"H = \tfrac{1}{\sqrt{2}}\begin{pmatrix}1&1\\1&-1\end{pmatrix}"

_DEF_ALPHA = "(sqrt(3)-1)/2"
_DEF_BETA  = "sqrt(2)/2"

_CS = sp.diag(1, 1, 1, sp.I)   # controlled-S


# ── simulator-driving helpers ─────────────────────────────────────────────────

def _set_system(sys_new: QuantumSystem, basis: str | None = None):
    sys_new.current_step = len(sys_new.gate_history)
    st.session_state.system = sys_new
    # Let the qubit-count selectbox re-derive its index from the new system
    # (its stale widget state would otherwise reset the system right back).
    st.session_state.pop("num_qubits_sel", None)
    if basis is not None:
        st.session_state.basis = basis
        st.session_state["_basis_sync"] = True
    bump_sys()
    st.rerun()


def _set_basis(basis: str):
    st.session_state.basis = basis
    st.session_state["_basis_sync"] = True
    bump_sys()
    st.rerun()


def _run(n: int, ops: list, init=None, basis: str = "comp"):
    """Build an n-qubit system, apply ops [(name, qidx), …], load it.

    Special op name "CS" applies the controlled-S matrix as a custom gate.
    """
    s = (QuantumSystem.ground(n) if init is None
         else QuantumSystem(n, sp.Matrix(init)))
    for op in ops:
        if op[0] == "CS":
            s = apply_custom_gate(s, _CS, op[1])
        else:
            s = apply_gate(s, op[0], op[1])
    _set_system(s, basis=basis)


def _one_qubit(c0, c1) -> QuantumSystem:
    return QuantumSystem(1, sp.Matrix([sp.simplify(c0), sp.simplify(c1)]))


def _parse_alpha_beta():
    """Parse the α, β inputs (persisted in session state). Returns (α, β) or None."""
    a_txt = st.session_state.get("wt_alpha", _DEF_ALPHA)
    b_txt = st.session_state.get("wt_beta", _DEF_BETA)
    try:
        return parse_expr(a_txt), parse_expr(b_txt)
    except ValueError:
        return None


def _psi_coeffs(a, b):
    """|ψ⟩ = α|0⟩ + β|+⟩ in the standard basis."""
    return (sp.simplify(a + b / sp.sqrt(2)), sp.simplify(b / sp.sqrt(2)))


def _is_normalized(c0, c1) -> bool:
    norm = sp.Abs(c0) ** 2 + sp.Abs(c1) ** 2
    return abs(complex(sp.N(norm)) - 1) < 1e-9


# ── walkthrough: Hadamard & the diagonal basis ────────────────────────────────

def _wt_hadamard():
    st.caption("Problems from [Dirac notation worksheet 1]"
               "(https://cdn-uploads.piazza.com/paste/kjj5qpj830c3qm/"
               "39c9456ca2562b2be5440ff54e94b32f3053ca25bf624c8b64b49c64eb0b54a3/"
               "dirac_worksheet_1.pdf).")
    st.latex(_H_LATEX + r",\qquad |\pm\rangle = \tfrac{|0\rangle \pm |1\rangle}{\sqrt{2}}")

    with st.expander("Part 1 — Show H² = I"):
        st.latex(r"H^2 = \tfrac{1}{2}\begin{pmatrix}1&1\\1&-1\end{pmatrix}"
                 r"\begin{pmatrix}1&1\\1&-1\end{pmatrix}"
                 r"= \tfrac{1}{2}\begin{pmatrix}2&0\\0&2\end{pmatrix} = I")
        st.caption("Apply H twice to |0⟩ — the state comes back. Step through "
                   "both multiplications in the Linear Algebra panel.")
        if st.button("▶ Run H, H on |0⟩", key="wt1"):
            _run(1, [("H", (0,)), ("H", (0,))])

    with st.expander("Part 2 — H|0⟩ = |+⟩ and H|1⟩ = |−⟩"):
        st.latex(r"H|0\rangle = \tfrac{1}{\sqrt{2}}\begin{pmatrix}1\\1\end{pmatrix}"
                 r"= |+\rangle, \qquad "
                 r"H|1\rangle = \tfrac{1}{\sqrt{2}}\begin{pmatrix}1\\-1\end{pmatrix}"
                 r"= |-\rangle")
        st.caption("After running, switch the basis selector to Hadamard — the "
                   "coefficient on |+⟩ (or |−⟩) is exactly 1.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("▶ H|0⟩", key="wt2a"):
                _run(1, [("H", (0,))])
        with c2:
            if st.button("▶ H|1⟩", key="wt2b"):
                _run(1, [("H", (0,))], init=[0, 1])

    with st.expander("Part 3 — |ψ⟩ = α|0⟩ + β|+⟩ in the standard basis"):
        st.latex(r"|\psi\rangle = \alpha|0\rangle + \beta|+\rangle"
                 r"= \left(\alpha + \tfrac{\beta}{\sqrt{2}}\right)|0\rangle"
                 r"+ \tfrac{\beta}{\sqrt{2}}|1\rangle")
        st.markdown("Unit-vector condition:")
        st.latex(r"\left|\alpha + \tfrac{\beta}{\sqrt{2}}\right|^2"
                 r"+ \left|\tfrac{\beta}{\sqrt{2}}\right|^2 = 1"
                 r"\;\Longleftrightarrow\;"
                 r"|\alpha|^2 + |\beta|^2 + \sqrt{2}\,\mathrm{Re}(\bar{\alpha}\beta) = 1")
        st.caption("Note α, β alone do *not* satisfy |α|²+|β|²=1, because "
                   "|0⟩ and |+⟩ are not orthogonal. Pick your own values "
                   "(SymPy syntax) — they must pass the condition above.")
        ca, cb = st.columns(2)
        with ca:
            st.text_input("α", value=_DEF_ALPHA, key="wt_alpha")
        with cb:
            st.text_input("β", value=_DEF_BETA, key="wt_beta")
        ab = _parse_alpha_beta()
        if ab is None:
            st.error("Can't parse α or β — use SymPy syntax like `sqrt(2)/2`.")
        else:
            c0, c1_ = _psi_coeffs(*ab)
            if not _is_normalized(c0, c1_):
                nval = complex(sp.N(sp.Abs(c0)**2 + sp.Abs(c1_)**2)).real
                st.error(f"Not a unit vector: ⟨ψ|ψ⟩ = {nval:.4f}. "
                         "Adjust α, β to satisfy the condition.")
            elif st.button("▶ Load |ψ⟩", key="wt3"):
                _set_system(_one_qubit(c0, c1_), basis="comp")

    with st.expander("Part 4 — |ψ⟩ in the diagonal basis"):
        st.latex(r"|0\rangle = \tfrac{|+\rangle + |-\rangle}{\sqrt{2}}"
                 r"\;\Rightarrow\;"
                 r"|\psi\rangle = \left(\tfrac{\alpha}{\sqrt{2}} + \beta\right)|+\rangle"
                 r"+ \tfrac{\alpha}{\sqrt{2}}|-\rangle")
        st.caption("With |ψ⟩ loaded from Part 3, switch the panels to the "
                   "Hadamard basis and compare the coefficients.")
        if st.button("▶ Show in diagonal basis", key="wt4"):
            _set_basis("hadamard")

    with st.expander("Part 5 — H|ψ⟩ in the standard basis"):
        st.latex(r"H|\psi\rangle = \alpha H|0\rangle + \beta H|+\rangle"
                 r"= \alpha|+\rangle + \beta|0\rangle"
                 r"= \left(\tfrac{\alpha}{\sqrt{2}} + \beta\right)|0\rangle"
                 r"+ \tfrac{\alpha}{\sqrt{2}}|1\rangle")
        st.caption("Same coefficients as Part 4 — H swaps the two bases. "
                   "This loads |ψ⟩ from your Part 3 inputs and applies H.")
        ab = _parse_alpha_beta()
        if ab is not None:
            c0, c1_ = _psi_coeffs(*ab)
            if _is_normalized(c0, c1_) and st.button("▶ Load |ψ⟩, apply H", key="wt5"):
                _set_system(apply_gate(_one_qubit(c0, c1_), "H", (0,)),
                            basis="comp")

    with st.expander("Part 6 — H|ψ⟩ in the diagonal basis"):
        st.latex(r"H|\psi\rangle = \left(\alpha + \tfrac{\beta}{\sqrt{2}}\right)|+\rangle"
                 r"+ \tfrac{\beta}{\sqrt{2}}|-\rangle")
        st.caption("The standard-basis coefficients of |ψ⟩ (Part 3) reappear as "
                   "the diagonal-basis coefficients of H|ψ⟩.")
        if st.button("▶ Show in diagonal basis", key="wt6"):
            _set_basis("hadamard")

    with st.expander("Part 7 — Measurement probabilities of H|ψ⟩"):
        st.latex(r"P(0) = \left|\tfrac{\alpha}{\sqrt{2}} + \beta\right|^2, \qquad "
                 r"P(1) = \left|\tfrac{\alpha}{\sqrt{2}}\right|^2")
        st.caption("Measuring H|ψ⟩ in the standard basis is the same as "
                   "measuring |ψ⟩ in the diagonal basis. Try the Measure "
                   "button in the gate row — outcomes follow these "
                   "probabilities.")
        sys: QuantumSystem = st.session_state.system
        if sys.num_qubits == 1:
            sv = sys.active_state()
            p0 = sp.simplify(sp.Abs(sv[0]) ** 2)
            p1 = sp.simplify(sp.Abs(sv[1]) ** 2)
            st.markdown("**Current simulator state:**")
            st.latex(rf"P(0) = {sp.latex(p0)} \approx {float(sp.N(p0)):.4f}")
            st.latex(rf"P(1) = {sp.latex(p1)} \approx {float(sp.N(p1)):.4f}")
        else:
            st.info("Load a 1-qubit state (Part 5) to see live probabilities.")


# ── walkthrough: Bell states & entanglement ───────────────────────────────────

_BELL_OPS = [("H", (0,)), ("CNOT", (0, 1))]

def _wt_bell():
    st.latex(r"|\Phi^+\rangle = \tfrac{|00\rangle + |11\rangle}{\sqrt{2}}"
             r"= \mathrm{CNOT}\,(H \otimes I)\,|00\rangle")

    with st.expander("Part 1 — Build a Bell state"):
        st.latex(r"|00\rangle \xrightarrow{H_0} \tfrac{(|0\rangle+|1\rangle)|0\rangle}{\sqrt{2}}"
                 r"\xrightarrow{\mathrm{CNOT}} \tfrac{|00\rangle+|11\rangle}{\sqrt{2}}")
        st.caption("After running, look at the panels: both Bloch arrows "
                   "collapse to the **center** of the sphere, the phase-disk "
                   "purity rings drop to half, and the Q-sphere shows two "
                   "antipodal points. No single-qubit state describes either "
                   "qubit — that *is* entanglement.")
        if st.button("▶ Build |Φ+⟩", key="bell1"):
            _run(2, _BELL_OPS)

    with st.expander("Part 2 — The four Bell states"):
        st.latex(r"|\Phi^\pm\rangle = \tfrac{|00\rangle \pm |11\rangle}{\sqrt{2}},"
                 r"\qquad |\Psi^\pm\rangle = \tfrac{|01\rangle \pm |10\rangle}{\sqrt{2}}")
        st.caption("Z on either qubit toggles Φ+↔Φ−; X on qubit 1 toggles "
                   "Φ↔Ψ. All four look identical on the Bloch spheres and "
                   "phase disks — only the Q-sphere phases/positions differ.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("▶ |Φ−⟩", key="bell2a"):
                _run(2, _BELL_OPS + [("Z", (0,))])
            if st.button("▶ |Ψ+⟩", key="bell2b"):
                _run(2, _BELL_OPS + [("X", (1,))])
        with c2:
            if st.button("▶ |Ψ−⟩", key="bell2c"):
                _run(2, _BELL_OPS + [("Z", (0,)), ("X", (1,))])
            if st.button("▶ |Φ+⟩", key="bell2d"):
                _run(2, _BELL_OPS)

    with st.expander("Part 3 — Measurement correlation"):
        st.latex(r"\tfrac{|00\rangle+|11\rangle}{\sqrt{2}}"
                 r"\xrightarrow{\text{measure } q_0}"
                 r"\begin{cases}|00\rangle & p=\tfrac12\\ |11\rangle & p=\tfrac12\end{cases}")
        st.caption("Measuring qubit 0 collapses **both** qubits — qubit 1 was "
                   "never touched, yet its state becomes definite. Run this "
                   "repeatedly: the outcome is random but the two qubits "
                   "always agree.")
        if st.button("▶ Build |Φ+⟩ and measure qubit 0", key="bell3"):
            s = QuantumSystem.ground(2)
            for name, qidx in _BELL_OPS:
                s = apply_gate(s, name, qidx)
            s, out, _ = measure_qubit(s, 0)
            st.session_state["last_meas"] = (0, out)
            _set_system(s, basis="comp")


# ── walkthrough: no-cloning theorem ───────────────────────────────────────────

def _wt_nocloning():
    st.markdown("**Claim:** no unitary U satisfies "
                "U(|ψ⟩|0⟩) = |ψ⟩|ψ⟩ for *every* state |ψ⟩.")

    with st.expander("Part 1 — CNOT clones basis states"):
        st.latex(r"\mathrm{CNOT}\,|0\rangle|0\rangle = |0\rangle|0\rangle,"
                 r"\qquad \mathrm{CNOT}\,|1\rangle|0\rangle = |1\rangle|1\rangle")
        st.caption("On |0⟩ and |1⟩ inputs, CNOT really does copy qubit 0 "
                   "onto qubit 1.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("▶ CNOT on |0⟩|0⟩", key="nc1a"):
                _run(2, [("CNOT", (0, 1))])
        with c2:
            if st.button("▶ CNOT on |1⟩|0⟩", key="nc1b"):
                _run(2, [("X", (0,)), ("CNOT", (0, 1))])

    with st.expander("Part 2 — …but fails on superpositions"):
        st.latex(r"\mathrm{CNOT}\,|+\rangle|0\rangle"
                 r"= \tfrac{|00\rangle+|11\rangle}{\sqrt{2}}"
                 r"\;\neq\; |+\rangle|+\rangle"
                 r"= \tfrac{|00\rangle+|01\rangle+|10\rangle+|11\rangle}{2}")
        st.caption("Instead of two copies of |+⟩ you get a Bell state — "
                   "compare the two results in every panel (entangled center "
                   "arrows vs. two clean |+⟩ arrows on the equator).")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("▶ CNOT on |+⟩|0⟩", key="nc2a"):
                _run(2, [("H", (0,)), ("CNOT", (0, 1))])
        with c2:
            if st.button("▶ Load |+⟩|+⟩", key="nc2b"):
                _run(2, [("H", (0,)), ("H", (1,))])

    with st.expander("Part 3 — Why no U can ever work"):
        st.markdown("Suppose U clones both basis states. By **linearity**:")
        st.latex(r"U(\alpha|0\rangle + \beta|1\rangle)|0\rangle"
                 r"= \alpha|00\rangle + \beta|11\rangle")
        st.markdown("But cloning requires:")
        st.latex(r"(\alpha|0\rangle{+}\beta|1\rangle)(\alpha|0\rangle{+}\beta|1\rangle)"
                 r"= \alpha^2|00\rangle + \alpha\beta|01\rangle"
                 r"+ \alpha\beta|10\rangle + \beta^2|11\rangle")
        st.markdown("These agree only when αβ = 0 — i.e. for basis states. "
                    "Linearity alone forbids universal cloning. ∎")


# ── walkthrough: Deutsch–Jozsa ────────────────────────────────────────────────

_DJ_PRE  = [("X", (1,)), ("H", (0,)), ("H", (1,))]
_DJ_POST = [("H", (0,))]
_DJ_ORACLES = {
    "f = 0 (constant)":   [],
    "f = 1 (constant)":   [("X", (1,))],
    "f(x) = x (balanced)": [("CNOT", (0, 1))],
    "f(x) = x̄ (balanced)": [("X", (1,)), ("CNOT", (0, 1))],
}

def _wt_dj():
    st.markdown("**Problem:** f: {0,1} → {0,1} is promised constant or "
                "balanced. Classically you need 2 queries; quantumly, **one**.")
    st.latex(r"|0\rangle|1\rangle \xrightarrow{H\otimes H}"
             r"|+\rangle|-\rangle \xrightarrow{U_f}"
             r"\pm|\pm\rangle|-\rangle \xrightarrow{H_0}"
             r"\pm|f(0){\oplus}f(1)\rangle|-\rangle")

    with st.expander("How it works — phase kickback"):
        st.latex(r"U_f|x\rangle|-\rangle = (-1)^{f(x)}|x\rangle|-\rangle")
        st.markdown("The ancilla in |−⟩ turns each query into a *phase*. "
                    "Constant f leaves |+⟩ alone (global phase); balanced f "
                    "flips it to |−⟩. The final H converts that to a "
                    "measurable bit: **q₀ = 0 ⇒ constant, q₀ = 1 ⇒ balanced** "
                    "— check P(q₀) in the measure row or the Bra-Ket panel.")

    st.markdown("**Run the circuit with each oracle:**")
    for name, orc in _DJ_ORACLES.items():
        if st.button(f"▶ {name}", key=f"dj_{name}"):
            _run(2, _DJ_PRE + orc + _DJ_POST)


# ── walkthrough: Grover search ────────────────────────────────────────────────

_GROVER_HH   = [("H", (0,)), ("H", (1,))]
_GROVER_XX   = [("X", (0,)), ("X", (1,))]
_GROVER_DIFF = _GROVER_HH + _GROVER_XX + [("CZ", (0, 1))] + _GROVER_XX + _GROVER_HH

def _grover_oracle(w: str) -> list:
    """Phase-flip |w⟩: conjugate CZ with X on the qubits where w has a 0."""
    pre = [("X", (q,)) for q in (0, 1) if w[q] == "0"]
    return pre + [("CZ", (0, 1))] + pre

def _wt_grover():
    st.markdown("**Problem:** one of 4 items is marked. Classically you "
                "expect ~2–3 lookups; Grover finds it in **one**.")
    st.latex(r"|s\rangle = \tfrac{1}{2}\textstyle\sum_x |x\rangle"
             r"\xrightarrow{O_w} \text{flip sign of } |w\rangle"
             r"\xrightarrow{D = 2|s\rangle\langle s| - I} |w\rangle")

    with st.expander("Why one iteration is exact here"):
        st.markdown("Each Grover iteration rotates the state by "
                    "2·arcsin(1/√N) toward |w⟩. For N = 4 that's exactly 60°, "
                    "and the start angle is 30° — one step lands precisely on "
                    "|w⟩. The simulator shows amplitude **exactly ±1**, not "
                    "0.999. (The −1 is an unobservable global phase.)")

    w = st.selectbox("Marked item w", ["00", "01", "10", "11"], key="grover_w")
    st.caption("Circuit: H⊗H · oracle(w) · diffusion. Step through it in the "
               "panels — watch the marked amplitude flip negative, then the "
               "diffusion operator amplify it to 1.")
    if st.button("▶ Run Grover", key="grover_run"):
        _run(2, _GROVER_HH + _grover_oracle(w) + _GROVER_DIFF)


# ── walkthrough: QFT ──────────────────────────────────────────────────────────

_QFT2 = [("H", (0,)), ("CS", (0, 1)), ("H", (1,)), ("SWAP", (0, 1))]

def _wt_qft():
    st.markdown("**The quantum Fourier transform** on 2 qubits maps a basis "
                "state |j⟩ to a uniform superposition with phases that "
                "*encode j in their winding rate*:")
    st.latex(r"\mathrm{QFT}|j\rangle = \tfrac{1}{2}\sum_{k=0}^{3}"
             r"e^{2\pi i\,jk/4}|k\rangle = \tfrac{1}{2}\sum_k i^{\,jk}|k\rangle")
    st.markdown("Circuit: H on q₀, controlled-S, H on q₁, SWAP.")
    st.caption("Run each input and compare on the **Q-sphere**: the "
               "amplitudes all have magnitude 1/2, but the phase colors "
               "rotate j quarter-turns per step. Exact arithmetic shows "
               "literal i and −i, not 0.7071i.")

    cols = st.columns(2)
    for i, j in enumerate(["00", "01", "10", "11"]):
        with cols[i % 2]:
            if st.button(f"▶ QFT |{j}⟩", key=f"qft_{j}"):
                init = [0, 0, 0, 0]
                init[int(j, 2)] = 1
                _run(2, _QFT2, init=init)


# ── dispatcher ────────────────────────────────────────────────────────────────

_WALKTHROUGHS = {
    "Hadamard & the diagonal basis": _wt_hadamard,
    "Bell states & entanglement":    _wt_bell,
    "No-cloning theorem":            _wt_nocloning,
    "Deutsch–Jozsa algorithm":       _wt_dj,
    "Grover search (2 qubits)":      _wt_grover,
    "Quantum Fourier transform":     _wt_qft,
}


def walkthrough_sidebar():
    st.header("📚 Walkthroughs")
    choice = st.selectbox("Example set", list(_WALKTHROUGHS), key="wt_choice")
    st.caption("Open a part, read the derivation, then click ▶ to load it "
               "into the simulator and watch the panels on the right.")
    _WALKTHROUGHS[choice]()
