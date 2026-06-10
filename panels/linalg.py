"""Linear algebra panel — SymPy symbolic equations."""
from __future__ import annotations
import sympy as sp
import streamlit as st

from basis import state_to_display_coeffs, display_coeffs_to_state
from validation import is_unitary
from quantum_core import QuantumSystem, GateOp
from format_utils import bump_sys, fmt_entry_latex, parse_matrix_code, show_latex

_BASIS_OPTIONS = ["comp", "hadamard", "circular"]
_BASIS_LABELS  = {"comp": "Computational |0⟩|1⟩",
                  "hadamard": "Hadamard |+⟩|−⟩",
                  "circular": "Circular |i⟩|−i⟩"}

def linalg_panel():
    st.subheader("Linear Algebra")

    basis = st.selectbox("Basis", _BASIS_OPTIONS, format_func=_BASIS_LABELS.get,
                         key="basis_la")
    if basis != st.session_state.basis:
        st.session_state.basis = basis
        st.session_state["_basis_sync"] = True
        bump_sys()
        st.rerun()

    sys: QuantumSystem = st.session_state.system
    n   = sys.num_qubits
    dim = 2 ** n
    exact = st.session_state.get("exact_mode", True)
    st.markdown("---")

    if sys.gate_history:
        _step_eq(sys, basis, n, exact)
    else:
        active_sv = sys.active_state()
        coeffs    = state_to_display_coeffs(active_sv, basis, n)
        _state_eq(coeffs, active_sv, basis, n, exact)

    st.markdown("---")
    with st.expander("Edit operation matrix (code)", expanded=False):
        _code_editor(sys, n, dim)


# ── LaTeX helpers ─────────────────────────────────────────────────────────────

def _v(c, exact: bool) -> str:
    """Format a SymPy entry for a LaTeX vector/matrix."""
    return fmt_entry_latex(c, exact)


def _vec_latex(vals: list[str]) -> str:
    return "\\begin{pmatrix}" + "\\\\".join(vals) + "\\end{pmatrix}"


def _mat_latex(M: sp.Matrix, exact: bool, max_dim: int = 4) -> str:
    d = M.shape[0]
    if d > max_dim:
        return f"[{d}\\!\\times\\!{d}]"
    rows = "\\\\".join(
        " & ".join(_v(M[i, j], exact) for j in range(d))
        for i in range(d))
    return "\\begin{pmatrix}" + rows + "\\end{pmatrix}"


# ── equation renderers ────────────────────────────────────────────────────────

def _state_eq(coeffs: sp.Matrix, active_sv: sp.Matrix, basis: str, n: int, exact: bool):
    dim = len(coeffs)
    # basis vectors = columns of change-of-basis matrix
    basis_vecs = [display_coeffs_to_state(_basis_vec(dim, i), basis, n) for i in range(dim)]
    if dim <= 4:
        terms = []
        for i, (c, bv) in enumerate(zip(coeffs, basis_vecs)):
            if i:
                terms.append("+")
            terms.append(_v(c, exact))
            terms.append(_vec_latex([_v(bv[j], exact) for j in range(dim)]))
        terms += ["=", _vec_latex([_v(active_sv[i], exact) for i in range(dim)])]
        st.latex(" ".join(terms))
    else:
        st.latex("|\\psi\\rangle = " + _vec_latex([_v(active_sv[i], exact) for i in range(dim)]))


def _step_eq(sys: QuantumSystem, basis: str, n: int, exact: bool):
    states = sys.intermediate_states()
    cur    = max(0, min(sys.current_step, len(sys.gate_history) - 1))
    op     = sys.gate_history[cur]
    sv_in  = states[cur]
    sv_out = states[cur + 1]
    dim    = sv_in.shape[0]

    st.markdown(f"**Step {cur+1}/{len(sys.gate_history)}:**")
    show_latex(op.label_latex())
    psi_in  = _vec_latex([_v(sv_in[i],  exact) for i in range(dim)])
    psi_out = _vec_latex([_v(sv_out[i], exact) for i in range(dim)])

    t1, t2 = st.tabs(["Compact  G·ψ = ψ′", "Expanded  Σ cᵢ·G·eᵢ = ψ′"])
    with t1:
        if dim <= 4:
            st.latex(_mat_latex(op.full_matrix, exact) + " " + psi_in + " = " + psi_out)
        else:
            st.latex(f"G \\cdot {psi_in} = {psi_out}")
    with t2:
        evs  = [_basis_vec(dim, i) for i in range(dim)]
        Gevs = [op.full_matrix * ev for ev in evs]
        if dim <= 4:
            parts = []
            for i, (c, Gev) in enumerate(zip(sv_in, Gevs)):
                if i:
                    parts.append("+")
                Gev_simp = sp.Matrix([sp.simplify(Gev[j]) for j in range(dim)])
                parts.append(_v(c, exact) + "\\cdot" +
                              _vec_latex([_v(Gev_simp[j], exact) for j in range(dim)]))
            st.latex(" ".join(parts) + " = " + psi_out)
        else:
            st.latex(f"\\sum_i c_i \\cdot G\\,e_i = {psi_out}")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("◀ Prev", key="la_prev", disabled=cur <= 0):
            sys.current_step = cur - 1
            st.rerun()
    with c2:
        if st.button("Next ▶", key="la_next",
                     disabled=cur >= len(sys.gate_history) - 1):
            sys.current_step = cur + 1
            st.rerun()


def _basis_vec(dim: int, i: int) -> sp.Matrix:
    v = sp.zeros(dim, 1)
    v[i] = sp.Integer(1)
    return v


# ── code editor ───────────────────────────────────────────────────────────────

def _code_editor(sys: QuantumSystem, n: int, dim: int):
    default_M    = sys.composed_matrix() if sys.gate_history else sp.eye(dim)
    default_code = _m2code(default_M)
    st.caption("SymPy or NumPy expression. Variables: `sp` (sympy), `np` (numpy).")
    code = st.text_area("Matrix code", value=default_code,
                        height=max(120, (dim + 4) * 18), key="la_code",
                        label_visibility="collapsed")
    if st.button("Parse & validate", key="la_parse"):
        try:
            M = parse_matrix_code(code)
        except ValueError as e:
            st.error(str(e)); return
        if M.shape != (dim, dim):
            st.error(f"Expected {dim}×{dim}, got {M.shape}"); return
        if not is_unitary(M):
            st.error("Not unitary — U†U ≠ I"); return
        new_sv = sp.Matrix([sp.simplify(x) for x in (M * sys.initial_state)])
        op     = GateOp("U", tuple(range(n)), full_matrix=M)
        st.session_state.system = QuantumSystem(
            n, new_sv, [op], 1, initial_state=sys.initial_state)
        bump_sys()
        st.success("✓ Unitary. State updated.")
        st.rerun()


def _m2code(M: sp.Matrix) -> str:
    rows = ",\n".join(
        "    [" + ", ".join(str(M[i, j]) for j in range(M.shape[1])) + "]"
        for i in range(M.shape[0]))
    return f"sp.Matrix([\n{rows}\n])"
