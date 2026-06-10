"""Bra-ket panel — SymPy symbolic coefficients."""
import sympy as sp
import streamlit as st

from basis import (state_to_display_coeffs, display_coeffs_to_state,
                   basis_ket_labels, basis_ket_labels_latex)
from quantum_core import QuantumSystem
from format_utils import bump_sys, sync_needed, fmt_expr, fmt_entry_latex, parse_expr, show_latex

_BASIS_OPTIONS = ["comp", "hadamard", "circular"]
_BASIS_LABELS  = {"comp": "Computational |0⟩|1⟩",
                  "hadamard": "Hadamard |+⟩|−⟩",
                  "circular": "Circular |i⟩|−i⟩"}


def braket_panel():
    st.subheader("Bra-Ket")

    basis = st.selectbox("Basis", _BASIS_OPTIONS, format_func=_BASIS_LABELS.get,
                         key="basis_bk")
    if basis != st.session_state.basis:
        st.session_state.basis = basis
        st.session_state["_basis_sync"] = True
        bump_sys()
        st.rerun()

    sys: QuantumSystem = st.session_state.system
    n   = sys.num_qubits
    dim = 2 ** n
    exact  = st.session_state.get("exact_mode", True)
    labels = basis_ket_labels(basis, n)

    # Always edit the INITIAL state (before gates), so steps reflect user input.
    init_coeffs = state_to_display_coeffs(sys.initial_state, basis, n)

    if sync_needed("bk"):
        for i in range(dim):
            st.session_state[f"bk_c_{i}"] = fmt_expr(init_coeffs[i], exact)

    st.markdown("---")

    if sys.gate_history:
        st.markdown("**Applied:**")
        show_latex(r" \cdot ".join(op.label_latex() for op in reversed(sys.gate_history)))

    # ── initial state inputs (always shown) ───────────────────────────────────
    new_coeffs_list: list[sp.Expr] = []
    parse_errors: list[str] = []
    TERMS_PER_ROW = 4

    for row_start in range(0, dim, TERMS_PER_ROW):
        row_end = min(row_start + TERMS_PER_ROW, dim)
        n_row   = row_end - row_start

        spec = []
        col_map = []
        for j in range(n_row):
            plus_idx = None
            if j > 0:
                plus_idx = len(spec)
                spec.append(0.25)
            input_idx = len(spec); spec.append(1.0)
            label_idx = len(spec); spec.append(0.55)
            col_map.append((input_idx, label_idx, plus_idx))
        spec.append(max(0.1, 6.0 - sum(spec)))

        cols = st.columns(spec)
        for j, idx in enumerate(range(row_start, row_end)):
            input_idx, label_idx, plus_idx = col_map[j]
            default = fmt_expr(init_coeffs[idx], exact)
            if plus_idx is not None:
                with cols[plus_idx]:
                    st.markdown(
                        "<div style='padding-top:0.4rem;text-align:center'>+</div>",
                        unsafe_allow_html=True)
            with cols[input_idx]:
                text = st.text_input("", value=default, key=f"bk_c_{idx}",
                                     label_visibility="collapsed")
            with cols[label_idx]:
                st.markdown(
                    f"<div style='padding-top:0.4rem'>{labels[idx]}</div>",
                    unsafe_allow_html=True)
            try:
                new_coeffs_list.append(parse_expr(text))
            except ValueError as e:
                parse_errors.append(f"Term {idx}: {e}")
                new_coeffs_list.append(init_coeffs[idx])

    # ── norm check ────────────────────────────────────────────────────────────
    new_sv_raw = sp.Matrix(new_coeffs_list)
    norm_sq = sp.simplify(sum(sp.Abs(new_sv_raw[i])**2 for i in range(new_sv_raw.shape[0])))
    try:
        norm_val = complex(norm_sq.evalf())
        norm_ok = abs(norm_val.real - 1) < 1e-6 and abs(norm_val.imag) < 1e-9
    except Exception:
        norm_ok = (norm_sq == sp.Integer(1))

    any_changed = not all(
        sp.simplify(new_coeffs_list[i] - init_coeffs[i]) == sp.Integer(0)
        for i in range(dim)
    )

    if parse_errors:
        for msg in parse_errors:
            st.error(msg)
    elif norm_ok:
        show_latex(r"\text{Norm: }" + sp.latex(norm_sq))
        st.success("Normalized ✓")
    else:
        show_latex(r"\text{Norm: }" + sp.latex(norm_sq))
        st.error("Must equal 1")

    if st.button("Apply", key="bk_apply",
                 disabled=not (any_changed and norm_ok and not parse_errors)):
        new_initial = display_coeffs_to_state(new_sv_raw, basis, n)
        if sys.gate_history:
            composed = sys.composed_matrix()
            new_sv = sp.Matrix([sp.simplify(x) for x in (composed * new_initial)])
            st.session_state.system = QuantumSystem(
                n, new_sv, sys.gate_history, sys.current_step,
                initial_state=new_initial)
        else:
            st.session_state.system = QuantumSystem(n, new_initial)
        bump_sys()
        st.rerun()

    # ── step-through (shown after inputs when in simplify mode) ───────────────
    if sys.gate_history:
        st.markdown("---")
        _show_steps(sys, basis, n, exact)


# ── step-through ──────────────────────────────────────────────────────────────

def _show_steps(sys: QuantumSystem, basis: str, n: int, exact: bool):
    labels_latex = basis_ket_labels_latex(basis, n)
    states = sys.intermediate_states()
    cur    = sys.current_step if sys.current_step >= 0 else len(states) - 1

    lines = []
    for i, op in enumerate(sys.gate_history):
        coeffs = state_to_display_coeffs(states[i], basis, n)
        expr   = _ket_expr_latex(coeffs, labels_latex, exact)
        ops    = r" \cdot ".join(g.label_latex() for g in reversed(sys.gate_history[i:]))
        arrow  = r"\rightarrow" if i == cur else r"\phantom{\rightarrow}"
        lines.append(rf"{arrow} & {ops}\!\left({expr}\right)")

    final = state_to_display_coeffs(states[-1], basis, n)
    arrow = r"\rightarrow" if cur >= len(sys.gate_history) else r"\phantom{\rightarrow}"
    lines.append(rf"{arrow} & = {_ket_expr_latex(final, labels_latex, exact)}")

    show_latex(r"\begin{aligned}" + r" \\ ".join(lines) + r"\end{aligned}")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("◀ Prev", key="bk_prev", disabled=cur <= 0):
            sys.current_step = cur - 1
            st.rerun()
    with c2:
        if st.button("Next ▶", key="bk_next",
                     disabled=cur >= len(sys.gate_history)):
            sys.current_step = cur + 1
            st.rerun()


def _ket_expr_latex(coeffs: sp.Matrix, labels_latex: list[str], exact: bool, thr=1e-9) -> str:
    terms = []
    for i, lbl in enumerate(labels_latex):
        c = coeffs[i]
        try:
            if abs(complex(c.evalf())) < thr:
                continue
        except Exception:
            pass
        terms.append(rf"{fmt_entry_latex(c, exact)}{lbl}")
    return " + ".join(terms) if terms else "0"
