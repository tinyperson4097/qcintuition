"""
Shared formatting utilities — SymPy-based exact arithmetic.
"""
import sympy as sp


# ── expression parsing ────────────────────────────────────────────────────────

_SYMPIFY_LOCALS = {
    "pi": sp.pi, "I": sp.I, "sqrt": sp.sqrt,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "exp": sp.exp, "Rational": sp.Rational, "oo": sp.oo,
}

def parse_expr(s: str) -> sp.Expr:
    """Parse a user math string into a SymPy expression.

    Accepts: pi/2, sqrt(2)/2, 1/sqrt(2), (1+I)/sqrt(2), 0.707, ^, etc.
    """
    s = s.strip().replace("^", "**").replace("√", "sqrt")
    try:
        return sp.sympify(s, locals=_SYMPIFY_LOCALS)
    except Exception as exc:
        raise ValueError(f"Cannot parse '{s}': {exc}") from exc


# ── display formatting ────────────────────────────────────────────────────────

def fmt_expr(c, exact: bool = True) -> str:
    """Human-readable string for text inputs — parseable by parse_expr."""
    if not isinstance(c, sp.Basic):
        c = sp.sympify(c)
    c = sp.simplify(c)
    if exact:
        return str(c)
    try:
        val = complex(c.evalf())
        if abs(val.imag) < 1e-10:
            return f"{val.real:.4f}"
        return f"{val.real:.4f}+{val.imag:.4f}*I" if val.imag >= 0 else f"{val.real:.4f}{val.imag:.4f}*I"
    except Exception:
        return str(c)


def fmt_entry_latex(c, exact: bool = True) -> str:
    """LaTeX string for equation display."""
    if not isinstance(c, sp.Basic):
        c = sp.sympify(c)
    c = sp.simplify(c)
    if exact:
        return sp.latex(c)
    try:
        val = complex(c.evalf())
        if abs(val.imag) < 1e-10:
            return f"{val.real:.2f}"
        sign = "+" if val.imag >= 0 else ""
        return f"{val.real:.2f}{sign}{val.imag:.2f}i"
    except Exception:
        return sp.latex(c)


def fmt_angle_latex(v, exact: bool = True) -> str:
    """LaTeX string for an angle in radians — recognises π-fractions from floats."""
    if not isinstance(v, sp.Basic):
        # nsimplify with pi as a candidate finds exact forms like π/2, 3π/4, etc.
        try:
            v = sp.nsimplify(float(v), [sp.pi], rational=False, tolerance=1e-5)
        except Exception:
            v = sp.sympify(v)
    return fmt_entry_latex(v, exact)


def parse_matrix_code(code: str) -> sp.Matrix:
    """Evaluate user matrix code with `sp`/`np` in scope; return sp.Matrix.

    Builtins are disabled; only sympy/numpy expressions are usable.
    """
    import numpy as np
    try:
        raw = eval(code, {"sp": sp, "sympy": sp, "np": np, "numpy": np,
                          "__builtins__": {}})  # noqa: S307
    except Exception as exc:
        raise ValueError(f"Parse error: {exc}") from exc
    try:
        return sp.Matrix(raw)
    except Exception as exc:
        raise ValueError(f"Cannot convert to a SymPy matrix: {exc}") from exc


# ── latex rendering ──────────────────────────────────────────────────────────
# streamlit is imported lazily below so that core modules (quantum_core,
# basis, validation) and the test suite can import format_utils without it.

def show_latex(expr) -> None:
    """Render a SymPy expression or raw LaTeX string via st.latex()."""
    import streamlit as st
    if isinstance(expr, sp.Basic):
        st.latex(sp.latex(expr))
    elif isinstance(expr, str):
        st.latex(expr)
    else:
        st.latex(sp.latex(sp.sympify(expr)))


# ── system version tracking ───────────────────────────────────────────────────

def bump_sys():
    import streamlit as st
    st.session_state["_sys_ver"] = st.session_state.get("_sys_ver", 0) + 1


def sync_needed(tag: str) -> bool:
    """Return True (and record sync) if the system changed since last panel sync."""
    import streamlit as st
    cur  = st.session_state.get("_sys_ver", 0)
    last = st.session_state.get(f"_ver_{tag}", -1)
    if cur != last:
        st.session_state[f"_ver_{tag}"] = cur
        return True
    return False
