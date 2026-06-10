"""
State and matrix validation.

Note: this SymPy version's MutableDenseMatrix lacks `.conj()`, so `.H` and
`applyfunc(sp.conjugate)` raise AttributeError. Matrix conjugate-transposes
are therefore done numerically (NumPy); vector norms use element-wise sp.Abs.
"""
import numpy as np
import sympy as sp


def _to_numpy(M) -> np.ndarray:
    """Evaluate a SymPy matrix (or array-like) to a complex NumPy array."""
    if isinstance(M, sp.MatrixBase):
        return np.array(M.evalf().tolist(), dtype=complex)
    return np.asarray(M, dtype=complex)


def is_unitary(M, tol: float = 1e-9) -> bool:
    """Check U†U = I numerically."""
    A = _to_numpy(M)
    n = A.shape[0]
    return np.allclose(A.conj().T @ A, np.eye(n), atol=tol)


def is_normalized(sv, tol: float = 1e-9) -> bool:
    """Check Σ|cᵢ|² = 1."""
    norm_sq = sp.simplify(sum(sp.Abs(sv[i]) ** 2 for i in range(sv.shape[0])))
    try:
        return abs(complex(norm_sq.evalf()) - 1) < tol
    except Exception:
        return norm_sq == sp.Integer(1)


def normalize(sv: sp.Matrix) -> sp.Matrix:
    """Return sv / ‖sv‖ with exact symbolic entries."""
    norm_sq = sp.simplify(sum(sp.Abs(sv[i]) ** 2 for i in range(sv.shape[0])))
    norm = sp.sqrt(norm_sq)
    if norm == 0:
        raise ValueError("Zero vector cannot be normalized.")
    return sp.Matrix([sp.simplify(sv[i] / norm) for i in range(sv.shape[0])])
