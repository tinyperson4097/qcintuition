"""
Basis conversion utilities — exact symbolic (SymPy).

Display bases:
  "comp"     — {|0⟩, |1⟩}     standard computational
  "hadamard" — {|+⟩, |−⟩}     H-eigenstates
  "circular" — {|i⟩, |−i⟩}    Y-eigenstates

Change-of-basis matrices have the display-basis kets as columns
(in computational coordinates). For n qubits: n-fold Kronecker product.
"""
import sympy as sp
from gates import SINGLE_QUBIT
from quantum_core import kron_sp

_H = SINGLE_QUBIT["H"]

# Columns are the basis kets expressed in the computational basis,
# e.g. circular: |i⟩ = (|0⟩ + i|1⟩)/√2, |−i⟩ = (|0⟩ − i|1⟩)/√2.
_TO_DISPLAY: dict[str, sp.Matrix] = {
    "comp":     sp.eye(2),
    "hadamard": _H,
    "circular": sp.Matrix([[1, 1], [sp.I, -sp.I]]) / sp.sqrt(2),
}

_LATEX_LABEL_FN = {
    "comp":     lambda bits: f"|{bits}\\rangle",
    "hadamard": lambda bits: "|" + "".join("{+}" if b == "0" else "{-}" for b in bits) + "\\rangle",
    "circular": lambda bits: "|" + "".join("i" if b == "0" else "{-i}" for b in bits) + "\\rangle",
}

_UNICODE_LABEL_FN = {
    "comp":     lambda bits: "|" + bits + "⟩",
    "hadamard": lambda bits: "|" + bits.replace("0", "+").replace("1", "−") + "⟩",
    "circular": lambda bits: "|" + bits.replace("0", "i").replace("1", "−i") + "⟩",
}


# Cache U and U⁻¹ per (basis, n) — symbolic inversion is expensive and
# these are requested on every Streamlit rerun by all three panels.
_CHANGE_CACHE: dict[tuple[str, int], tuple[sp.Matrix, sp.Matrix]] = {}


def _change_matrices(basis: str, n_qubits: int) -> tuple[sp.Matrix, sp.Matrix]:
    """Return (U, U⁻¹) for the n-qubit change-of-basis, cached."""
    key = (basis, n_qubits)
    if key not in _CHANGE_CACHE:
        U = _TO_DISPLAY[basis]
        result = U
        for _ in range(n_qubits - 1):
            result = kron_sp(result, U)
        _CHANGE_CACHE[key] = (result, result.inv())
    return _CHANGE_CACHE[key]


def state_to_display_coeffs(sv: sp.Matrix, basis: str, n_qubits: int) -> sp.Matrix:
    """Project comp-basis state vector into display basis coefficients."""
    if basis == "comp":
        return sp.Matrix(sv)
    _, U_inv = _change_matrices(basis, n_qubits)
    result = U_inv * sv
    return sp.Matrix([sp.simplify(result[i]) for i in range(result.shape[0])])


def display_coeffs_to_state(coeffs: sp.Matrix, basis: str, n_qubits: int) -> sp.Matrix:
    """Inverse of state_to_display_coeffs."""
    if basis == "comp":
        return sp.Matrix(coeffs)
    U, _ = _change_matrices(basis, n_qubits)
    result = U * coeffs
    return sp.Matrix([sp.simplify(result[i]) for i in range(result.shape[0])])


def basis_ket_labels_latex(basis: str, n_qubits: int) -> list[str]:
    fn = _LATEX_LABEL_FN[basis]
    return [fn(format(i, f"0{n_qubits}b")) for i in range(2 ** n_qubits)]


def basis_ket_labels(basis: str, n_qubits: int) -> list[str]:
    fn = _UNICODE_LABEL_FN[basis]
    return [fn(format(i, f"0{n_qubits}b")) for i in range(2 ** n_qubits)]
