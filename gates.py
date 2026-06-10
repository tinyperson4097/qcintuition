"""
Gate matrices as SymPy Matrix objects — exact symbolic arithmetic.
"""
import sympy as sp

_S2 = sp.sqrt(2)

SINGLE_QUBIT: dict[str, sp.Matrix] = {
    "I":  sp.eye(2),
    "X":  sp.Matrix([[0, 1], [1, 0]]),
    "Y":  sp.Matrix([[0, -sp.I], [sp.I, 0]]),
    "Z":  sp.Matrix([[1, 0], [0, -1]]),
    "H":  sp.Matrix([[1, 1], [1, -1]]) / _S2,
    "S":  sp.Matrix([[1, 0], [0, sp.I]]),
    "S†": sp.Matrix([[1, 0], [0, -sp.I]]),
    "T":  sp.Matrix([[1, 0], [0, sp.exp(sp.I * sp.pi / 4)]]),
    "T†": sp.Matrix([[1, 0], [0, sp.exp(-sp.I * sp.pi / 4)]]),
}

TWO_QUBIT: dict[str, sp.Matrix] = {
    "CNOT": sp.Matrix([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]),
    "CZ":   sp.Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]]),
    "SWAP": sp.Matrix([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]),
}

_ccnot = sp.eye(8)
_ccnot[6, 6] = sp.Integer(0); _ccnot[7, 7] = sp.Integer(0)
_ccnot[6, 7] = sp.Integer(1); _ccnot[7, 6] = sp.Integer(1)
THREE_QUBIT: dict[str, sp.Matrix] = {"CCNOT": _ccnot}


def phase_gate(phi) -> sp.Matrix:
    """P(φ) = diag(1, e^{iφ})"""
    phi_s = sp.sympify(phi)
    return sp.Matrix([[sp.Integer(1), sp.Integer(0)],
                      [sp.Integer(0), sp.exp(sp.I * phi_s)]])


ROTATION_GATES = ("Rx", "Ry", "Rz")

def rotation_gate(name: str, theta) -> sp.Matrix:
    """Rotation by θ around the x/y/z Bloch axis: R_a(θ) = e^{-iθ·σ_a/2}."""
    t = sp.sympify(theta) / 2
    if name == "Rx":
        return sp.Matrix([[sp.cos(t),          -sp.I * sp.sin(t)],
                          [-sp.I * sp.sin(t),   sp.cos(t)]])
    if name == "Ry":
        return sp.Matrix([[sp.cos(t), -sp.sin(t)],
                          [sp.sin(t),  sp.cos(t)]])
    if name == "Rz":
        return sp.Matrix([[sp.exp(-sp.I * t), sp.Integer(0)],
                          [sp.Integer(0),     sp.exp(sp.I * t)]])
    raise ValueError(f"Unknown rotation gate: {name}")


ALL_GATE_NAMES = (list(SINGLE_QUBIT) + ["P"] + list(ROTATION_GATES)
                  + list(TWO_QUBIT) + list(THREE_QUBIT) + ["Custom"])

BLOCH_EFFECT = {
    "I":    "Identity — no change to state",
    "X":    "π rotation around X-axis  (bit-flip: |0⟩ ↔ |1⟩)",
    "Y":    "π rotation around Y-axis  (bit-flip + phase-flip)",
    "Z":    "π rotation around Z-axis  (phase-shift: |1⟩ → −|1⟩)",
    "H":    "π rotation around (X+Z)/√2 axis — maps |0⟩ to |+⟩",
    "S":    "π/2 rotation around Z-axis",
    "S†":   "−π/2 rotation around Z-axis",
    "T":    "π/4 rotation around Z-axis",
    "T†":   "−π/4 rotation around Z-axis",
    "P":    "φ rotation around Z-axis (depends on φ parameter)",
    "Rx":   "θ rotation around X-axis",
    "Ry":   "θ rotation around Y-axis",
    "Rz":   "θ rotation around Z-axis",
    "Custom": "User-defined 2×2 unitary on the selected qubit",
    "U":    "User-defined unitary",
    "CNOT": "Conditional X on target when control = |1⟩",
    "CZ":   "Conditional Z on target when control = |1⟩",
    "SWAP": "Swaps the two qubit states",
    "CCNOT":"Toffoli: conditional X when both controls = |1⟩",
}
