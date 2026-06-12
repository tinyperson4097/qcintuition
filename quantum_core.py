"""
Core quantum state and gate logic. No UI dependencies.

Conventions
-----------
- Big-endian qubit ordering: qubit 0 is the most-significant bit.
- State vectors are SymPy column matrices (shape 2^n × 1).
- Gate matrices are SymPy matrices (square, size 2^n × 2^n after expansion).
- All arithmetic is exact symbolic (SymPy).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import sympy as sp
from gates import (SINGLE_QUBIT, TWO_QUBIT, THREE_QUBIT,
                   ROTATION_GATES, phase_gate, rotation_gate)
from validation import is_unitary, normalize
from format_utils import fmt_angle_latex


# ── Kronecker product ─────────────────────────────────────────────────────────

def kron_sp(A: sp.Matrix, B: sp.Matrix) -> sp.Matrix:
    """SymPy Kronecker (tensor) product."""
    m, n = A.shape
    p, q = B.shape
    result = sp.zeros(m * p, n * q)
    for i in range(m):
        for j in range(n):
            result[i*p:(i+1)*p, j*q:(j+1)*q] = A[i, j] * B
    return result


# ── GateOp ────────────────────────────────────────────────────────────────────

@dataclass
class GateOp:
    name: str
    qubit_indices: tuple
    phi: object = sp.Integer(0)   # SymPy expression or 0
    full_matrix: sp.Matrix = field(default=None, repr=False)

    def label_latex(self) -> str:
        idx = ",".join(str(q) for q in self.qubit_indices)
        name = self.name
        if name == "P":
            angle = fmt_angle_latex(self.phi)
            return f"P_{{{idx}}}({angle})"
        if name in ROTATION_GATES:
            axis  = name[1]            # "Rx" → "x"
            angle = fmt_angle_latex(self.phi)
            return f"R_{{{axis},{idx}}}({angle})"
        if name in ("M0", "M1"):
            return rf"M_{{{idx}}}{{=}}{name[1]}"
        if name.endswith("†"):
            base = name[:-1]
            return (f"{base}^\\dagger_{{{idx}}}" if len(base) == 1
                    else f"\\text{{{base}}}^\\dagger_{{{idx}}}")
        if len(name) == 1:
            return f"{name}_{{{idx}}}"
        return f"\\text{{{name}}}_{{{idx}}}"


# ── QuantumSystem ─────────────────────────────────────────────────────────────

@dataclass
class QuantumSystem:
    num_qubits: int
    state_vector: sp.Matrix            # CURRENT state (after gates), comp basis
    gate_history: list[GateOp] = field(default_factory=list)
    current_step: int = -1
    initial_state: sp.Matrix = None    # state BEFORE gates; defaults to state_vector
    # Per-instance cache: instances are replaced (never mutated) when state or
    # history changes, so caching derived states across reruns is safe.
    _states_cache: list = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.initial_state is None:
            self.initial_state = sp.Matrix(self.state_vector)

    @classmethod
    def ground(cls, num_qubits: int = 1) -> "QuantumSystem":
        """Initialise to |00…0⟩."""
        dim = 2 ** num_qubits
        sv = sp.zeros(dim, 1)
        sv[0] = sp.Integer(1)
        return cls(num_qubits=num_qubits, state_vector=sv)

    @property
    def dim(self) -> int:
        return 2 ** self.num_qubits

    def intermediate_states(self) -> list[sp.Matrix]:
        """[initial, after gate 1, …, final] — computed once, then cached."""
        if self._states_cache is None:
            sv = self.initial_state
            states = [sv]
            for op in self.gate_history:
                sv = _simp_vec(op.full_matrix * sv)
                states.append(sv)
            self._states_cache = states
        return self._states_cache

    def active_state(self) -> sp.Matrix:
        """State at the current step-through position (final if step < 0)."""
        if self.current_step < 0 or not self.gate_history:
            return self.state_vector
        states = self.intermediate_states()
        idx = min(self.current_step, len(states) - 1)
        return states[idx]

    def composed_matrix(self) -> sp.Matrix:
        """Product of all gates: history [G1, G2] composes to G2·G1
        (rightmost factor acts first)."""
        result = sp.eye(self.dim)
        for op in self.gate_history:
            result = op.full_matrix * result
        return result


def _simp_vec(v: sp.Matrix) -> sp.Matrix:
    """Simplify each element of a column vector.

    Full sp.simplify is fast for 1-2 qubits but drags at dim 8/16, so larger
    vectors get a lighter expand/powsimp/radsimp pass instead.
    """
    if v.shape[0] <= 4:
        return sp.Matrix([sp.simplify(v[i]) for i in range(v.shape[0])])
    return sp.Matrix([_light_simp(v[i]) for i in range(v.shape[0])])


def _light_simp(e: sp.Expr) -> sp.Expr:
    """Cheap simplification: combines products of roots/exponentials and
    cancels expanded sums without the full simplify search."""
    e = sp.expand(e)
    e = sp.powsimp(e)
    return sp.radsimp(e)


# ── gate expansion ────────────────────────────────────────────────────────────

def build_full_gate(
    gate_matrix: sp.Matrix,
    qubit_indices: tuple[int, ...],
    n_qubits: int,
) -> sp.Matrix:
    n = n_qubits
    k = len(qubit_indices)

    if k == 1:
        q = qubit_indices[0]
        factors = [gate_matrix if i == q else sp.eye(2) for i in range(n)]
        result = factors[0]
        for f in factors[1:]:
            result = kron_sp(result, f)
        return result

    # Multi-qubit: bring target qubits to front, apply gate, restore order
    other_qubits = [q for q in range(n) if q not in qubit_indices]
    perm = list(qubit_indices) + other_qubits

    rest_dim = 2 ** (n - k)
    full_in_k_space = kron_sp(gate_matrix, sp.eye(rest_dim))
    P = _permutation_matrix(perm, n)
    # A permutation matrix is orthogonal: P⁻¹ = Pᵀ (much cheaper than .inv()).
    return P.T * full_in_k_space * P


def _permutation_matrix(qubit_order: list[int], n: int) -> sp.Matrix:
    dim = 2 ** n
    P = sp.zeros(dim, dim)
    for i in range(dim):
        bits = [(i >> (n - 1 - qubit_order[j])) & 1 for j in range(n)]
        j = sum(b << (n - 1 - pos) for pos, b in enumerate(bits))
        P[j, i] = sp.Integer(1)
    return P


# ── Bloch sphere ──────────────────────────────────────────────────────────────

def state_to_bloch(
    sv: sp.Matrix,
    qubit_index: int,
    n_qubits: int,
) -> tuple[float, float, float]:
    """Return (rx, ry, rz) as Python floats for Plotly."""
    rho = _reduced_density_matrix(sv, qubit_index, n_qubits)
    sx = sp.Matrix([[0, 1], [1, 0]])
    sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sz = sp.Matrix([[1, 0], [0, -1]])
    def component(sigma):
        return float(sp.re(sp.simplify((rho * sigma).trace())).evalf())
    return component(sx), component(sy), component(sz)


def bloch_to_state(theta, phi) -> sp.Matrix:
    t = sp.sympify(theta)
    p = sp.sympify(phi)
    alpha = sp.cos(t / 2)
    beta  = sp.exp(sp.I * p) * sp.sin(t / 2)
    return sp.Matrix([sp.simplify(alpha), sp.simplify(beta)])


def bloch_to_angles(rx: float, ry: float, rz: float) -> tuple[float, float]:
    import math
    r = math.sqrt(rx**2 + ry**2 + rz**2)
    if r < 1e-12:
        return 0.0, 0.0
    theta = math.acos(max(-1.0, min(1.0, rz / r)))
    phi = math.atan2(ry, rx)
    if phi < 0:
        phi += 2 * math.pi
    return theta, phi


def _reduced_density_matrix(sv: sp.Matrix, qubit_index: int, n_qubits: int) -> sp.Matrix:
    n = n_qubits
    bit_pos = n - 1 - qubit_index
    dim_rest = 2 ** n // 2
    rho = sp.zeros(2, 2)
    for alpha in range(2):
        for beta in range(2):
            for k in range(dim_rest):
                ia = _full_idx(alpha, k, bit_pos, n)
                ib = _full_idx(beta,  k, bit_pos, n)
                rho[alpha, beta] += sv[ia] * sp.conjugate(sv[ib])
    return sp.Matrix([[sp.simplify(rho[i, j]) for j in range(2)] for i in range(2)])


def _full_idx(alpha: int, rest_val: int, bit_pos: int, n: int) -> int:
    result = alpha << bit_pos
    rest_bit = 0
    for pos in range(n):
        if pos != bit_pos:
            if (rest_val >> rest_bit) & 1:
                result |= (1 << pos)
            rest_bit += 1
    return result


# ── high-level operations ─────────────────────────────────────────────────────

def apply_gate(
    system: QuantumSystem,
    gate_name: str,
    qubit_indices: tuple[int, ...],
    phi=sp.Integer(0),
) -> QuantumSystem:
    if gate_name in SINGLE_QUBIT:
        raw = SINGLE_QUBIT[gate_name]
    elif gate_name == "P":
        raw = phase_gate(phi)
    elif gate_name in ROTATION_GATES:
        raw = rotation_gate(gate_name, phi)   # phi carries θ for R-gates
    elif gate_name in TWO_QUBIT:
        raw = TWO_QUBIT[gate_name]
    elif gate_name in THREE_QUBIT:
        raw = THREE_QUBIT[gate_name]
    else:
        raise ValueError(f"Unknown gate: {gate_name}")

    full   = build_full_gate(raw, qubit_indices, system.num_qubits)
    new_sv = _simp_vec(full * system.state_vector)
    op     = GateOp(name=gate_name, qubit_indices=qubit_indices,
                    phi=phi, full_matrix=full)
    return QuantumSystem(
        num_qubits=system.num_qubits,
        state_vector=new_sv,
        gate_history=system.gate_history + [op],
        current_step=system.current_step,
        initial_state=system.initial_state,
    )


def apply_custom_gate(
    system: QuantumSystem,
    matrix: sp.Matrix,
    qubit_indices: tuple[int, ...],
) -> QuantumSystem:
    """Apply a user-supplied unitary to the given qubit(s).

    The matrix must be 2^k × 2^k for k target qubits and pass U†U = I.
    """
    k = len(qubit_indices)
    expected = 2 ** k
    if matrix.shape != (expected, expected):
        raise ValueError(
            f"Expected a {expected}×{expected} matrix for {k} qubit(s), "
            f"got {matrix.shape[0]}×{matrix.shape[1]}")
    if not is_unitary(matrix):
        raise ValueError("Matrix is not unitary: U†U ≠ I")

    full   = build_full_gate(matrix, qubit_indices, system.num_qubits)
    new_sv = _simp_vec(full * system.state_vector)
    op     = GateOp(name="U", qubit_indices=qubit_indices, full_matrix=full)
    return QuantumSystem(
        num_qubits=system.num_qubits,
        state_vector=new_sv,
        gate_history=system.gate_history + [op],
        current_step=system.current_step,
        initial_state=system.initial_state,
    )


def measurement_probs(system: QuantumSystem, qubit_index: int) -> tuple:
    """Exact (P0, P1) for measuring one qubit in the computational basis."""
    rho = _reduced_density_matrix(system.state_vector, qubit_index,
                                  system.num_qubits)
    return (sp.simplify(sp.re(rho[0, 0])), sp.simplify(sp.re(rho[1, 1])))


def measure_qubit(
    system: QuantumSystem,
    qubit_index: int,
    outcome: int | None = None,
) -> tuple[QuantumSystem, int, tuple]:
    """Projective measurement of one qubit in the computational basis.

    Returns (new_system, outcome, (P0, P1)). With outcome=None the result is
    sampled from the Born probabilities. The collapse is recorded as a GateOp
    whose matrix is the projector scaled by 1/√p — linear though not unitary,
    which keeps step-through and undo working.
    """
    probs = measurement_probs(system, qubit_index)
    if outcome is None:
        import random
        outcome = 1 if random.random() < float(probs[1]) else 0
    p = probs[outcome]
    if float(p) < 1e-12:
        raise ValueError(f"Outcome {outcome} has probability 0")
    proj = sp.zeros(2, 2)
    proj[outcome, outcome] = sp.Integer(1)
    M = build_full_gate(proj, (qubit_index,), system.num_qubits) / sp.sqrt(p)
    new_sv = _simp_vec(M * system.state_vector)
    op = GateOp(name=f"M{outcome}", qubit_indices=(qubit_index,),
                full_matrix=M)
    return QuantumSystem(
        num_qubits=system.num_qubits,
        state_vector=new_sv,
        gate_history=system.gate_history + [op],
        current_step=system.current_step,
        initial_state=system.initial_state,
    ), outcome, probs


def set_state_from_bloch(
    system: QuantumSystem,
    qubit_index: int,
    theta,
    phi,
) -> QuantumSystem:
    new_qubit = bloch_to_state(theta, phi)
    if system.num_qubits == 1:
        return QuantumSystem(num_qubits=1, state_vector=new_qubit,
                             gate_history=[], current_step=-1)
    states = []
    for q in range(system.num_qubits):
        if q == qubit_index:
            states.append(new_qubit)
        else:
            states.append(sp.Matrix([sp.Integer(1), sp.Integer(0)]))
    sv = states[0]
    for s in states[1:]:
        sv = kron_sp(sv, s)
    return QuantumSystem(num_qubits=system.num_qubits, state_vector=_simp_vec(sv),
                         gate_history=[], current_step=-1)
