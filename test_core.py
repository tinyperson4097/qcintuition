"""Core test suite — exact SymPy backend, asserted numerically via NumPy."""
import numpy as np
import sympy as sp
from validation import is_unitary, is_normalized, _to_numpy
from gates import (SINGLE_QUBIT, TWO_QUBIT, THREE_QUBIT,
                   ROTATION_GATES, phase_gate, rotation_gate)
from basis import state_to_display_coeffs, display_coeffs_to_state, basis_ket_labels
from quantum_core import (
    QuantumSystem, apply_gate, apply_custom_gate, build_full_gate,
    state_to_bloch, bloch_to_state, bloch_to_angles,
)


def as_np(M) -> np.ndarray:
    """Flatten a SymPy matrix (or array-like) to a 1-D complex NumPy array."""
    return _to_numpy(M).ravel()


# ── validation ────────────────────────────────────────────────────────────────

def test_all_gates_unitary():
    all_gates = {**SINGLE_QUBIT, **TWO_QUBIT, **THREE_QUBIT}
    for name, M in all_gates.items():
        assert is_unitary(M), f"{name} is not unitary"

def test_phase_gate_unitary():
    for phi in [0, np.pi/4, np.pi/2, np.pi, 2*np.pi]:
        assert is_unitary(phase_gate(phi))

def test_non_unitary_rejected():
    """Invertible but non-unitary matrices must be rejected.

    Regression test: a previous implementation checked M⁻¹M = I,
    which holds for ANY invertible matrix.
    """
    assert not is_unitary(sp.Matrix([[2, 0], [0, 1]]))      # invertible, not unitary
    assert not is_unitary(sp.Matrix([[1, 1], [0, 1]]))      # shear
    assert not is_unitary(np.array([[1, 0], [0, 2]], dtype=complex))

def test_normalization():
    assert is_normalized(sp.Matrix([1, 1]) / sp.sqrt(2))
    assert not is_normalized(sp.Matrix([1, 1]))


# ── rotation gates ────────────────────────────────────────────────────────────

def test_rotation_gates_unitary():
    for name in ROTATION_GATES:
        for theta in [0, sp.pi/4, sp.pi/2, sp.pi, 2*sp.pi]:
            assert is_unitary(rotation_gate(name, theta)), f"{name}({theta})"

def test_rx_pi_is_minus_iX():
    """Rx(π) = −iX."""
    expected = -1j * _to_numpy(SINGLE_QUBIT["X"])
    assert np.allclose(_to_numpy(rotation_gate("Rx", sp.pi)), expected)

def test_ry_pi_flips_ground():
    """Ry(π)|0⟩ = |1⟩ (no phase)."""
    sys = apply_gate(QuantumSystem.ground(1), "Ry", (0,), phi=sp.pi)
    assert np.allclose(as_np(sys.state_vector), [0, 1])

def test_rz_pi_half_is_S_up_to_phase():
    """Rz(π/2) = e^{−iπ/4}·S."""
    expected = np.exp(-1j*np.pi/4) * _to_numpy(SINGLE_QUBIT["S"])
    assert np.allclose(_to_numpy(rotation_gate("Rz", sp.pi/2)), expected)

def test_rotation_exact_form():
    """Rx(π/4) entries must stay symbolic — no floats."""
    M = rotation_gate("Rx", sp.pi/4)
    for entry in M:
        assert not entry.atoms(sp.Float), f"Float leaked: {entry}"


# ── custom gate ───────────────────────────────────────────────────────────────

def test_custom_gate_applies():
    H = sp.Matrix([[1, 1], [1, -1]]) / sp.sqrt(2)
    sys = apply_custom_gate(QuantumSystem.ground(1), H, (0,))
    assert np.allclose(as_np(sys.state_vector), [1/np.sqrt(2), 1/np.sqrt(2)])
    assert sys.gate_history[-1].name == "U"

def test_custom_gate_rejects_non_unitary():
    import pytest
    with pytest.raises(ValueError, match="not unitary"):
        apply_custom_gate(QuantumSystem.ground(1), sp.Matrix([[2, 0], [0, 1]]), (0,))

def test_custom_gate_rejects_wrong_shape():
    import pytest
    with pytest.raises(ValueError, match="Expected"):
        apply_custom_gate(QuantumSystem.ground(1), sp.eye(4), (0,))


# ── single-qubit gate application ─────────────────────────────────────────────

def test_X_flips_ground():
    sys = QuantumSystem.ground(1)
    sys2 = apply_gate(sys, "X", (0,))
    assert np.allclose(as_np(sys2.state_vector), [0, 1])

def test_H_creates_superposition():
    sys = QuantumSystem.ground(1)
    sys2 = apply_gate(sys, "H", (0,))
    assert np.allclose(as_np(sys2.state_vector), [1/np.sqrt(2), 1/np.sqrt(2)])

def test_HH_is_identity():
    sv = sp.Matrix([sp.Rational(3, 5), sp.Rational(4, 5)])
    sys = QuantumSystem(1, sv)
    sys2 = apply_gate(apply_gate(sys, "H", (0,)), "H", (0,))
    assert np.allclose(as_np(sys2.state_vector), [0.6, 0.8])

def test_ZZ_is_identity():
    sv = sp.Matrix([sp.Rational(3, 5), sp.Rational(4, 5) * sp.I])
    sys = QuantumSystem(1, sv)
    sys2 = apply_gate(apply_gate(sys, "Z", (0,)), "Z", (0,))
    assert np.allclose(as_np(sys2.state_vector), [0.6, 0.8j])


# ── multi-qubit gate expansion ────────────────────────────────────────────────

def test_I_on_qubit1_of_2():
    """Identity on qubit 1 of a 2-qubit system = full identity."""
    I2 = build_full_gate(SINGLE_QUBIT["I"], (1,), 2)
    assert np.allclose(_to_numpy(I2), np.eye(4))

def test_X_on_qubit0_of_2():
    """X on qubit 0 (MSB) of |00⟩ → |10⟩ (index 2)."""
    sys = QuantumSystem.ground(2)
    sys2 = apply_gate(sys, "X", (0,))
    assert np.allclose(as_np(sys2.state_vector), [0, 0, 1, 0])

def test_X_on_qubit1_of_2():
    """X on qubit 1 (LSB) of |00⟩ → |01⟩ (index 1)."""
    sys = QuantumSystem.ground(2)
    sys2 = apply_gate(sys, "X", (1,))
    assert np.allclose(as_np(sys2.state_vector), [0, 1, 0, 0])

def test_CNOT_entangles():
    """H on qubit 0 then CNOT → Bell state (|00⟩+|11⟩)/√2."""
    sys = QuantumSystem.ground(2)
    sys = apply_gate(sys, "H", (0,))
    sys = apply_gate(sys, "CNOT", (0, 1))
    bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
    assert np.allclose(as_np(sys.state_vector), bell)


# ── Bloch sphere ──────────────────────────────────────────────────────────────

def test_bloch_ground_state():
    rx, ry, rz = state_to_bloch(sp.Matrix([1, 0]), 0, 1)
    assert np.isclose(rz,  1.0)
    assert np.isclose(rx,  0.0)
    assert np.isclose(ry,  0.0)

def test_bloch_excited_state():
    rx, ry, rz = state_to_bloch(sp.Matrix([0, 1]), 0, 1)
    assert np.isclose(rz, -1.0)

def test_bloch_plus_state():
    sv = sp.Matrix([1, 1]) / sp.sqrt(2)
    rx, ry, rz = state_to_bloch(sv, 0, 1)
    assert np.isclose(rx, 1.0, atol=1e-6)
    assert np.isclose(rz, 0.0, atol=1e-6)

def test_bloch_roundtrip():
    theta, phi = np.pi / 3, np.pi / 4
    sv = bloch_to_state(theta, phi)
    rx, ry, rz = state_to_bloch(sv, 0, 1)
    t2, p2 = bloch_to_angles(rx, ry, rz)
    assert np.isclose(t2, theta, atol=1e-6)
    assert np.isclose(p2, phi, atol=1e-6)


# ── basis conversion ──────────────────────────────────────────────────────────

def test_basis_labels_1qubit():
    assert basis_ket_labels("comp", 1) == ["|0⟩", "|1⟩"]
    assert basis_ket_labels("hadamard", 1) == ["|+⟩", "|−⟩"]

def test_basis_labels_2qubit():
    labels = basis_ket_labels("comp", 2)
    assert labels == ["|00⟩", "|01⟩", "|10⟩", "|11⟩"]

def test_basis_roundtrip():
    sv = sp.Matrix([sp.Rational(3, 5), sp.Rational(4, 5)])
    for basis in ["comp", "hadamard", "circular"]:
        coeffs = state_to_display_coeffs(sv, basis, 1)
        sv2 = display_coeffs_to_state(coeffs, basis, 1)
        assert np.allclose(as_np(sv), as_np(sv2)), f"Roundtrip failed for basis={basis}"

def test_circular_basis_is_exact():
    """Circular change-of-basis must stay symbolic (sp.I, not float 1j)."""
    sv = sp.Matrix([1, 0])
    coeffs = state_to_display_coeffs(sv, "circular", 1)
    for c in coeffs:
        assert not c.atoms(sp.Float), f"Float leaked into exact coefficients: {c}"

def test_plus_in_hadamard_basis():
    """|+⟩ = (|0⟩+|1⟩)/√2 should have coefficient 1 on |+⟩, 0 on |−⟩."""
    sv = sp.Matrix([1, 1]) / sp.sqrt(2)
    coeffs = state_to_display_coeffs(sv, "hadamard", 1)
    assert np.isclose(abs(complex(coeffs[0])), 1.0, atol=1e-6)
    assert np.isclose(abs(complex(coeffs[1])), 0.0, atol=1e-6)


# ── history, steps, initial state ─────────────────────────────────────────────

def test_intermediate_states_length():
    sys = QuantumSystem.ground(1)
    sys = apply_gate(sys, "H", (0,))
    sys = apply_gate(sys, "Z", (0,))
    states = sys.intermediate_states()
    assert len(states) == 3   # initial + after H + after Z

def test_initial_state_preserved_through_gates():
    sys = QuantumSystem.ground(1)
    sys = apply_gate(sys, "H", (0,))
    sys = apply_gate(sys, "T", (0,))
    assert np.allclose(as_np(sys.initial_state), [1, 0])
    # First intermediate state is the initial state
    assert np.allclose(as_np(sys.intermediate_states()[0]), [1, 0])

def test_composed_matrix_order():
    """History [G1, G2] must compose to G2·G1 (G1 acts first)."""
    sys = QuantumSystem.ground(1)
    sys = apply_gate(sys, "H", (0,))
    sys = apply_gate(sys, "S", (0,))
    expected = _to_numpy(SINGLE_QUBIT["S"]) @ _to_numpy(SINGLE_QUBIT["H"])
    assert np.allclose(_to_numpy(sys.composed_matrix()), expected)

def test_composed_matrix_is_unitary():
    sys = QuantumSystem.ground(2)
    sys = apply_gate(sys, "H", (0,))
    sys = apply_gate(sys, "CNOT", (0, 1))
    assert is_unitary(sys.composed_matrix())
