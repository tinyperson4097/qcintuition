"""Circuit diagram panel — gate history drawn as a qiskit circuit."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from qiskit import QuantumCircuit
from qiskit.circuit import Gate

from quantum_core import QuantumSystem

# app gate name → QuantumCircuit method (same qubit numbering: q0 drawn on top)
_SIMPLE = {"I": "id", "X": "x", "Y": "y", "Z": "z", "H": "h",
           "S": "s", "S†": "sdg", "T": "t", "T†": "tdg",
           "CNOT": "cx", "CZ": "cz", "SWAP": "swap", "CCNOT": "ccx"}
_ANGLED = {"P": "p", "Rx": "rx", "Ry": "ry", "Rz": "rz"}


def circuit_panel():
    sys: QuantumSystem = st.session_state.system
    if not sys.gate_history:
        return
    st.markdown("**Circuit**")
    fig = _circuit_fig(sys)
    st.pyplot(fig, width="content")
    plt.close(fig)


def _circuit_fig(sys: QuantumSystem):
    n  = sys.num_qubits
    has_meas = any(op.name in ("M0", "M1") for op in sys.gate_history)
    qc = QuantumCircuit(n, n) if has_meas else QuantumCircuit(n)
    for op in sys.gate_history:
        if op.name in _SIMPLE:
            getattr(qc, _SIMPLE[op.name])(*op.qubit_indices)
        elif op.name in _ANGLED:
            getattr(qc, _ANGLED[op.name])(float(op.phi), *op.qubit_indices)
        elif op.name in ("M0", "M1"):
            q = op.qubit_indices[0]
            qc.measure(q, q)
        else:
            # custom unitary — opaque labeled box on its target qubits
            qc.append(Gate(name=op.name, num_qubits=len(op.qubit_indices),
                           params=[]), list(op.qubit_indices))
    fig = qc.draw("mpl", style="iqp", scale=0.7)
    fig.patch.set_alpha(0)
    return fig
