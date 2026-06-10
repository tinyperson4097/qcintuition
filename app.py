import streamlit as st
from quantum_core import QuantumSystem
from panels.braket import braket_panel
from panels.linalg import linalg_panel
from panels.bloch import bloch_panel
from panels.gates_ui import apply_gates_section
from format_utils import show_latex, bump_sys

st.set_page_config(page_title="Quantum Intuition", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
  /* hide number-input spinner arrows */
  div[data-testid="stNumberInput"] button { display: none !important; }
  input[type=number] { -moz-appearance: textfield; }
  input[type=number]::-webkit-outer-spin-button,
  input[type=number]::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }

  div[data-testid="column"] { padding: 0 0.25rem; }
  .stButton button { width: 100%; }
  /* compact text inputs */
  div[data-testid="stTextInput"] input {
    padding: 0.2rem 0.4rem !important;
    font-size: 0.82rem !important;
    height: 1.9rem !important;
    width: 10ch !important;
    min-width: 0 !important;
  }
  div[data-testid="stTextInput"] { max-width: 6ch !important; }
  div[data-testid="stTextInput"] label { font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)


# ── session init ──────────────────────────────────────────────────────────────

def _init():
    if "system"        not in st.session_state:
        st.session_state.system        = QuantumSystem.ground(1)
    if "basis"         not in st.session_state:
        st.session_state.basis         = "comp"
    if "exact_mode"    not in st.session_state:
        st.session_state.exact_mode    = True
    if "_sys_ver"      not in st.session_state:
        st.session_state["_sys_ver"]   = 0
    if "_basis_sync"   not in st.session_state:
        st.session_state["_basis_sync"] = False

_init()

# Sync all three basis selectors at the top of every run where basis changed.
# Must happen BEFORE any widget renders (hence before any panel code runs).
if st.session_state["_basis_sync"]:
    for _k in ("basis_bk", "basis_la", "basis_bs"):
        st.session_state[_k] = st.session_state.basis
    st.session_state["_basis_sync"] = False

# ── title + cheat sheet ───────────────────────────────────────────────────────

st.title("Quantum Intuition Dashboard")

with st.expander("📖 Cheat sheet", expanded=False):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**State & norm**")
        st.latex(r"|\psi\rangle = \alpha|0\rangle + \beta|1\rangle")
        st.latex(r"|\alpha|^2 + |\beta|^2 = 1")
        show_latex(r"e^{i\gamma}|\psi\rangle \equiv |\psi\rangle \quad \text{(global phase)}")
        st.markdown("**Computational basis**")
        st.latex(r"|0\rangle = \begin{pmatrix}1\\0\end{pmatrix}, \quad |1\rangle = \begin{pmatrix}0\\1\end{pmatrix}")

    with c2:
        st.markdown("**Bloch sphere**")
        st.latex(r"\alpha = \cos\tfrac{\theta}{2}, \quad \beta = e^{i\varphi}\sin\tfrac{\theta}{2}")
        st.latex(r"(x,y,z) = (\sin\theta\cos\varphi,\;\sin\theta\sin\varphi,\;\cos\theta)")
        show_latex(r"|0\rangle \text{ north pole},\quad |1\rangle \text{ south pole}")
        show_latex(r"|{+}\rangle \text{ equator } \varphi=0,\quad |{-}\rangle \text{ equator } \varphi=\pi")

    with c3:
        st.markdown("**Hadamard & circular bases**")
        st.latex(r"|{+}\rangle = \tfrac{|0\rangle+|1\rangle}{\sqrt{2}}, \quad |{-}\rangle = \tfrac{|0\rangle-|1\rangle}{\sqrt{2}}")
        st.latex(r"|i\rangle = \tfrac{|0\rangle+i|1\rangle}{\sqrt{2}}, \quad |{-i}\rangle = \tfrac{|0\rangle-i|1\rangle}{\sqrt{2}}")
        st.markdown("**Gates & multi-qubit**")
        st.latex(r"U^\dagger U = I \quad \text{(unitary)}")
        st.latex(r"\text{Bell: }\tfrac{|00\rangle+|11\rangle}{\sqrt{2}}, \quad |\psi_1\rangle\otimes|\psi_2\rangle")

# ── top controls ──────────────────────────────────────────────────────────────

ctl_n, ctl_exact, ctl_reset = st.columns([1, 1.2, 0.8])
with ctl_n:
    n = st.selectbox("Number of qubits", [1, 2, 3, 4],
                     index=st.session_state.system.num_qubits - 1,
                     key="num_qubits_sel")
    if n != st.session_state.system.num_qubits:
        st.session_state.system = QuantumSystem.ground(n)
        bump_sys()
        st.rerun()

with ctl_exact:
    exact = st.toggle("Exact form", value=st.session_state.exact_mode, key="exact_toggle")
    if exact != st.session_state.exact_mode:
        st.session_state.exact_mode = exact
        bump_sys()
        st.rerun()

with ctl_reset:
    st.markdown("<div style='padding-top:1.7rem'></div>", unsafe_allow_html=True)
    if st.button("Reset |0…0⟩", key="global_reset"):
        st.session_state.system = QuantumSystem.ground(n)
        bump_sys()
        st.rerun()

# ── gate row ──────────────────────────────────────────────────────────────────
with st.container(border=True):
    apply_gates_section()

st.divider()

# ── three panels ──────────────────────────────────────────────────────────────
col_bk, col_la, col_bs = st.columns(3)
with col_bk:
    braket_panel()
with col_la:
    linalg_panel()
with col_bs:
    bloch_panel()
