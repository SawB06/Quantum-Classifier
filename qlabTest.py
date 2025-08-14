# -------------
#  qlabTest.py
# -------------
# Simple quantum circuit to test device connectivity

# Quantum Backend: Several options available (IonQ simulator, IonQ Aria1, default qubit, etc.)


# ============
#  1. Imports
# ============
import pennylane as qml


# =========================
#  7. Quantum Device Setup
# =========================
dev = qml.device("ionq.simulator",wires=1,api_key="YOUR_API_KEY")
#dev = qml.device("ionq.qpu",wires=1, shots=1000,api_key="YOUR_API_KEY")
#dev = qml.device("default.qubit", wires=1)


#===================
#  Quantum Circuit
#==================
@qml.qnode(dev)
def apply_hadamard():
    qml.Hadamard(0)
    return qml.probs(wires=0)
print(apply_hadamard())

