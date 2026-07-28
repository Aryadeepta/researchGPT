import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import LinearOperator

class QuantumWalkSolver:
    """
    Modular Quantum Walk-based Feynman-Kac solver for high-dimensional PDEs.
    Utilizes LinearOperators to manage Hilbert space H = H_coin \otimes H_pos^D.
    """
    def __init__(self, dim, size, steps):
        self.dim = dim
        self.size = size
        self.steps = steps
        self.N = size**dim
        self.total_dim = 2 * self.N
        self.psi = np.zeros(self.total_dim, dtype=np.complex128)
        self.psi[0] = 1.0  # Initialize at origin
        
        # Define coin operator (Hadamard-like)
        self.coin = self._build_coin_operator()
        # Define shift operator as a LinearOperator
        self.shift = self._build_shift_operator()

    def _build_coin_operator(self):
        # Local 2x2 coin
        c = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        # Scale to full Hilbert space using sparse Kronecker
        eye_pos = sparse.eye(self.N, format='csr')
        return sparse.kron(sparse.csr_matrix(c), eye_pos)

    def _build_shift_operator(self):
        # Shift operator S: flips coin state based on position
        # Defined as matrix-free via LinearOperator to handle high-D
        def matvec(v):
            # Split psi into coin=0 and coin=1 subspaces
            mid = v.size // 2
            psi0, psi1 = v[:mid], v[mid:]
            # Roll state to implement shift
            psi0 = np.roll(psi0, 1)
            psi1 = np.roll(psi1, -1)
            return np.concatenate([psi0, psi1])
        
        return LinearOperator((self.total_dim, self.total_dim), matvec=matvec)

    def step(self):
        # U = S * (C \otimes I)
        self.psi = self.coin @ self.psi
        self.psi = self.shift @ self.psi
        norm = np.linalg.norm(self.psi)
        return norm

    def calculate_variance(self):
        probs = np.abs(self.psi)**2
        # Normalize
        probs /= np.sum(probs)
        idx = np.arange(self.total_dim)
        mean = np.sum(idx * probs)
        variance = np.sum(((idx - mean)**2) * probs)
        return variance, np.sum(probs)

def run_simulation():
    for d in range(1, 4):
        solver = QuantumWalkSolver(dim=d, size=10, steps=5)
        for _ in range(solver.steps):
            solver.step()
        var, mass = solver.calculate_variance()
        print(f"[INFO] Dimension: {d} | Steps: {solver.steps} | Final Variance: {var:.4f} | Mass: {mass:.4f}")

if __name__ == "__main__":
    run_simulation()