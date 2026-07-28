"""
Multi-Agent Quantum Walk-Based Feynman-Kac Solver for High-Dimensional PDEs on Spherical Manifolds.
This solver is fully modular, type-safe, and designed to scale seamlessly across dimensions (d=1, 2, 3).
"""

import numpy as np
from scipy import sparse
import time

def safe_kron(A, B):
    """
    Computes Kronecker product of two sparse matrices safely to prevent deprecation
    and future warnings in SciPy.
    """
    return sparse.kron(sparse.coo_matrix(A), sparse.coo_matrix(B)).tocsr()

class SphericalQuantumWalkSolver:
    def __init__(self, dim, size, dt=0.01, potential_fn=None):
        """
        Parameters:
        -----------
        dim : int
            Dimensionality of the hypersphere S^dim (e.g., d=1, 2, 3)
        size : int
            Number of grid points per coordinate dimension
        dt : float
            Time step size
        potential_fn : callable, optional
            Potential V(x) mapping coordinate vectors to scalar potentials
        """
        self.dim = dim
        self.size = size
        self.dt = dt
        self.potential_fn = potential_fn
        self.pos_dim = size ** dim
        self.coin_dim = 2 ** dim
        self.epsilon = 1e-3  # Pole regularization factor
        
        # Build embedding coordinates & local metrics
        self.coords, self.metric_diag = self._build_geometry()
        
        # Build state vector and operators
        self.psi = self._initialize_state()
        self.S = self._build_shift_operator()
        self.C = self._build_spatially_dependent_coin()
        self.W = self._build_feynman_kac_weights()

    def _build_geometry(self):
        """Precomputes coordinate mapping, embedding vectors, and local metrics."""
        coords = np.zeros((self.pos_dim, self.dim + 1))
        metric_diag = np.zeros((self.pos_dim, self.dim))
        
        for p in range(self.pos_dim):
            # Reconstruct multi-index
            multi_idx = []
            temp = p
            for j in range(self.dim):
                multi_idx.append(temp // (self.size**(self.dim - 1 - j)))
                temp = temp % (self.size**(self.dim - 1 - j))
            
            # Compute angles
            angles = []
            for j in range(self.dim - 1):
                angles.append(multi_idx[j] * np.pi / max(self.size - 1, 1))
            angles.append(multi_idx[-1] * 2 * np.pi / self.size)
            
            # Compute embedding coordinates y in R^(d+1)
            y = np.zeros(self.dim + 1)
            sin_prod = 1.0
            for j in range(self.dim - 1):
                y[j] = sin_prod * np.cos(angles[j])
                sin_prod *= np.sin(angles[j])
            y[self.dim - 1] = sin_prod * np.cos(angles[-1])
            y[self.dim] = sin_prod * np.sin(angles[-1])
            coords[p] = y
            
            # Compute diagonal components of metric tensor g^jj
            g_inv = np.zeros(self.dim)
            sin_prod = 1.0
            for j in range(self.dim):
                g_inv[j] = 1.0 / (sin_prod**2 + self.epsilon)
                if j < self.dim - 1:
                    sin_prod *= np.sin(angles[j])
            metric_diag[p] = g_inv
            
        return coords, metric_diag

    def _initialize_state(self):
        """Initializes a normalized symmetric coin state localized at the center of the grid."""
        psi = np.zeros(self.coin_dim * self.pos_dim, dtype=np.complex128)
        
        # Symmetric coin state
        coin_state = np.ones(self.coin_dim, dtype=np.complex128) / np.sqrt(self.coin_dim)
        
        # Localized position state at center
        center_pos = self.pos_dim // 2
        for c in range(self.coin_dim):
            psi[c * self.pos_dim + center_pos] = coin_state[c]
            
        return psi

    def _build_shift_operator(self):
        """Constructs the high-dimensional shift operator via sparse Kronecker tensor products."""
        def get_translation_1d(N, s):
            data = np.ones(N, dtype=np.complex128)
            rows = (np.arange(N) + s) % N
            cols = np.arange(N)
            return sparse.coo_matrix((data, (rows, cols)), shape=(N, N), dtype=np.complex128).tocsr()

        S_blocks = [[None for _ in range(self.coin_dim)] for _ in range(self.coin_dim)]
        
        for c in range(self.coin_dim):
            # Interpret the bits of the coin state as step directions
            directions = []
            for j in range(self.dim):
                bit = (c >> (self.dim - 1 - j)) & 1
                directions.append(1 if bit == 1 else -1)
            
            # Take tensor product of 1D translation operators
            S_c = get_translation_1d(self.size, directions[0])
            for j in range(1, self.dim):
                S_c = safe_kron(S_c, get_translation_1d(self.size, directions[j]))
                
            S_blocks[c][c] = S_c
            
        return sparse.bmat(S_blocks, format='csr')

    def _build_spatially_dependent_coin(self):
        """Builds a spatially-dependent unitary coin operator matching the local geometry."""
        coin_blocks = [[None for _ in range(self.coin_dim)] for _ in range(self.coin_dim)]
        
        # Precompute coin entries for all position states p
        C_entries = np.zeros((self.pos_dim, self.coin_dim, self.coin_dim), dtype=np.complex128)
        
        for p in range(self.pos_dim):
            g_inv = self.metric_diag[p]
            denom = np.sum(g_inv)
            
            # Construct 1D coins for each dimension
            coins_1d = []
            for j in range(self.dim):
                cos_val = np.sqrt(g_inv[j] / denom)
                sin_val = np.sqrt(1.0 - cos_val**2)
                # Form unitary orthogonal 2x2 matrix
                c_1d = np.array([[cos_val, sin_val], [sin_val, -cos_val]], dtype=np.complex128)
                coins_1d.append(c_1d)
                
            # Perform Kronecker product to get total coin operator at position p
            c_p = coins_1d[0]
            for j in range(1, self.dim):
                c_p = np.kron(c_p, coins_1d[j])
                
            C_entries[p] = c_p

        # Place local coin coefficients into the global block matrix structure
        for c in range(self.coin_dim):
            for c_prime in range(self.coin_dim):
                diag_vals = C_entries[:, c, c_prime]
                coin_blocks[c][c_prime] = sparse.diags(diag_vals, dtype=np.complex128, format='csr')
                
        return sparse.bmat(coin_blocks, format='csr')

    def _build_feynman_kac_weights(self):
        """Constructs the diagonal Feynman-Kac weight matrix representing potential V(x)."""
        V_vals = np.zeros(self.pos_dim)
        if self.potential_fn is not None:
            for p in range(self.pos_dim):
                V_vals[p] = self.potential_fn(self.coords[p])
                
        weights = np.exp(-V_vals * self.dt)
        # Tile the weights across all coin states to match joint space dimension
        return sparse.diags(np.tile(weights, self.coin_dim), dtype=np.complex128, format='csr')

    def step(self):
        """Performs one step of the path-integral quantum walk simulation."""
        # Evolution: psi = W * S * C * psi
        self.psi = self.W.dot(self.S.dot(self.C.dot(self.psi)))
        return self.get_diagnostics()

    def get_diagnostics(self):
        """Computes mass, probability distribution, and variance of the distribution on S^d."""
        prob_dist = np.zeros(self.pos_dim)
        for c in range(self.coin_dim):
            prob_dist += np.abs(self.psi[c * self.pos_dim : (c + 1) * self.pos_dim])**2
            
        total_mass = np.sum(prob_dist)
        
        # Calculate directional variance on the sphere
        if total_mass > 1e-15:
            norm_prob = prob_dist / total_mass
            mean_y = np.dot(norm_prob, self.coords)
            variance = 1.0 - np.sum(mean_y**2)
        else:
            variance = 0.0
            
        return total_mass, variance


if __name__ == "__main__":
    print("="*80)
    print("             QUANTUM WALK-BASED FEYNMAN-KAC PDE SOLVER DIAGNOSTICS")
    print("="*80)
    
    # Test potential function V(y) = 1.0 - y_0^2 (harmonic-like potential on hypersphere)
    def test_potential(y):
        return 2.0 * (1.0 - y[0]**2)

    for d in [1, 2, 3]:
        print(f"\n[INIT] Instantiating spherical quantum walk for Dimension d = {d}...")
        start_time = time.time()
        
        # Instantiating solver
        solver = SphericalQuantumWalkSolver(dim=d, size=8, dt=0.01, potential_fn=test_potential)
        init_mass, init_var = solver.get_diagnostics()
        print(f"       Hilbert Space Dimension: {solver.coin_dim * solver.pos_dim}")
        print(f"       Initial Mass (Norm): {init_mass:.6f} | Initial Directional Variance: {init_var:.6f}")
        
        # Run 5 coherent simulation steps
        print("       Evolving solver steps...")
        for step_idx in range(1, 6):
            mass, var = solver.step()
            print(f"       Step {step_idx:02d} -> Mass: {mass:.6f} | Directional Variance: {var:.6f}")
            
        elapsed = time.time() - start_time
        print(f"[SUCCESS] Completed Dimension {d} in {elapsed:.4f} seconds.")
        
    print("\n" + "="*80)
    print("             UNITARITY & STABILITY VERIFICATION TEST")
    print("="*80)
    
    # Unitarity verification (Potential V = 0)
    print("[TEST] Running flat potential (V = 0) to verify strict norm preservation (unitarity)...")
    unitary_solver = SphericalQuantumWalkSolver(dim=2, size=8, dt=0.01, potential_fn=None)
    for step_idx in range(1, 6):
        mass, var = unitary_solver.step()
        print(f"       Step {step_idx:02d} -> Mass: {mass:.12f} (Delta: {abs(mass - 1.0):.2e}) | Variance: {var:.6f}")
        assert abs(mass - 1.0) < 1e-12, "Unitarity violation detected!"
    print("[SUCCESS] Unitary evolution verified with zero probability leakage.")
    print("="*80)