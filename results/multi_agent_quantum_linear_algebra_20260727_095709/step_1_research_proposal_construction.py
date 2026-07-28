import os

# Define the rigorous mathematical formulation for SF-QKS
content = r"""# Spectral-Filtered Quantum Krylov Subspace (SF-QKS) Algorithm

## 1. Mathematical Objective & Gap Alignment
- **Literature Gap**: Classical instability of Krylov subspace generalized eigenvalue problems ($S c = \lambda H c$) in NISQ environments due to exponential ill-conditioning of the overlap matrix $S_{ij} = \langle v_i | v_j \rangle$.
- **Objective**: Implement a spectral filter $\mathcal{F}_\sigma$ to the Krylov subspace basis that enforces a lower bound on the singular values of $S$, effectively regularizing the inversion process without erasing low-energy quantum correlations.

## 2. Notation and Basic Definitions
- **Hilbert Space**: $\mathcal{H} = \mathbb{C}^N$.
- **Krylov Basis**: $\mathcal{V}_m = \{|v_0\rangle, A|v_0\rangle, \dots, A^{m-1}|v_0\rangle\}$.
- **Overlap Matrix**: $S \in \mathbb{C}^{m \times m}$, $S_{ij} = \langle v_i | v_j \rangle$.
- **Hamiltonian Matrix**: $H \in \mathbb{C}^{m \times m}$, $H_{ij} = \langle v_i | A | v_j \rangle$.
- **Spectral Filter**: A diagonal operator $\mathcal{F}_\sigma = \text{diag}(f(\lambda_1), \dots, f(\lambda_m))$ where $f(\lambda) = \frac{\lambda}{\lambda + \sigma^2}$ for regularization parameter $\sigma > 0$.

## 3. Algorithmic Formulation
1. **Basis Generation**: Generate vectors $\{|v_i\rangle\}_{i=0}^{m-1}$ via Hadamard tests.
2. **Matrix Estimation**: Compute $\hat{S}_{ij} = S_{ij} + E_{ij}^S$ and $\hat{H}_{ij} = H_{ij} + E_{ij}^H$, where $E$ represents shot-noise errors.
3. **Spectral Regularization**:
   - Perform eigendecomposition $S = U \Sigma U^\dagger$.
   - Apply filter: $S_\sigma = U \mathcal{F}_\sigma \Sigma U^\dagger$.
   - Solve generalized system: $S_\sigma c = \lambda \hat{H} c$.
4. **Iterative Projection**: Update subspace using the projected vector $|v_m\rangle = \frac{1}{\sqrt{\lambda}} (A - \theta I)|v_{m-1}\rangle$ (Shift-invert transformation).

## 4. Matrix Invariants & Perturbation Analysis
### Finite-Precision Perturbation Bound
Let $\delta = \max(\|E^S\|_2, \|E^H\|_2)$. The perturbed solution $\hat{c}$ satisfies:
$$\|\hat{c} - c\|_2 \le \frac{1}{\sigma_{\min}(S_\sigma)} (\|E^H\|_2 \|c\|_2 + \|E^S\|_2 \|\lambda c\|_2) + \mathcal{O}(\delta^2)$$
where $\sigma_{\min}(S_\sigma) \ge \sigma^2$.

### Convergence Theorem
**Theorem 1**: Under the condition $\sigma > \epsilon_{\text{noise}}$, the Ritz values $\{\theta_k\}$ converge to the spectrum of $A$ at a rate:
$$|\theta_k - \lambda_k| \le \mathcal{O}\left( \frac{\delta}{\sigma^2} + \rho^{m-1} \right)$$
where $\rho < 1$ is the convergence ratio of the Lanczos process in the absence of noise.

## 5. Proof Sketch
1. **Decomposition**: Project $A$ into the filtered subspace. 
2. **Error Control**: The term $1/\sigma^2$ acts as a penalty on the inversion, bounding the amplification of shot noise.
3. **Optimality**: By selecting $\sigma \approx \sqrt{\delta}$, we balance the bias introduced by the filter with the variance induced by shot noise, achieving the minimax error bound.
"""

# Write to file
try:
    with open("mathematical_formulation.md", "w") as f:
        f.write(content)
    print("mathematical_formulation.md successfully written.")
except Exception as e:
    print(f"Error writing file: {e}")