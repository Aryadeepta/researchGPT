import os

markdown_content = """# Skill: Mathematical Formulation of Arnoldi Variant

## 1. Literature Gap & Mathematical Objective
- **Target Literature Gap**: Unresolved structural incompatibility between look-ahead near-breakdown recovery ($h_{j+1,j} \approx 0$ without invariant subspace convergence) and the strict upper Hessenberg requirements of Sorensen's Implicitly Restarted Arnoldi Method (IRAM) (Literature Gap 2).
- **Mathematical Objective**: Rigorously derive a structure-preserving Look-Ahead Arnoldi (LA-IRAM) factorization that dynamically resolves near-breakdown via local condensed block projections while guaranteeing:
  1. Strict bounded loss of orthogonality $\|V_m^H V_m - I_m\|_2 \le c m \epsilon_{\text{mach}}$.
  2. Exact compatibility with polynomial shift transformations $(A - \mu I) V_m = V_m H_m^+$, preserving low-rank residual updates.
- **Primary Challenge**: Standard look-ahead Krylov methods create arbitrary multi-column subdiagonal blocks $H_{j+l, j}$, which destroy the single-step Hessenberg bulge-chasing updates essential for implicit QR shift application.

---

## 2. Notation and Primitive Relations
- **Matrix / Vector Spaces**: $A \in \mathbb{C}^{n \times n}$, initial normalized vector $v_1 \in \mathbb{C}^n$ with $\|v_1\|_2 = 1$.
- **Krylov Subspace**: $\mathcal{K}_m(A, v_1) = \operatorname{span}\{v_1, A v_1, A^2 v_1, \dots, A^{m-1} v_1\}$.
- **Orthonormal Basis Matrix**: $V_m = [v_1, v_2, \dots, v_m] \in \mathbb{C}^{n \times m}$.
- **Look-Ahead Block Index Set**: Let $\mathcal{B} = \{j_1, j_2, \dots, j_p\} \subset \{1, \dots, m\}$ denote indices where near-breakdown occurs ($\|w_{j_k}'\|_2 \le \tau_{\text{break}} \|A\|_2$).
- **Look-Ahead Subspace Block**: For $j \in \mathcal{B}$, define the $l$-step candidate matrix $W_{j, l} = [w_j', A w_j', \dots, A^{l-1} w_j'] \in \mathbb{C}^{n \times l}$.
- **Condensed Upper Hessenberg Matrix**: $H_m = V_m^H A V_m \in \mathbb{C}^{m \times m}$, which is upper Hessenberg except for $l_k \times l_k$ lower triangular subdiagonal block insertions at breakdown locations $j_k$.
- **Orthogonality Deviation Matrix**: $E_m = V_m^H V_m - I_m \in \mathbb{C}^{m \times m}$.

---

## 3. Algorithmic Formulation

### Algorithm: Structure-Preserving Look-Ahead Arnoldi Factorization (LA-IRAM)

**Input**: Matrix $A \in \mathbb{C}^{n \times n}$, initial vector $v_1 \in \mathbb{C}^n$, target dimension $m$, max look-ahead step $L_{\max}$, tolerances $\eta, \tau_{\text{break}}, \epsilon_{\text{mach}}$.  
**Output**: Basis $V_{m+1} \in \mathbb{C}^{n \times (m+1)}$, Block Hessenberg matrix $\tilde{H}_m \in \mathbb{C}^{(m+1) \times m}$, look-ahead index map $\mathcal{B}$.

1. Initialize $v_1 = v_1 / \|v_1\|_2$, $j = 1$, $\mathcal{B} = \emptyset$.
2. While $j \le m$:
   1. Compute standard candidate $w_j = A v_j$.
   2. Compute projection coefficients: $h_{i,j} = \langle v_i, w_j \rangle = v_i^H w_j$ for $i = 1, \dots, j$.
   3. Compute primary orthogonalized vector:
      $$w_j' = w_j - \sum_{i=1}^j h_{i,j} v_i$$
   4. **[Adaptive Reorthogonalization Trigger]**:
      - Evaluate stability ratio $\theta_j = \frac{\|w_j'\|_2}{\|w_j\|_2}$.
      - If $\theta_j < \eta$:
        - Compute corrections $\delta h_{i,j} = v_i^H w_j'$ for $i = 1, \dots, j$.
        - Update $w_j'' = w_j' - \sum_{i=1}^j \delta h_{i,j} v_i$.
        - Update $h_{i,j} \leftarrow h_{i,j} + \delta h_{i,j}$.
        - Set $w_j' = w_j''$.
   5. **[Near-Breakdown Verification & Look-Ahead Trigger]**:
      - If $\|w_j'\|_2 \le \tau_{\text{break}} \|A\|_2$:
        - **[Happy Breakdown Check]**: If $\|w_j'\|_2 \le \epsilon_{\text{mach}} \|A\|_2$, stop (exact invariant subspace found).
        - **[Look-Ahead Window Search]**:
          - Find minimal window length $l \in \{2, \dots, L_{\max}\}$ such that the rank-indicator matrix 
            $$S_l(j) = \begin{bmatrix} v_j^H w_j' & v_j^H A w_j' & \dots & v_j^H A^{l-1} w_j' \\ 0 & v_{j+1}^H A w_j' & \dots & v_{j+1}^H A^{l-1} w_j' \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \dots & v_{j+l-1}^H A^{l-1} w_j' \end{bmatrix}$$
            satisfies $\sigma_{\min}(S_l(j)) \ge \gamma > 0$.
          - Form local Krylov block $W_{j,l} = [w_j', A w_j', \dots, A^{l-1} w_j']$.
          - Compute thin QR factorization of projected complement $(I - V_j V_j^H) W_{j,l} = Q_{j,l} R_{j,l}$.
          - Append $Q_{j,l}$ columns to $V$: $[v_{j+1}, \dots, v_{j+l}] = Q_{j,l}$.
          - Construct block Hessenberg submatrix $H_{j:j+l, j:j+l-1} = V_{j+l}^H A V_{j:j+l-1}$.
          - Record breakdown index $j \in \mathcal{B}$ with block size $l$.
          - Set $j \leftarrow j + l$.
      - Else:
        - Set $h_{j+1,j} = \|w_j'\|_2$.
        - Set $v_{j+1} = w_j' / h_{j+1,j}$.
        - Set $j \leftarrow j + 1$.

---

## 4. Matrix Invariants & Numerical Perturbation Analysis

### Exact Structural Relation Invariant
In exact arithmetic, the Look-Ahead Arnoldi factorization satisfies:
$$A V_m = V_m H_m + h_{m+1,m} v_{m+1} e_m^T + \sum_{k \in \mathcal{B}} U_k \Delta_k E_k^T$$
where $U_k \in \mathbb{C}^{n \times l_k}$ is an orthogonal local correction matrix, $\Delta_k \in \mathbb{C}^{l_k \times l_k}$ is a upper triangular coupling matrix, and $E_k^T$ maps index locations $j_k, \dots, j_k + l_k - 1$.

### Finite-Precision Perturbed Relation
Under floating-point arithmetic with machine precision $\epsilon_{\text{mach}}$, the perturbed relation is:
$$A V_m + F_m = V_m H_m + h_{m+1,m} v_{m+1} e_m^T$$
where $F_m = [f_1, f_2, \dots, f_m]$ represents the backward error accumulation vector sequence.

### Backward Error Vector Norm Bound
For any step $j$ (standard or look-ahead), the error column $f_j$ obeys:
$$\|f_j\|_2 \le c_1 n \epsilon_{\text{mach}} \|A\|_2 + O(\epsilon_{\text{mach}}^2)$$
where $c_1$ is a small integer constant depending only on the local Gram-Schmidt projection order.

### Recurrence Relation for Loss of Orthogonality
Let $E_m = V_m^H V_m - I_m$. For a standard step $j \notin \mathcal{B}$, the departure from orthogonality evolves as:
$$\|E_{j+1}\|_F \le \|E_j\|_F + c_2 \epsilon_{\text{mach}} \left( 1 + \frac{\|A\|_2}{h_{j+1,j}} \cdot \mathbf{1}_{\{\theta_j \ge \eta\}} \right) + O(\epsilon_{\text{mach}}^2)$$
When $\theta_j < \eta$, the adaptive reorthogonalization resets the projection error, yielding:
$$\|E_{j+1}\|_F \le \|E_j\|_F + c_3 \epsilon_{\text{mach}} + O(\epsilon_{\text{mach}}^2)$$
For a look-ahead step $j \in \mathcal{B}$ of window size $l$, pivoting on $S_l(j)$ ensures:
$$\|E_{j+l}\|_F \le \|E_j\|_F + c_4 l \cdot \frac{\epsilon_{\text{mach}}}{\sigma_{\min}(S_l(j))} \|A\|_2 \le \|E_j\|_F + \frac{c_4 l}{\gamma} \epsilon_{\text{mach}} \|A\|_2$$

---

## 5. Theoretical Guarantees & Theorems

### Theorem 1 (Global Orthogonality Guarantee Under Near-Breakdown Recovery)
Let $A \in \mathbb{C}^{n \times n}$ and $V_m \in \mathbb{C}^{n \times m}$ be generated by the LA-IRAM algorithm with reorthogonalization threshold $\eta = 1/\sqrt{2}$ and look-ahead threshold parameter $\gamma > 0$. Then the loss of orthogonality of $V_m$ is bounded independently of any near-zero subdiagonal element $h_{j+1,j}$:
$$\|V_m^H V_m - I_m\|_2 \le \left( c_3 m + \frac{c_4 m}{\gamma} \|A\|_2 \right) \epsilon_{\text{mach}} + O(\epsilon_{\text{mach}}^2)$$

*Proof Sketch*:
1. By induction on iteration index $j$.
2. For $j \notin \mathcal{B}$, if $\|w_j'\|_2 \ge \eta \|w_j\|_2$, standard MGS rounding analysis yields $\|V_{j+1}^H v_{j+1}\|_2 \le c_3 \epsilon_{\text{mach}}$. If $\|w_j'\|_2 < \eta \|w_j\|_2$, the secondary Gram-Schmidt pass cancels the $O(\epsilon_{\text{mach}} / h_{j+1,j})$ amplification, restoring norm error to $O(\epsilon_{\text{mach}})$.
3. For $j \in \mathcal{B}$, single-step division by $h_{j+1,j} \le \tau_{\text{break}} \|A\|_2$ is bypassed. The block QR decomposition of $(I - V_j V_j^H) W_{j,l}$ uses matrix pivoting bounded below by $\sigma_{\min}(S_l(j)) \ge \gamma$.
4. Summing local errors over $m$ steps proves linear growth in $m$ scaled by $\gamma^{-1}$, preventing exponential or unbounded growth. $\blacksquare$

### Theorem 2 (Structure-Preserving Implicit Shift Compatibility)
Let $(A V_m - V_m H_m) = h_{m+1,m} v_{m+1} e_m^T - F_m$ be a valid LA-IRAM factorization, and let $\mu \in \mathbb{C}$ be an implicit shift. Let $H_m - \mu I = Q R$ be the QR factorization of the shifted block Hessenberg matrix. Updating $V_m^+ = V_m Q$ and $H_m^+ = Q^H H_m Q$ yields a transformed relation:
$$A V_m^+ + F_m^+ = V_m^+ H_m^+ + v_{m+1} \tilde{f}_m^T$$
where the updated backward error satisfies $\|F_m^+\|_2 \le \|F_m\|_2 + c_5 m \epsilon_{\text{mach}} \|A\|_2$, and the block Hessenberg structure is preserved outside of local index blocks $\mathcal{B}$.

*Proof Sketch*:
1. Post-multiplying the finite-precision relation by $Q$ gives $A V_m Q + F_m Q = V_m H_m Q + h_{m+1,m} v_{m+1} e_m^T Q$.
2. Substituting $H_m Q = Q H_m^+$ yields $A V_m^+ + F_m Q = V_m^+ H_m^+ + h_{m+1,m} v_{m+1} e_m^T Q$.
3. Since $Q$ is unitary ($\|Q\|_2 = 1$), $\|F_m Q\|_2 = \|F_m\|_2$.
4. Local $l_k \times l_k$ subdiagonal look-ahead blocks in $H_m$ commute with Givens rotations/Householder reflectors in $Q$ outside their immediate bandwidth, restricting structural bulge propagation to $l_k$ steps and preserving the restart capability of IRAM. $\blacksquare$
"""

target_file = "mathematical_formulation.md"
with open(target_file, "w", encoding="utf-8") as f:
    f.write(markdown_content)

print(f"File '{target_file}' generated successfully.")