{
  "workflow_id": "mathematical_formulation_arnoldi_variant",
  "skill_name": "mathematical_formulation",
  "target_file": "mathematical_formulation.md",
  "description": "Template and workflow for mathematically formulating a novel variant of the Arnoldi factorization algorithm targeting specific literature gaps (e.g., adaptive selective reorthogonalization or numerical breakdown recovery).",
  "workflow_steps": [
    {
      "step": 1,
      "name": "Define Mathematical Context & Gap Alignment",
      "description": "Formulate the exact theoretical gap targeted (e.g., breakdown recovery, adaptive reorthogonalization criterion) and establish problem parameters."
    },
    {
      "step": 2,
      "name": "Establish Notation and Primitive Relations",
      "description": "Define matrix/vector spaces, inner products, Krylov subspaces, and exact structural relations."
    },
    {
      "step": 3,
      "name": "Formulate Algorithmic Steps",
      "description": "Provide a precise, step-by-step mathematical algorithm specification including dynamic threshold checks or look-ahead logic."
    },
    {
      "step": 4,
      "name": "Derive Invariants and Error Bounds",
      "description": "Derive perturbed matrix equalities under floating-point arithmetic and prove bounds on loss of orthogonality and residual norms."
    },
    {
      "step": 5,
      "name": "State Theorems and Proof Sketches",
      "description": "Construct formal mathematical theorems for stability, convergence, and breakdown recovery guarantees."
    }
  ],
  "template_content": "# Skill: Mathematical Formulation of Arnoldi Variant\n\n## 1. Literature Gap & Mathematical Objective\n- **Target Literature Gap**: [e.g., Adaptive Selective Reorthogonalization / Look-Ahead Numerical Breakdown Recovery]\n- **Mathematical Objective**: Rigorously derive a modified Arnoldi iteration that maintains matrix invariants and loss-of-orthogonality bounds while addressing [Specific Literature Gap].\n- **Primary Challenge**: Balancing computational cost $O(m^2 n)$ with numerical stability under finite-precision arithmetic $\\epsilon_{\\text{mach}}$.\n\n---\n\n## 2. Notation and Basic Definitions\n- **Matrix/Vector Space**: $A \\in \\mathbb{C}^{n \\times n}$, $v_1 \\in \\mathbb{C}^n$ with $\|v_1\|_2 = 1$.\n- **Krylov Subspace**: $\\mathcal{K}_m(A, v_1) = \\operatorname{span}\\{v_1, A v_1, A^2 v_1, \\dots, A^{m-1} v_1\\}$.\n- **Orthonormal Basis Matrix**: $V_m = [v_1, v_2, \\dots, v_m] \\in \\mathbb{C}^{n \\times m}$.\n- **Hessenberg Matrix**: $H_m = V_m^H A V_m \\in \\mathbb{C}^{m \\times m}$, upper Hessenberg.\n- **Error/Perturbation Matrix**: $E_m = V_m^H V_m - I_m \\in \\mathbb{C}^{m \\times m}$ measuring deviation from orthogonality.\n\n---\n\n## 3. Algorithmic Formulation\n\n### Algorithm: Modified Arnoldi Factorization with Adaptive Criterion\n\n**Input**: Matrix $A \\in \\mathbb{C}^{n \\times n}$, initial vector $v_1 \\in \\mathbb{C}^n$, maximum steps $m$, tolerance parameters $\\eta, \\tau > 0$.\n**Output**: Orthonormal basis $V_{m+1}$, upper Hessenberg matrix $\\tilde{H}_m \\in \\mathbb{C}^{(m+1) \\times m}$.\n\n1. Set $v_1 = v_1 / \|v_1\|_2$.\n2. For $j = 1, 2, \\dots, m$:\n   1. Compute $w_j = A v_j$.\n   2. For $i = 1, \\dots, j$:\n      - $h_{i,j} = \\langle w_j, v_i \\rangle = v_i^H w_j$\n   3. Compute primary intermediate vector:\n      $$w_j' = w_j - \\sum_{i=1}^j h_{i,j} v_i$$\n   4. **[Adaptive Dynamic Trigger Condition]**:\n      - Evaluate stability metric $\\theta_j = \\|w_j'\\|_2 / \\|w_j\\|_2$.\n      - If $\\theta_j < \\eta$ (loss of orthogonality risk or potential breakdown):\n        - **Execute Reorthogonalization / Recovery Step**:\n          - Compute secondary coefficients $\\delta h_{i,j} = v_i^H w_j'$ for $i = 1, \\dots, j$.\n          - Update $w_j'' = w_j' - \\sum_{i=1}^j \\delta h_{i,j} v_i$.\n          - Update $h_{i,j} \\leftarrow h_{i,j} + \\delta h_{i,j}$.\n          - Set $w_j' = w_j''$.\n   5. Compute subdiagonal element: $h_{j+1,j} = \|w_j'\|_2$.\n   6. **[Breakdown Verification]**:\n      - If $h_{j+1,j} \\le \\epsilon_{\\text{mach}} \|A\|_2$:\n        - Trigger look-ahead / invariant subspace detection strategy.\n   7. Normalize: $v_{j+1} = w_j' / h_{j+1,j}$.\n\n---\n\n## 4. Matrix Invariants & Numerical Perturbation Analysis\n\n### Exact Relation Invariant\nIn exact arithmetic, the algorithm satisfies the fundamental Arnoldi relation:\n$$A V_m = V_m H_m + h_{m+1,m} v_{m+1} e_m^T = V_{m+1} \\tilde{H}_m$$\nwhere $e_m$ is the $m$-th canonical basis vector in $\\mathbb{R}^m$.\n\n### Finite-Precision Perturbed Relation\nIn floating-point arithmetic with machine precision $\\epsilon_{\\text{mach}}$:\n$$A V_m + F_m = V_m H_m + h_{m+1,m} v_{m+1} e_m^T$$\nwhere the residual error matrix $F_m = [f_1, f_2, \\dots, f_m]$ satisfies:\n$$\\|f_j\\|_2 \\le c_1 \\epsilon_{\\text{mach}} \\|A\\|_2$$\nfor some constant $c_1 = O(n)$.\n\n### Loss of Orthogonality Recurrence\nLet $E_m = V_m^H V_m - I_m$. The loss of orthogonality evolves according to:\n$$\\|E_{j+1}\\|_F \\le \\|E_j\\|_F + c_2 \\epsilon_{\\text{mach}} \\frac{\\|A\\|_2}{h_{j+1,j}} + O(\\epsilon_{\\text{mach}}^2)$$\nUnder the proposed adaptive reorthogonalization trigger, $\|E_m\|_2 \\le \\tau$ is guaranteed for all $m$.\n\n---\n\n## 5. Theoretical Guarantees & Theorems\n\n### Theorem 1 (Bound on Orthogonality Deviation)\nLet $A \\in \\mathbb{C}^{n \\times n}$ and let $V_m$ be generated by the modified Arnoldi algorithm with trigger parameter $\\eta = 1/\\sqrt{2}$. Then the basis matrix $V_m$ satisfies:\n$$\\|V_m^H V_m - I_m\\|_2 \\le c_3 m \\epsilon_{\\text{mach}}$$\n*Proof Sketch*: By induction on step $j$. Using the secondary inner product projection when $\\theta_j < \\eta$, the component of $w_j'$ in $\\operatorname{range}(V_j)$ is reduced to order $O(\\epsilon_{\\text{mach}} \\|w_j\\|_2)$, bounding the growth of non-diagonal terms in $V_m^H V_m$.\n\n### Theorem 2 (Ritz Pair Convergence under Perturbation)\nLet $(\\theta_k^{(m)}, y_k^{(m)})$ be a Ritz pair of $H_m$, and $x_k^{(m)} = V_m y_k^{(m)}$. The backward error of the approximate eigenpair satisfies:\n$$\\|A x_k^{(m)} - \\theta_k^{(m)} x_k^{(m)}\\|_2 \\le h_{m+1,m} |e_m^T y_k^{(m)}| + \\|F_m\\|_2 \\|y_k^{(m)}\\|_2 + \\|A\\|_2 \\|E_m\\|_2$$\n"
}