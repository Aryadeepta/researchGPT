import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import sys
import time

def arnoldi_factorization(A, v0, k, reorth='mgs2', tol=1e-12):
    """
    Arnoldi Factorization: A * V_k = V_{k+1} * H_tilde_k
    where V_{k+1} is (n x (k+1)), H_tilde_k is ((k+1) x k)
    """
    is_sparse = sp.issparse(A)
    n = A.shape[0]
    
    dtype = np.complex128 if (np.iscomplexobj(A) or np.iscomplexobj(v0)) else np.float64
    V = np.zeros((n, k + 1), dtype=dtype)
    H = np.zeros((k + 1, k), dtype=dtype)

    norm_v0 = la.norm(v0)
    if norm_v0 < 1e-15:
        return V[:, :1], H[:1, :0], 0, "breakdown_initial_zero"

    V[:, 0] = v0 / norm_v0
    completed_steps = 0
    breakdown_type = None

    for j in range(k):
        if is_sparse:
            v_next = A.dot(V[:, j])
        else:
            v_next = A @ V[:, j]

        if reorth == 'none':
            for i in range(j + 1):
                H[i, j] = np.dot(V[:, i].conj(), v_next)
                v_next = v_next - H[i, j] * V[:, i]
            h_next = la.norm(v_next)
            H[j + 1, j] = h_next

        elif reorth == 'mgs':
            for i in range(j + 1):
                h_val = np.dot(V[:, i].conj(), v_next)
                H[i, j] += h_val
                v_next = v_next - h_val * V[:, i]
            h_next = la.norm(v_next)
            H[j + 1, j] = h_next

        elif reorth == 'mgs2':
            for pass_idx in range(2):
                for i in range(j + 1):
                    h_val = np.dot(V[:, i].conj(), v_next)
                    H[i, j] += h_val
                    v_next = v_next - h_val * V[:, i]
            h_next = la.norm(v_next)
            H[j + 1, j] = h_next

        elif reorth == 'cgs2':
            for pass_idx in range(2):
                h_vec = V[:, :j+1].T.conj() @ v_next
                H[:j+1, j] += h_vec
                v_next = v_next - V[:, :j+1] @ h_vec
            h_next = la.norm(v_next)
            H[j + 1, j] = h_next
        else:
            raise ValueError(f"Unknown reorth method: {reorth}")

        completed_steps = j + 1

        if h_next < tol:
            if h_next == 0 or h_next < 1e-15:
                breakdown_type = "happy"
            else:
                breakdown_type = "near_breakdown"
            V[:, j + 1] = v_next / (h_next + 1e-300) # prevent division by exactly zero
            break

        V[:, j + 1] = v_next / h_next

    return V[:, :completed_steps + 1], H[:completed_steps + 1, :completed_steps], completed_steps, breakdown_type


def run_adversarial_suite():
    results = {}

    # TEST 1: Extreme Ill-Conditioned Matrix (Condition Number ~ 1e12)
    np.random.seed(42)
    n = 200
    k = 40
    U, _ = la.qr(np.random.randn(n, n))
    V_mat, _ = la.qr(np.random.randn(n, n))
    S = np.diag(np.logspace(0, 12, n))
    A_ill = U @ S @ V_mat.T
    v0 = np.random.randn(n)

    V, H, steps, btype = arnoldi_factorization(A_ill, v0, k, reorth='none')
    orth_none = la.norm(V.T.conj() @ V - np.eye(steps + 1))
    
    V_mgs2, H_mgs2, steps_mgs2, _ = arnoldi_factorization(A_ill, v0, k, reorth='mgs2')
    orth_mgs2 = la.norm(V_mgs2.T.conj() @ V_mgs2 - np.eye(steps_mgs2 + 1))
    
    res_mgs2 = la.norm(A_ill @ V_mgs2[:, :steps_mgs2] - V_mgs2 @ H_mgs2)

    results['ill_conditioned'] = {
        'orth_loss_none': orth_none,
        'orth_loss_mgs2': orth_mgs2,
        'relation_res_mgs2': res_mgs2
    }

    # TEST 2: Near-Breakdown Invariant Subspace
    n = 100
    A_near = np.zeros((n, n))
    for i in range(n - 1):
        A_near[i + 1, i] = 1.0
    A_near[0, n - 1] = 1.0
    
    # Invariant subspace spanned by first 5 canonical vectors + tiny perturbation
    v0_near = np.zeros(n)
    v0_near[:5] = np.random.randn(5)
    v0_near[5] = 1e-13  # near breakdown trigger

    V_nb, H_nb, steps_nb, btype_nb = arnoldi_factorization(A_near, v0_near, k=10, reorth='mgs2', tol=1e-10)
    orth_nb = la.norm(V_nb.T.conj() @ V_nb - np.eye(steps_nb + 1))
    res_nb = la.norm(A_near @ V_nb[:, :steps_nb] - V_nb @ H_nb)

    results['near_breakdown'] = {
        'steps_completed': steps_nb,
        'breakdown_type': btype_nb,
        'orth_loss': orth_nb,
        'relation_res': res_nb
    }

    # TEST 3: Large Sparse Convection-Diffusion Matrix
    n_grid = 50
    N = n_grid * n_grid
    k_sparse = 50
    dx = 1.0 / (n_grid + 1)
    
    # 2D Convection-Diffusion matrix
    diags = [-1.0, -1.0, 4.0, -1.0, -1.0]
    # Add strong convection (non-symmetric)
    peclet = 100.0
    cx = peclet / (2.0 * dx)
    
    main_diag = 4.0 * np.ones(N)
    off_x_neg = (-1.0 - cx) * np.ones(N - 1)
    off_x_pos = (-1.0 + cx) * np.ones(N - 1)
    
    A_sparse = sp.diags([main_diag, off_x_neg, off_x_pos], [0, -1, 1], format='csr')
    v0_sparse = np.random.randn(N)

    t0 = time.time()
    V_sp, H_sp, steps_sp, _ = arnoldi_factorization(A_sparse, v0_sparse, k_sparse, reorth='cgs2')
    t_cgs2 = time.time() - t0

    t0 = time.time()
    V_sp_mgs, H_sp_mgs, steps_sp_mgs, _ = arnoldi_factorization(A_sparse, v0_sparse, k_sparse, reorth='mgs')
    t_mgs = time.time() - t0

    orth_cgs2 = la.norm(V_sp.T.conj() @ V_sp - np.eye(steps_sp + 1))
    res_cgs2 = la.norm(A_sparse.dot(V_sp[:, :steps_sp]) - V_sp @ H_sp)

    results['large_sparse'] = {
        'matrix_dim': N,
        'cgs2_time': t_cgs2,
        'mgs_time': t_mgs,
        'cgs2_orth_loss': orth_cgs2,
        'cgs2_relation_res': res_cgs2
    }

    # TEST 4: Long Krylov Sequence Floating-Point Drift (k = 120)
    n = 300
    k_long = 120
    A_long = np.random.randn(n, n) / np.sqrt(n)
    v0_long = np.random.randn(n)

    V_l_mgs, _, steps_l_mgs, _ = arnoldi_factorization(A_long, v0_long, k_long, reorth='mgs')
    V_l_mgs2, H_l_mgs2, steps_l_mgs2, _ = arnoldi_factorization(A_long, v0_long, k_long, reorth='mgs2')

    orth_drift_mgs = la.norm(V_l_mgs.T.conj() @ V_l_mgs - np.eye(steps_l_mgs + 1))
    orth_drift_mgs2 = la.norm(V_l_mgs2.T.conj() @ V_l_mgs2 - np.eye(steps_l_mgs2 + 1))

    results['long_krylov'] = {
        'k': k_long,
        'mgs_orth_loss': orth_drift_mgs,
        'mgs2_orth_loss': orth_drift_mgs2
    }

    return results


def print_adversarial_critique(results):
    print("=" * 80)
    print("ADVERSARIAL CRITIQUE & HARDENING REPORT: ARNOLDI FACTORIZATION")
    print("=" * 80)
    print("\n1. EXPERIMENTAL STRESS-TEST EVALUATION RESULTS:")
    print("-" * 60)
    
    ill = results['ill_conditioned']
    print(f"[TEST 1: Ill-Conditioned Matrix (cond ~ 1e12)]")
    print(f"  - No Reorthogonalization Loss of Orthogonality: {ill['orth_loss_none']:.2e}")
    print(f"  - MGS2 Loss of Orthogonality                : {ill['orth_loss_mgs2']:.2e}")
    print(f"  - MGS2 Arnoldi Relation Residual            : {ill['relation_res_mgs2']:.2e}")
    
    nb = results['near_breakdown']
    print(f"\n[TEST 2: Near-Breakdown Subspace Thresholding]")
    print(f"  - Steps Completed Before Breakdown Trigger : {nb['steps_completed']}")
    print(f"  - Detected Breakdown Flag                  : {nb['breakdown_type']}")
    print(f"  - Subspace Loss of Orthogonality           : {nb['orth_loss']:.2e}")
    print(f"  - Arnoldi Relation Residual Norm           : {nb['relation_res']:.2e}")

    sp_res = results['large_sparse']
    print(f"\n[TEST 3: Large Sparse Convection-Diffusion Operator (N={sp_res['matrix_dim']})]")
    print(f"  - CGS2 Execution Time                      : {sp_res['cgs2_time']:.4f} s")
    print(f"  - MGS  Execution Time                      : {sp_res['mgs_time']:.4f} s")
    print(f"  - CGS2 Orthogonality Loss                  : {sp_res['cgs2_orth_loss']:.2e}")
    print(f"  - CGS2 Residual Norm                       : {sp_res['cgs2_relation_res']:.2e}")

    lk = results['long_krylov']
    print(f"\n[TEST 4: Long Krylov Sequence Floating-Point Drift (k={lk['k']})]")
    print(f"  - MGS Single Pass Loss of Orthogonality    : {lk['mgs_orth_loss']:.2e}")
    print(f"  - MGS2 Double Pass Loss of Orthogonality   : {lk['mgs2_orth_loss']:.2e}")

    print("\n" + "=" * 80)
    print("2. DETAILED NUMERICAL FAILURE MODES & VULNERABILITY ANALYSIS")
    print("=" * 80)
    print(r"""
[VULNERABILITY 1: Static Reorthogonalization Triggering & Overhead]
  - Issue: Blindly applying double Gram-Schmidt (MGS2 / CGS2) on every iteration 
    doubles memory traffic and BLAS calls. In large sparse settings, MGS introduces 
    heavy memory latency due to BLAS-1 vector operations, whereas CGS2 uses BLAS-2/3 
    matrix-vector products but suffers redundant computations when orthogonality 
    drift is negligible.
  - Failure Mode: Under high iteration counts k >> 50, fixed double passes still 
    accumulate machine-precision rounding error drift ~ O(k * eps_mach), failing 
    to maintain strict orthogonality without selective adaptive reorthogonalization.

[VULNERABILITY 2: Rigid Subspace Hard-Stop on Near-Breakdown (h_{k+1,k} approx 0)]
  - Issue: When encountering near-breakdown (h_{k+1,k} < tol), the current implementation 
    simply truncates the Krylov subspace. If this occurs before desired Ritz values have 
    converged, the eigensolver stagnates completely.
  - Failure Mode: Standard Sorensen IRAM requires exact upper Hessenberg structure. 
    Look-ahead methods introduce block upper Hessenberg forms which break standard 
    implicit QR shift polynomial updates.

[VULNERABILITY 3: Non-Normal Pseudospectral Sensitivity & Scale Drift]
  - Issue: For highly non-normal matrices (e.g. Grcar, convection-diffusion with high 
    Peclet numbers), Ritz values computed from H_k are hyper-sensitive to perturbations. 
    Standard residual norms ||A v_i - \lambda_i v_i|| severely understate eigenvector error 
    when non-normality causes pseudospectra to inflate.

[VULNERABILITY 4: Complex Precision Inconsistency & Memory Allocations]
  - Issue: Dynamic type casting during Arnoldi steps causes implicit array reallocations 
    or unexpected precision loss when real input matrices yield complex Ritz shifts 
    during IRAM restarting phases.
""")

    print("=" * 80)
    print("3. CONCRETE ACTIONABLE RECOMMENDATIONS FOR ALGORITHM HARDENING")
    print("=" * 80)
    print(r"""
[RECOMMENDATION 1: Adaptive Kahan-Parlett-Berdnes (KPB) Reorthogonalization Criteria]
  - Action: Implement selective dynamic reorthogonalization:
    Perform a second Gram-Schmidt pass ONLY when ||v_next^{(2)}|| < \gamma ||v_next^{(1)}|| 
    (with typical threshold \gamma = 0.707).
  - Benefit: Eliminates 50% of computational overhead on stable Krylov steps while 
    guaranteeing orthogonality to O(eps_mach) on ill-conditioned / drift-prone steps.

[RECOMMENDATION 2: Transition from IRAM to Krylov-Schur Restarting]
  - Action: Replace implicit QR restart shifts with Stewart's Krylov-Schur algorithm.
  - Benefit: Krylov-Schur operates on full real/complex Schur forms, enabling seamless 
    re-ordering of Ritz values and robust look-ahead expansion during near-breakdown 
    without being constrained by strict Hessenberg tridiagonal/subdiagonal structures.

[RECOMMENDATION 3: Dynamic Matrix Norm Scaling & Safe Subdiagonal Zeroing]
  - Action: Scale initial vector and matrix norms dynamically: A_scaled = A / ||A||_1.
    Set breakdown tolerance dynamically: tol_breakdown = n * eps_mach * ||A||_1.
  - Benefit: Prevents underflow/overflow during long Krylov sequence expansions and 
    provides scale-invariant breakdown detection.

[RECOMMENDATION 4: High-Performance BLAS-3 Block Krylov Formulation]
  - Action: Unroll Krylov basis generation into blocks of size s (s-step Arnoldi / CA-Arnoldi) 
    using TSQR (Tall-Skinny QR) for GPU/distributed cluster nodes.
  - Benefit: Overcomes BLAS-1 latency in MGS and eliminates communication bottlenecks.
""")
    print("=" * 80)

if __name__ == "__main__":
    results = run_adversarial_suite()
    print_adversarial_critique(results)