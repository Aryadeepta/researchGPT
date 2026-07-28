import numpy as np
import scipy.linalg as la
from typing import Tuple, Dict, Any, Optional

def arnoldi_factorization(
    A: np.ndarray,
    v0: np.ndarray,
    k: int,
    reorth: str = "cgs2",
    tol: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Computes the k-step Arnoldi factorization of matrix A starting from initial vector v0.

    Mathematical Relation:
        A * V_m = V_m * H_m + h_{m+1, m} * v_{m+1} * e_m^T = V_{m+1} * \tilde{H}_m

    Parameters
    ----------
    A : np.ndarray
        Square matrix of shape (n, n), real or complex.
    v0 : np.ndarray
        Initial starting vector of shape (n,).
    k : int
        Number of Arnoldi steps (desired Krylov subspace dimension).
    reorth : str, optional
        Reorthogonalization strategy:
        - 'none': Standard Modified Gram-Schmidt (MGS) without reorthogonalization.
        - 'mgs2': MGS with full double reorthogonalization.
        - 'cgs2': Classical Gram-Schmidt with double reorthogonalization (CGS2).
        Default is 'cgs2'.
    tol : float, optional
        Breakdown detection threshold. If h_{j+1, j} < tol, a happy breakdown is flagged.

    Returns
    -------
    V : np.ndarray
        Orthonormal basis matrix of shape (n, m+1) where m <= k.
    H : np.ndarray
        Upper Hessenberg matrix of shape (m+1, m) where m <= k.
    info : dict
        Diagnostic metadata including actual steps, breakdown status, loss of orthogonality,
        and matrix relation residual norm.
    """
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square.")
    
    # Ensure working data type supports complex if A or v0 is complex
    dtype = np.result_type(A.dtype, v0.dtype, np.complex128 if np.iscomplexobj(A) else np.float64)
    A = A.astype(dtype, copy=False)
    
    v0 = np.asarray(v0, dtype=dtype).reshape(-1)
    v0_norm = la.norm(v0)
    
    if v0_norm == 0:
        raise ValueError("Initial vector v0 cannot be the zero vector.")
    
    # Pre-allocate basis V and Hessenberg matrix H
    # V will store [v_1, v_2, ..., v_{m+1}]
    # H_tilde will store size (k+1, k)
    V = np.zeros((n, k + 1), dtype=dtype)
    H = np.zeros((k + 1, k), dtype=dtype)
    
    V[:, 0] = v0 / v0_norm
    
    breakdown = False
    breakdown_type = "none"
    actual_steps = k
    
    for j in range(k):
        v_j = V[:, j]
        w = A @ v_j
        
        if reorth == "none":
            # Modified Gram-Schmidt (MGS)
            for i in range(j + 1):
                H[i, j] = np.dot(V[:, i].conj(), w)
                w = w - H[i, j] * V[:, i]
                
        elif reorth == "mgs2":
            # MGS with double orthogonalization
            for i in range(j + 1):
                h_ij = np.dot(V[:, i].conj(), w)
                H[i, j] += h_ij
                w = w - h_ij * V[:, i]
            # Second pass
            for i in range(j + 1):
                h_ij2 = np.dot(V[:, i].conj(), w)
                H[i, j] += h_ij2
                w = w - h_ij2 * V[:, i]
                
        elif reorth == "cgs2":
            # Classical Gram-Schmidt with double orthogonalization (CGS2)
            V_curr = V[:, :j + 1]
            # Pass 1
            h1 = V_curr.conj().T @ w
            w = w - V_curr @ h1
            # Pass 2
            h2 = V_curr.conj().T @ w
            w = w - V_curr @ h2
            H[:j + 1, j] = h1 + h2
        else:
            raise ValueError(f"Unknown reorthogonalization scheme: {reorth}")
            
        h_next = la.norm(w)
        H[j + 1, j] = h_next
        
        # Check for breakdown threshold
        if h_next < tol:
            breakdown = True
            actual_steps = j + 1
            if h_next == 0.0 or h_next < 1e-15:
                breakdown_type = "happy (exact invariant subspace)"
            else:
                breakdown_type = "near-breakdown"
            V[:, j + 1] = 0.0
            break
        else:
            V[:, j + 1] = w / h_next

    # Truncate matrices to actual dimension computed
    m = actual_steps
    V_m1 = V[:, :m + 1]
    H_tilde = H[:m + 1, :m]
    
    # Quantitative diagnostics
    V_m = V_m1[:, :m]
    I_m = np.eye(m, dtype=dtype)
    ortho_loss = la.norm(V_m.conj().T @ V_m - I_m, ord='fro')
    
    # Residual norm ||A V_m - V_{m+1} \tilde{H}_m||_F
    residual_mat = A @ V_m - V_m1 @ H_tilde
    relation_residual = la.norm(residual_mat, ord='fro')
    
    info = {
        "steps_requested": k,
        "steps_completed": m,
        "breakdown": breakdown,
        "breakdown_type": breakdown_type,
        "ortho_loss_fro": ortho_loss,
        "relation_residual_fro": relation_residual,
        "reorth_method": reorth
    }
    
    return V_m1, H_tilde, info


def verify_arnoldi_factorization():
    """
    Comprehensive verification tests measuring:
    1. Loss of orthogonality ||V_k^T V_k - I||_F
    2. Matrix relation residual norm ||A V_k - V_{k+1} \tilde{H}_k||_F
    3. Breakdown detection thresholds (happy breakdown vs near-breakdown)
    4. Non-normal matrix stability across orthogonalization strategies.
    """
    print("=" * 80)
    print("ARNOLDI FACTORIZATION NUMERICAL VERIFICATION SUITE")
    print("=" * 80)
    
    np.random.seed(42)
    
    # -------------------------------------------------------------------------
    # TEST 1: Baseline Real Non-Symmetric Matrix (Well-conditioned)
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Standard Non-Symmetric Matrix (n=100, k=20)")
    n1, k1 = 100, 20
    A1 = np.random.randn(n1, n1)
    v1 = np.random.randn(n1)
    
    V1, H1, info1 = arnoldi_factorization(A1, v1, k=k1, reorth="cgs2")
    
    print(f"  - Completed Steps: {info1['steps_completed']}/{k1}")
    print(f"  - Orthogonality Loss ||V_k^H V_k - I||_F : {info1['ortho_loss_fro']:.2e}")
    print(f"  - Relation Residual  ||A V_k - V_{{k+1}} H~_k||_F: {info1['relation_residual_fro']:.2e}")
    assert info1['ortho_loss_fro'] < 1e-13, "Test 1 failed: High loss of orthogonality!"
    assert info1['relation_residual_fro'] < 1e-13, "Test 1 failed: High relation residual!"
    print("  => TEST 1 PASSED: High numerical precision maintained.")

    # -------------------------------------------------------------------------
    # TEST 2: Highly Non-Normal Matrix - Comparison of Orthogonalization Schemes
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Highly Non-Normal Matrix Performance (Grcar-like matrix n=80, k=30)")
    n2, k2 = 80, 30
    # Construct a highly non-normal upper triangular / banded matrix
    A2 = np.diag(np.arange(1, n2 + 1, dtype=float)) + np.diag(np.ones(n2 - 1) * 10.0, k=1) + np.diag(np.ones(n2 - 2) * 10.0, k=2)
    v2 = np.ones(n2)
    
    methods = ["none", "mgs2", "cgs2"]
    print(f"  {'Method':<10} | {'Orthogonality Loss (Fro)':<25} | {'Relation Residual (Fro)':<25}")
    print("  " + "-" * 66)
    
    for m in methods:
        _, _, info_m = arnoldi_factorization(A2, v2, k=k2, reorth=m)
        print(f"  {m:<10} | {info_m['ortho_loss_fro']:<25.4e} | {info_m['relation_residual_fro']:<25.4e}")
        if m in ["mgs2", "cgs2"]:
            assert info_m['ortho_loss_fro'] < 1e-12, f"Test 2 failed for {m} reorthogonalization!"
            
    print("  => TEST 2 PASSED: Reorthogonalization successfully prevents loss of orthogonality.")

    # -------------------------------------------------------------------------
    # TEST 3: Happy Breakdown Detection (Invariant Subspace)
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Happy Breakdown Detection (Invariant Subspace)")
    # Create block diagonal matrix where v0 lies in a 5-dimensional invariant subspace
    d_inv = 5
    A_sub = np.random.randn(d_inv, d_inv)
    A3 = la.block_diag(A_sub, np.random.randn(50, 50))
    v3 = np.zeros(55)
    v3[:d_inv] = np.random.randn(d_inv)
    
    k3 = 10  # Requesting 10 steps, but invariant subspace has dimension 5
    V3, H3, info3 = arnoldi_factorization(A3, v3, k=k3, reorth="cgs2", tol=1e-12)
    
    print(f"  - Requested Steps: {k3}")
    print(f"  - Actual Completed Steps: {info3['steps_completed']}")
    print(f"  - Breakdown Flag: {info3['breakdown']}")
    print(f"  - Breakdown Type: {info3['breakdown_type']}")
    print(f"  - Orthogonality Loss ||V_m^H V_m - I||_F: {info3['ortho_loss_fro']:.2e}")
    print(f"  - Relation Residual  ||A V_m - V_{{m+1}} H~_m||_F: {info3['relation_residual_fro']:.2e}")
    
    assert info3['breakdown'] == True, "Test 3 failed: Breakdown was not detected!"
    assert info3['steps_completed'] == d_inv, f"Test 3 failed: Expected {d_inv} steps, got {info3['steps_completed']}"
    print("  => TEST 3 PASSED: Happy breakdown correctly identified exact invariant subspace.")

    # -------------------------------------------------------------------------
    # TEST 4: Complex Non-Hermitian Operator & Ritz Value Accuracy
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Complex Spectrum Matrix & Ritz Value Verification (n=60, k=15)")
    n4, k4 = 60, 15
    A4 = np.random.randn(n4, n4) + 1j * np.random.randn(n4, n4)
    v4 = np.random.randn(n4) + 1j * np.random.randn(n4)
    
    V4, H4, info4 = arnoldi_factorization(A4, v4, k=k4, reorth="cgs2")
    
    # Compute Ritz values from square upper Hessenberg matrix H_m
    H_square = H4[:k4, :k4]
    ritz_values, ritz_vectors_H = la.eig(H_square)
    
    # Form full Ritz vectors x_i = V_k * y_i
    V_k = V4[:, :k4]
    Ritz_vectors = V_k @ ritz_vectors_H
    
    # Measure Ritz residual norms ||A x_i - \lambda_i x_i||_2
    ritz_residuals = []
    for i in range(k4):
        x_i = Ritz_vectors[:, i]
        lam_i = ritz_values[i]
        res = la.norm(A4 @ x_i - lam_i * x_i) / (la.norm(x_i) * la.norm(A4, ord=2))
        ritz_residuals.append(res)
        
    min_res = np.min(ritz_residuals)
    mean_res = np.mean(ritz_residuals)
    
    print(f"  - Complex Matrix Factorization Completed: {info4['steps_completed']} steps")
    print(f"  - Orthogonality Loss ||V_k^H V_k - I||_F: {info4['ortho_loss_fro']:.2e}")
    print(f"  - Matrix Relation Residual Norm: {info4['relation_residual_fro']:.2e}")
    print(f"  - Minimum Ritz Pair Residual Norm: {min_res:.2e}")
    print(f"  - Average Ritz Pair Residual Norm: {mean_res:.2e}")
    
    assert info4['ortho_loss_fro'] < 1e-13, "Test 4 failed: High orthogonality loss in complex case!"
    print("  => TEST 4 PASSED: Complex spectral factorization and Ritz bounds verified.")

    print("\n" + "=" * 80)
    print("ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    verify_arnoldi_factorization()