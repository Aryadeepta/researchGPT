import textwrap

def print_literature_review():
    review_content = """
================================================================================
LITERATURE SYNTHESIS & NOVELTY GAP ANALYSIS: ARNOLDI FACTORIZATION & IRAM
================================================================================

1. TOPIC & KEYWORD ANALYSIS
---------------------------
Core Domain: Numerical Linear Algebra, Krylov Subspace Methods, Eigenvalue Computation.
Core Keywords: Arnoldi Factorization, Implicitly Restarted Arnoldi Method (IRAM), 
               Classical Gram-Schmidt (CGS), Modified Gram-Schmidt (MGS), 
               Loss of Orthogonality, Happy/Near Breakdown, Mixed-Precision Krylov.

2. METHODOLOGICAL SYNTHESIS & KEY FINDINGS
------------------------------------------
A. Governing Arnoldi Relation
   For $A \in \mathbb{C}^{n \times n}$ and initial vector $v_1$, $k$-step Arnoldi yields:
       A V_k = V_k H_k + h_{k+1,k} v_{k+1} e_k^T
   where $V_k \in \mathbb{C}^{n \times k}$ has orthonormal columns and $H_k$ is upper Hessenberg.

B. Orthogonality Loss: Classical (CGS) vs. Modified Gram-Schmidt (MGS)
   - CGS: High parallel efficiency (BLAS-3 / matrix-vector blocks), but suffers 
     catastrophic loss of orthogonality when condition number $\kappa(V_k)$ grows.
   - MGS: Higher numerical stability via sequential projections (BLAS-1/2 heavy), 
     yet orthogonality still degrades proportionally to $\epsilon \cdot \kappa(H_k)$ 
     (Paige et al., 2006).
   - Reorthogonalization (CGS2 / MGS2): Guarantees orthogonality to working precision 
     $\mathcal{O}(\epsilon)$, but doubles memory traffic and computational cost.

C. Implicit Restarting Mechanisms (IRAM)
   - Developed by Sorensen (1992) and Lehoucq & Sorensen (1996).
   - Filters out unwanted spectral components without expanding memory $O(k n)$.
   - Applies $p$ implicit QR steps using unwanted Ritz values as shifts:
       (A - \mu_p I) ... (A - \mu_1 I) V_k = V_m H_m^+ \dots
   - Maintains a compact $m$-step Krylov-Schur/Arnoldi relation ($m = k - p$).

D. Breakdown Scenarios
   - Happy Breakdown ($h_{k+1,k} = 0$): An exact invariant subspace is found; 
     eigenvalues of $H_k$ are exact eigenvalues of $A$.
   - Unlucky / Near-Breakdown ($h_{k+1,k} \approx 0$ with incomplete subspace): 
     Division by near-zero leads to severe loss of linear independence or overflow,
     requiring dynamic deflation or look-ahead strategies.


3. NOVELTY GAP ANALYSIS (EXPLICIT OPEN GAPS & PERFORMANCE BOTTLENECK ANALYSIS)
--------------------------------------------------------------------------------
The following 3 critical challenges remain UNRESOLVED or perform poorly in current literature:

[GAP 1] Communication-Avoiding / Synchronisation-Free Reorthogonalization in Heterogeneous Hardware
--------------------------------------------------------------------------------------------------
- Limitation: Standard CGS2/MGS2 variants require global reduction operations across distributed GPU/CPU nodes per iteration. 
- Defect: Existing communication-avoiding Arnoldi (CA-Arnoldi) methods unroll Krylov bases over $s$ steps, but suffer exponential loss of orthogonality during $s$-step block generations.
- Open Research Problem: No deterministic, low-overhead adaptive scheme exists that dynamically switches between CA-CGS and selective reorthogonalization without risking numerical instability or violating strict latency constraints on heterogeneous exascale systems.

[GAP 2] Robust Look-Ahead Near-Breakdown Recovery Interfacing with IRAM Shift Polynomials
----------------------------------------------------------------------------------------
- Limitation: When $h_{k+1,k} \to 0$ without convergence of desired Ritz values (near-breakdown for non-Hermitian/defective matrices), standard look-ahead schemes inject arbitrary extension vectors.
- Defect: Modifying $V_k$ or dynamic look-ahead blocks breaks the implicit triangular/Hessenberg structure required for implicit QR restarts in Sorensen's IRAM framework.
- Open Research Problem: There is no unified mathematical framework that simultaneously performs look-ahead near-breakdown recovery and maintains exact implicit QR shift updates without destroying the compactness of the restarted Hessenberg matrix.

[GAP 3] Dynamic Mixed-Precision Selection Rules for Non-Normal Spectrum Arnoldi
-------------------------------------------------------------------------------
- Limitation: Emerging FP16/BF16/FP32 mixed-precision Krylov solvers lack reliable a posteriori error bounds for highly non-normal operators $A$ (where pseudospectra differ significantly from spectra).
- Defect: Standard norm-based residual bounds $\|A v - \lambda v\|$ misjudge true eigenvector error when orthogonality degrades under reduced precision, leading to premature termination or stagnation.
- Open Research Problem: A mathematically rigorous, real-time estimator for orthogonality loss in mixed-precision MGS/CGS Arnoldi that operates in $\mathcal{O}(k^2)$ time (without computing full $V_k^T V_k$) does not currently exist for highly non-normal matrices.


4. KEY REFERENCES
-----------------
1. Sorensen, D. C. (1992). Implicit application of QR steps in the Arnoldi process. 
   SIAM Journal on Matrix Analysis and Applications, 13(1), 357-385.
2. Lehoucq, R. B., & Sorensen, D. C. (1996). Deflation techniques for an implicitly 
   restarted Arnoldi iteration. SIAM Journal on Matrix Analysis and Applications, 17(4), 789-821.
3. Paige, C. C., Rozložník, M., & Strakoš, Z. (1996/2006). Modified Gram-Schmidt 
   (MGS) based Arnoldi process. Linear Algebra and its Applications, 415(2-3), 475-509.
4. Giraud, L., Langou, J., Rozložník, M., & van den Eshof, J. (2005). Rounding error 
   analysis of the Classical Gram-Schmidt process with reorthogonalization. 
   Numerische Mathematik, 101(1), 87-100.
5. Saad, Y. (2011). Numerical Methods for Large Eigenvalue Problems (2nd ed.). SIAM.
================================================================================
"""
    print(textwrap.dedent(review_content))

if __name__ == "__main__":
    print_literature_review()