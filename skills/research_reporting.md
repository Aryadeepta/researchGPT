```json
{
  "skill_name": "research_reporting.md",
  "title": "Benchmarking and Scientific Reporting Workflow for Krylov/Arnoldi Implementations",
  "goal": "Conduct rigorous benchmarking experiments comparing the proposed implementation against standard Krylov/Arnoldi baselines across matrix condition numbers, subspace dimensions, and computational overhead, then compile the findings into a publication-ready scientific document.",
  "workflow": {
    "phase_1_experimental_design": [
      {
        "step_id": "1.1",
        "task": "Define Matrix Testbed and Conditioning Parameters",
        "details": "Generate synthetic matrices and select real-world benchmark matrices (e.g., SuiteSparse collection) with condition numbers scaled log-uniformly from 10^2 to 10^10. Ensure both symmetric/Hermitian and non-symmetric instances are included.",
        "inputs": ["Target matrix dimensions N", "Condition number range [10^2, 10^10]", "Sparsity patterns"],
        "outputs": ["Matrix dataset catalog", "Generator scripts with fixed seeds for reproducibility"]
      },
      {
        "step_id": "1.2",
        "task": "Configure Baseline Implementations",
        "details": "Set up standard reference Krylov/Arnoldi solvers (e.g., ARPACK, SciPy `eigs`/`gmres`, MATLAB `arnoldi`) with identical stopping criteria (e.g., relative residual tolerance 1e-8) and orthogonalization schemes (e.g., MGS, CGS2).",
        "inputs": ["Reference libraries", "Convergence thresholds", "Orthogonalization strategy"],
        "outputs": ["Baseline execution pipeline"]
      }
    ],
    "phase_2_benchmarking_execution": [
      {
        "step_id": "2.1",
        "task": "Parameter Sweep across Subspace Dimensions",
        "details": "Execute proposed and standard algorithms across subspace dimensions m in {10, 20, 50, 100, 200}. Record convergence rate, iteration count, and re-orthogonalization frequency.",
        "inputs": ["Matrix dataset", "Subspace sizes m"],
        "outputs": ["Convergence metrics logs (JSON/CSV)"]
      },
      {
        "step_id": "2.2",
        "task": "Profile Compute Overhead and Numerical Stability",
        "details": "Measure wall-clock execution time, peak memory usage (RAM/VRAM), per-iteration FLOP counts, and loss of orthogonality ||Q^T Q - I||_F.",
        "inputs": ["Execution environment (CPU/GPU specs)", "Profiling tools (Perf/Nvprof/tracemalloc)"],
        "outputs": ["Resource consumption logs", "Orthogonality drift metrics"]
      }
    ],
    "phase_3_data_aggregation_analysis": [
      {
        "step_id": "3.1",
        "task": "Tabular and Statistical Aggregation",
        "details": "Compute mean, standard deviation, and speedup ratios across 10 independent runs per configuration. Format data into latex-ready summary tables.",
        "inputs": ["Raw log files"],
        "outputs": ["Formatted metrics tables (Markdown/LaTeX)"]
      },
      {
        "step_id": "3.2",
        "task": "Generate Comparative Visualizations",
        "details": "Plot condition number vs. convergence time, subspace size vs. memory overhead, and residual norm vs. iteration count curves.",
        "inputs": ["Aggregated CSV data"],
        "outputs": ["Vector graphics plots (PDF/PNG)"]
      }
    ],
    "phase_4_scientific_report_compilation": [
      {
        "step_id": "4.1",
        "task": "Draft Methodology and Experimental Setup",
        "details": "Document mathematical foundations of the proposed method vs. Arnoldi baselines, system hardware parameters, and test matrix properties.",
        "inputs": ["System specs", "Algorithm formulation"],
        "outputs": ["Markdown Sections: Methodology & Setup"]
      },
      {
        "step_id": "4.2",
        "task": "Synthesize Findings and Limitations",
        "details": "Highlight key takeaways, speedup bounds, regimes where proposed method outperforms baselines, and identified numerical edge cases.",
        "inputs": ["Aggregated data", "Generated plots"],
        "outputs": ["Markdown Sections: Results, Discussion, & Conclusion"]
      },
      {
        "step_id": "4.3",
        "task": "Final Document Assembly (research_reporting.md)",
        "details": "Assemble full report adhering to academic formatting standards (Abstract, Introduction, Method, Experiments, Results, Discussion, Conclusion, References).",
        "inputs": ["All section drafts", "Tables", "Figures"],
        "outputs": ["research_reporting.md"]
      }
    ]
  },
  "template_structure": {
    "filename": "research_reporting.md",
    "sections": [
      "# Benchmarking and Comparative Evaluation: Proposed Implementation vs Standard Krylov/Arnoldi Methods",
      "## Executive Summary / Abstract",
      "## 1. Introduction & Background",
      "## 2. Experimental Methodology & Baseline Setup",
      "### 2.1 Proposed Algorithm Overview",
      "### 2.2 Baseline Variants (ARPACK, Standard Arnoldi/MGS)",
      "### 2.3 Test Matrix Characteristics & Condition Numbers",
      "## 3. Experimental Results",
      "### 3.1 Robustness across Matrix Condition Numbers",
      "### 3.2 Impact of Subspace Dimension (m)",
      "### 3.3 Computational Overhead & Memory Profiling",
      "## 4. Discussion & Comparative Analysis",
      "## 5. Limitations & Edge Cases",
      "## 6. Conclusion & Future Work",
      "## Appendix: Tabular Metrics and Raw Benchmark Logs"
    ]
  }
}
```