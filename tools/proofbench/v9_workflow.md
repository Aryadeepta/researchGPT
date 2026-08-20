# ProofBench V9 workflow

The supervisor owns state. The local agent emits exactly one `proofctl` action:
`status`, `read`, `diagnostic`, `diff`, `check`, `patch`, `revert`, or `finish`.
Only bounded, SHA-guarded unified patches may edit candidates. Python is frozen after
public execution passes; Lean is frozen after compile, theorem-shape, and axiom checks.
Plateau is based on validator rank/fingerprint, never a source digest. Luna may repair
once after a plateau; Terra once only after Luna failed to improve and another plateau.
Hidden suites are retained privately and logged only as commitment/count.
