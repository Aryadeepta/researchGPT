# Proof orchestrator

`proof_orchestrator.py` is a public calibration component, not a hidden-suite
runner.  It owns a bounded frontier of Lean-validated prefixes.  A temporary
probe appends `trace_state; fail` only to inspect unsolved goals; this is not
completion evidence.  Completion is always delegated to `proof_gym.validate`,
which compiles the candidate and checks theorem shape and axioms.

The optional planner receives a disposable JSON capsule made only from the
public declaration, current goal, validated prefix, and bounded public
diagnostics.  It is disabled unless `PROOFBENCH_ORCH_ENABLE_REMOTE=1` (or the
CLI flag) is set.  Plans and `have` chunks are fed back one item at a time
through Lean; they are never applied wholesale.

H1 integration is intentionally not exposed yet.  The adapter seam is the
`GymCase`-like public declaration plus a final verifier: an H1 adapter must
read only public `TASK.md`, `Spec.lean`, and public `Solution.lean`, construct
a declaration/expected theorem wrapper, and use `v9_controller.lean_hooks`
for final shape and axiom checks.  It must not call `hidden_suite`, pass hidden
cases, or reuse the qualification controller workspace.
