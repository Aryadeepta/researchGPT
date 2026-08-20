# Generic Python + Lean Proof-Development Workflow

This file contains generic workflow guidance only.

It must never contain:
- hidden benchmark inputs,
- hidden benchmark expected answers,
- hidden seeds,
- copied held-out solutions,
- previous held-out transcripts.

## Architecture

Separate:

    mathematical specification : Prop

from:

    executable checker : Bool

Then formally prove soundness:

    checker ... = true → specification ...

Do not require a monolithic `Decidable specification` merely because an
executable checker is desired.

## Lean loop

1. Compile immediately after introducing a definition.
2. After an error, classify it:
   - parser/API compatibility,
   - elaboration,
   - Bool/Prop mismatch,
   - recursion,
   - rewriting,
   - theorem-shape mismatch,
   - missing finite decision procedure.
3. Fix the smallest cause.
4. Re-run the smallest failing Lean file.
5. Run the complete proof file.
6. Use `#check` to verify final theorem shape.
7. Use `#print axioms` on final theorem(s).

Never use:
- sorry,
- admit,
- project-local axioms,
- unsafe shortcuts,
- sorryAx.

## Finite executable predicates

Prefer explicit finite operations such as:
- List.all,
- List.foldl,
- recursive Bool checkers,
- decide on atomic equality/inequality,
- finite dynamic programming.

When proving soundness, bridge the Boolean computation back to the explicit
mathematical proposition.

## Python loop

Separate:
1. parsing,
2. search,
3. certificate generation,
4. certificate validation.

Use:
- exact validators,
- deterministic algorithms where possible,
- resource-aware DP/backtracking instead of blind enumeration,
- explicit boundary tests,
- exact JSON I/O.

Before returning found=true, validate the witness independently.

## Debugging

After a failure:
1. state what failed;
2. identify the invariant violated;
3. fix the cause rather than special-casing the observed example;
4. rerun the smallest check;
5. then rerun broader validation.
