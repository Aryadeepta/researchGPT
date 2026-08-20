# H2 — resource-bounded directed path

Implement JSON-stdin/stdout `solution.py`. Input contains `n_states`, `start`,
`goal`, `max_steps`, `budget`, and edge triples `[src,dst,cost]`. A found
answer supplies valid edge indices forming a directed path from start to goal,
with at most the step and total-cost bounds. Otherwise return found=false.

In `Solution.lean`, import `Spec` and prove generic theorems
`solution_follows_sound` (followsBool implies followsProp) and
`solution_certificate_sound` (certBool implies CertProp), without proof bypasses.
