# H1 - bounded additive certificate

Write `solution.py`, reading one JSON object from stdin and writing one JSON
object to stdout. Return `{"found":true,"indices":[...]}` exactly when a
certificate can be found; otherwise return `{"found":false,"indices":[]}`.
Indices must be distinct and in range, have length `k`, select values totaling
`target`, and have index sum congruent to `residue` modulo positive `modulus`.

Write `Solution.lean`, import `Spec`, and prove without escape hatches:

`solution_certificate_sound : H1.certBool ... = true -> H1.CertProp ...`

Use the public definitions and a fully generic theorem.
