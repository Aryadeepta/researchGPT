# H5 - quadratic-checksum subset certificate

Implement JSON-stdin/stdout `solution.py`. Return exactly k distinct in-range
indices selecting values totaling `target`, whose selected squared-value sum is
congruent to `square_checksum` modulo positive `modulus`, or found=false.

In `Solution.lean`, import `Spec` and prove fully generic theorems
`solution_certificate_sound` (certBool implies CertProp) and
`solution_sumSq_perm : H5.SumSqPermutationInvariant`, without bypasses.
