# H4 - Langford-style pairing

Implement JSON-stdin/stdout `solution.py`. Given `n` and `anchor_parity`, find
a length `2*n` sequence with two copies of each 1..n. If occurrences of k are
i<j, require `j=i+k+1`; the first occurrence of n has the requested parity.
Return found=false and an empty sequence if unsatisfiable.

In `Solution.lean`, import `Spec` and prove the fully generic theorem
`solution_pair_sound : H4.pairBool ... = true -> H4.PairProp ...`.
