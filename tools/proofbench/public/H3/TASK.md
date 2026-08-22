# H3 - turnpike reconstruction

Implement JSON-stdin/stdout `solution.py`. Given `n` and pairwise absolute
distances with multiplicity, return found=true and sorted, distinct integer
points beginning at zero whose complete pair-distance multiset is exactly the
input, or found=false with an empty points list.

In `Solution.lean`, import `Spec` and prove the exact generic theorems:

`solution_translation_invariance : H3.TranslationInvariant`

`solution_reflection_invariance : H3.ReflectionInvariant`

No proof escape hatches are allowed.
