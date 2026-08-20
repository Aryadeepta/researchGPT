import Std
set_option autoImplicit false
namespace H5
def pickedSum (xs idxs : List Nat) : Nat := idxs.foldl (fun a i => a + xs.getD i 0) 0
def pickedSquareSum (xs idxs : List Nat) : Nat :=
  idxs.foldl (fun a i => let x := xs.getD i 0; a + x*x) 0
def sumSqAll (xs : List Nat) : Nat := (xs.map (fun x => x*x)).sum
def CertProp (xs : List Nat) (k target modulus checksum : Nat) (idxs : List Nat) : Prop :=
  idxs.Nodup ∧ idxs.length = k ∧ (∀ i ∈ idxs, i < xs.length) ∧
  pickedSum xs idxs = target ∧ modulus > 0 ∧
  pickedSquareSum xs idxs % modulus = checksum % modulus
def certBool (xs : List Nat) (k target modulus checksum : Nat) (idxs : List Nat) : Bool :=
  decide idxs.Nodup && decide (idxs.length = k) && idxs.all (fun i => decide (i < xs.length)) &&
  decide (pickedSum xs idxs = target) && decide (modulus > 0) &&
  decide (pickedSquareSum xs idxs % modulus = checksum % modulus)
def SumSqPermutationInvariant : Prop := ∀ xs ys : List Nat, xs.Perm ys → sumSqAll xs = sumSqAll ys
end H5
