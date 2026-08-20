
import Std
set_option autoImplicit false

namespace H1

def pickedSum
    (xs : List Nat)
    (idxs : List Nat) : Nat :=
  idxs.foldl
    (fun acc i => acc + xs.getD i 0)
    0

def CertProp
    (xs : List Nat)
    (k target modulus residue : Nat)
    (idxs : List Nat) : Prop :=
  idxs.Nodup ∧
  idxs.length = k ∧
  (∀ i ∈ idxs, i < xs.length) ∧
  pickedSum xs idxs = target ∧
  modulus > 0 ∧
  idxs.sum % modulus = residue % modulus

def certBool
    (xs : List Nat)
    (k target modulus residue : Nat)
    (idxs : List Nat) : Bool :=
  decide idxs.Nodup &&
  decide (idxs.length = k) &&
  idxs.all
    (fun i => decide (i < xs.length)) &&
  decide (pickedSum xs idxs = target) &&
  decide (modulus > 0) &&
  decide
    (idxs.sum % modulus =
     residue % modulus)

end H1
