
import Std
set_option autoImplicit false

namespace H4

def PairProp
    (sequence : List Nat)
    (k i j : Nat) : Prop :=
  i < sequence.length ∧
  j < sequence.length ∧
  i < j ∧
  j = i + k + 1 ∧
  sequence.getD i 0 = k ∧
  sequence.getD j 0 = k

def pairBool
    (sequence : List Nat)
    (k i j : Nat) : Bool :=
  decide (i < sequence.length) &&
  decide (j < sequence.length) &&
  decide (i < j) &&
  decide (j = i + k + 1) &&
  decide (sequence.getD i 0 = k) &&
  decide (sequence.getD j 0 = k)

end H4
