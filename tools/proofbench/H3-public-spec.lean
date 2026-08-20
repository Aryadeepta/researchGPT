
import Std
set_option autoImplicit false

namespace H3

def dist
    (a b : Int) : Nat :=
  Int.natAbs (b - a)

end H3
