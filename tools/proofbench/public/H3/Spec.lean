import Std
set_option autoImplicit false
namespace H3
def dist (a b : Int) : Nat := Int.natAbs (b - a)
def TranslationInvariant : Prop := ∀ a b t : Int, dist (a + t) (b + t) = dist a b
def ReflectionInvariant : Prop := ∀ a b c : Int, dist (c - a) (c - b) = dist a b
end H3
