import Std
set_option autoImplicit false
namespace H2
structure Edge where
  src : Nat
  dst : Nat
  cost : Nat
deriving DecidableEq, Repr, Inhabited
def chosenEdges (edges : List Edge) (idxs : List Nat) : List Edge := idxs.map (fun i => edges.getD i default)
def followsProp : List Edge → Nat → Nat → Prop
  | [], current, goal => current = goal
  | e :: rest, current, goal => e.src = current ∧ followsProp rest e.dst goal
def followsBool : List Edge → Nat → Nat → Bool
  | [], current, goal => decide (current = goal)
  | e :: rest, current, goal => decide (e.src = current) && followsBool rest e.dst goal
def edgeCost (edges : List Edge) : Nat := (edges.map Edge.cost).sum
def CertProp (edges : List Edge) (start goal maxSteps budget : Nat) (idxs : List Nat) : Prop :=
  (∀ i ∈ idxs, i < edges.length) ∧ idxs.length ≤ maxSteps ∧
  followsProp (chosenEdges edges idxs) start goal ∧ edgeCost (chosenEdges edges idxs) ≤ budget
def certBool (edges : List Edge) (start goal maxSteps budget : Nat) (idxs : List Nat) : Bool :=
  idxs.all (fun i => decide (i < edges.length)) && decide (idxs.length ≤ maxSteps) &&
  followsBool (chosenEdges edges idxs) start goal && decide (edgeCost (chosenEdges edges idxs) ≤ budget)
end H2
