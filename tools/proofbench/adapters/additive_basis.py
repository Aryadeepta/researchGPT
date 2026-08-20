"""Finite additive-basis semantics, independent of any research trial."""
from __future__ import annotations

import re
from typing import Any

from tools.proofbench.proof_engineering import additive_basis_model


class AdditiveBasisAdapter:
    """Runtime semantic adapter for arbitrary finite additive-basis inputs."""

    def __init__(self, specification: dict[str, Any]):
        self.specification = specification

    def describe(self) -> str:
        s = self.specification
        return f"Finite additive-basis semantics on 0..{s['n']} inclusive; forbidden: {s['forbidden']}; cardinality at most {s['max_selected']}."

    def build_constraint_model(self, obligation_hash: str):
        s = self.specification
        return additive_basis_model(int(s["n"]), set(s["forbidden"]), int(s["max_selected"]), obligation_hash)

    def render_semantics(self) -> str:
        s = self.specification; n = int(s["n"]); k = int(s["max_selected"])
        forbidden = ", ".join(map(str, sorted(s["forbidden"])))
        return f'''import Init.Tactics
set_option autoImplicit false
def choose : List Nat → Nat → List (List Nat)
  | [], 0 => [[]]
  | [], _ + 1 => []
  | x :: xs, 0 => [[]]
  | x :: xs, q + 1 => choose xs (q + 1) ++ (choose xs q).map (fun A => x :: A)
def below (u : List Nat) : Nat → List (List Nat)
  | 0 => []
  | q + 1 => below u q ++ choose u q
def N : Nat := {n}
def F : List Nat := [{forbidden}]
def U : List Nat := List.range (N + 1)
def candidates : List (List Nat) := below U ({k} + 1)
def admissible (A : List Nat) : Bool :=
  (F.all (fun z => !(A.contains z))) &&
  (List.range (N + 1)).all (fun t => A.any (fun a => A.any (fun b => a + b == t)))
def noAdmissible : Bool := candidates.all (fun A => !(admissible A))
'''

    def render_semantic_lemma(self, item: dict[str, Any], facts: dict[str, bool]) -> tuple[str, str] | None:
        if item.get("method") in {"unit-propagation", "cardinality-propagation"}:
            var, value = item.get("variable"), item.get("value")
            if var and value is not None and re.fullmatch(r"x_[0-9]+", var):
                i = int(var.split("_", 1)[1]); predicate = f"A.contains {i}" if value else f"!(A.contains {i})"
                statement = f"{i} must be selected" if value else f"{i} is forbidden"
                return statement, f"(candidates.all (fun A => !(admissible A) || {predicate})) = true"
        alternatives = item.get("alternatives")
        if alternatives and all(isinstance(a, list) for a in alternatives):
            statement = " or ".join(" and ".join(x.replace("x_", "") + " is selected" for x in a) for a in alternatives)
            terms = ["(" + " && ".join(f"A.contains {int(x.split('_', 1)[1])}" for x in a) + ")" for a in alternatives]
            return statement, f"(candidates.all (fun A => !(admissible A) || ({' || '.join(terms)}))) = true"
        return None
