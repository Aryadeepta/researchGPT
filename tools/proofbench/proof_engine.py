"""Generic, deterministic patch-and-validate engine for ProofBench V9.

This module deliberately knows nothing about a model provider or hidden cases.
The controller supplies public validators and keeps private qualification data.
"""
from __future__ import annotations
import difflib, hashlib, json, re, subprocess, time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

ALLOWED_FILES = ("solution.py", "Solution.lean")
MAX_PATCH_LINES, MAX_PATCH_CHARS = 80, 6000

class Phase(str, Enum): PYTHON="PYTHON"; LEAN="LEAN"; PUBLIC_COMPLETE="PUBLIC_COMPLETE"
class PyState(str, Enum): ABSENT="ABSENT"; COMPILE_FAILURE="COMPILE_FAILURE"; COMPILE_PASS="COMPILE_PASS"; PUBLIC_RUNTIME_FAILURE="PUBLIC_RUNTIME_FAILURE"; PUBLIC_PASS="PUBLIC_PASS"
class LeanState(str, Enum): ABSENT="ABSENT"; COMPILE_FAILURE="COMPILE_FAILURE"; COMPILE_PASS="COMPILE_PASS"; THEOREM_SHAPE_PASS="THEOREM_SHAPE_PASS"; AXIOM_INTEGRITY_PASS="AXIOM_INTEGRITY_PASS"
PY_RANK={x:i for i,x in enumerate(PyState)}; LEAN_RANK={x:i for i,x in enumerate(LeanState)}
FORBIDDEN=re.compile(r"\b(?:sorry|admit|sorryAx)\b|^\s*(?:axiom|unsafe\s+(?:def|theorem))\b",re.M)

def sha(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def fingerprint(text: str) -> str: return sha(re.sub(r"\d+", "#", re.sub(r"\s+", " ", text)).encode())[:16]
def run(command, cwd, seconds):
    return subprocess.run(["/usr/bin/timeout","--signal=TERM","--kill-after=5s",f"{seconds}s",*map(str,command)],cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)

@dataclass
class Result:
    ok: bool; code: str; diagnostic: str=""; checkpoint: int|None=None

class ProofEngine:
    def __init__(self, workspace, public_python=None, lean="lean", python="python3", persist=None,
                 theorem_shape=None, axiom_integrity=None):
        # Validators are deliberately injected: this generic engine has no task-specific
        # knowledge and must never turn a successful compiler invocation into a proof pass.
        self.workspace=Path(workspace); self.public_python=public_python or (lambda _p: (False,"NO_PUBLIC_PYTHON_VALIDATOR")); self.lean=lean; self.python=python; self.persist=persist or (lambda _n,_r:None)
        self.theorem_shape=theorem_shape or (lambda _p: (False,"NO_THEOREM_SHAPE_VALIDATOR"))
        self.axiom_integrity=axiom_integrity or (lambda _p: (False,"NO_AXIOM_INTEGRITY_VALIDATOR"))
        self.phase=Phase.PYTHON; self.py_state=PyState.ABSENT; self.lean_state=LeanState.ABSENT; self.frozen=set(); self.patch_count=0; self.actions=0; self.checkpoints=[]; self.current=self.best=self.previous=None; self.history=[]
        self.checkpoint("scaffold")
    def path(self,n): return self.workspace/n
    def state(self):
        return {"phase":self.phase.value,"python":self.py_state.value,"lean":self.lean_state.value,"frozen":sorted(self.frozen),"current_checkpoint":self.current,"best_checkpoint":self.best,"previous_checkpoint":self.previous,"patch_count":self.patch_count,"tool_action_count":self.actions}
    def checkpoint(self, reason):
        snap={n:self.path(n).read_text() if self.path(n).exists() else None for n in ALLOWED_FILES}; rank=self.rank(); item={"id":len(self.checkpoints),"reason":reason,"files":snap,"rank":rank}; self.checkpoints.append(item); self.previous=self.current; self.current=item["id"]
        if self.best is None or rank>=self.checkpoints[self.best]["rank"]: self.best=self.current
        self.persist("checkpoints", {k:v for k,v in item.items() if k!="files"}|{"best":self.best,"previous":self.previous}); return item["id"]
    def rank(self): return PY_RANK[self.py_state]*10+LEAN_RANK[self.lean_state]
    def read(self,n,start=None,end=None):
        if n not in ALLOWED_FILES and n not in ("TASK.md","Spec.lean"): return Result(False,"PATH_REJECTED")
        lines=self.path(n).read_text().splitlines(True); return Result(True,"OK","".join(lines[(start or 1)-1:end]))
    def patch(self,n, expected, diff):
        if n not in ALLOWED_FILES: return Result(False,"PATH_REJECTED")
        if n in self.frozen: return Result(False,"FILE_FROZEN")
        if self.phase == Phase.PYTHON and n != "solution.py": return Result(False,"PHASE_FILE_REJECTED")
        if self.phase == Phase.LEAN and n != "Solution.lean": return Result(False,"PHASE_FILE_REJECTED")
        if self.phase == Phase.PUBLIC_COMPLETE: return Result(False,"FILE_FROZEN")
        old=self.path(n).read_text() if self.path(n).exists() else ""
        if sha(old.encode()) != expected: return Result(False,"STALE_PATCH")
        changed=sum(1 for x in diff.splitlines() if x.startswith(("+","-")) and not x.startswith(("+++","---")))
        if len(diff)>MAX_PATCH_CHARS or changed>MAX_PATCH_LINES: return Result(False,"PATCH_TOO_LARGE")
        lines=diff.splitlines(True)
        if not any(x.startswith("--- ") for x in lines) or not any(x.startswith("+++ ") for x in lines): return Result(False,"MALFORMED_PATCH")
        # apply unified diff using stdlib restoration only after strict one-file headers
        headers=[x[4:].strip().split("\t")[0] for x in lines if x.startswith(("--- ","+++ "))]
        if any(Path(x).name != n for x in headers): return Result(False,"MULTI_FILE_PATCH")
        try:
            hunks=[]; i=0; oldlines=old.splitlines(True); out=[]; cursor=0
            while i<len(lines):
                if lines[i].startswith("@@ "):
                    m=re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@",lines[i]);
                    if not m: raise ValueError()
                    pos=int(m.group(1))-1; out.extend(oldlines[cursor:pos]); cursor=pos; i+=1
                    while i<len(lines) and not lines[i].startswith("@@ "):
                        x=lines[i]; tag=x[:1]; body=x[1:]
                        if tag==' ':
                            if cursor>=len(oldlines) or oldlines[cursor].rstrip("\n")!=body.rstrip("\n"): raise ValueError()
                            out.append(oldlines[cursor]); cursor+=1
                        elif tag=='-':
                            if cursor>=len(oldlines) or oldlines[cursor].rstrip("\n")!=body.rstrip("\n"): raise ValueError()
                            cursor+=1
                        elif tag=='+': out.append(body)
                        elif not x.startswith("\\ No newline"): raise ValueError()
                        i+=1
                else: i+=1
            out.extend(oldlines[cursor:]); new="".join(out)
            if not hunks and "@@" not in diff: raise ValueError()
        except Exception: return Result(False,"MALFORMED_PATCH")
        self.path(n).write_text(new); self.patch_count+=1; cp=self.checkpoint("patch"); self.persist("patches",{"file":n,"expected_sha256":expected,"patch_sha256":sha(diff.encode()),"patch":diff,"result":"accepted","checkpoint":cp}); return Result(True,"PATCH_ACCEPTED",checkpoint=cp)
    def check_python(self):
        p=self.path("solution.py")
        if not p.exists(): self.py_state=PyState.ABSENT; return Result(False,"MISSING_PYTHON")
        cp=run([self.python,"-m","py_compile",p.name],self.workspace,30)
        if cp.returncode: self.py_state=PyState.COMPILE_FAILURE; return Result(False,"PYTHON_COMPILE_FAILURE",cp.stdout)
        self.py_state=PyState.COMPILE_PASS; ok,msg=self.public_python(p)
        if not ok: self.py_state=PyState.PUBLIC_RUNTIME_FAILURE; return Result(False,"PYTHON_PUBLIC_RUNTIME_FAILURE",msg)
        self.py_state=PyState.PUBLIC_PASS; self.frozen.add("solution.py"); self.phase=Phase.LEAN; self.checkpoint("python frozen"); return Result(True,"PYTHON_PASS")
    def check_lean(self):
        if self.py_state != PyState.PUBLIC_PASS: return Result(False,"PYTHON_GATE")
        p=self.path("Solution.lean")
        if not p.exists(): self.lean_state=LeanState.ABSENT; return Result(False,"MISSING_LEAN")
        if FORBIDDEN.search(p.read_text()): self.lean_state=LeanState.COMPILE_FAILURE; return Result(False,"PROOF_INTEGRITY_FAILURE","forbidden proof construct")
        cp=run([self.lean,p.name],self.workspace,90)
        if cp.returncode: self.lean_state=LeanState.COMPILE_FAILURE; return Result(False,"LEAN_COMPILATION_FAILURE",cp.stdout)
        self.lean_state=LeanState.COMPILE_PASS
        ok,msg=self.theorem_shape(p)
        if not ok: return Result(False,"THEOREM_SHAPE_FAILURE",str(msg)[:4000])
        self.lean_state=LeanState.THEOREM_SHAPE_PASS
        ok,msg=self.axiom_integrity(p)
        if not ok: return Result(False,"AXIOM_INTEGRITY_FAILURE",str(msg)[:4000])
        self.lean_state=LeanState.AXIOM_INTEGRITY_PASS; self.frozen.add("Solution.lean"); self.phase=Phase.PUBLIC_COMPLETE; self.checkpoint("lean frozen"); return Result(True,"LEAN_PASS")
    def revert(self):
        if self.best is None: return Result(False,"NO_CHECKPOINT")
        snap=self.checkpoints[self.best]["files"]
        for n,v in snap.items():
            if v is not None:self.path(n).write_text(v)
        self.current=self.best; return Result(True,"REVERTED",checkpoint=self.best)
    def execute(self, action):
        self.actions+=1; tool=action.get("tool")
        if tool=="status": r=Result(True,"OK",json.dumps(self.state(),sort_keys=True))
        elif tool=="read": r=self.read(action.get("file",""),action.get("start"),action.get("end"))
        elif tool=="patch": r=self.patch(action.get("file",""),action.get("sha256",""),action.get("diff",""))
        elif tool in ("check","diagnostic") and action.get("target")=="python": r=self.check_python()
        elif tool in ("check","diagnostic") and action.get("target")=="lean": r=self.check_lean()
        elif tool=="check" and action.get("target")=="public":
            r=self.check_python()
            if r.ok: r=self.check_lean()
        elif tool=="revert": r=self.revert()
        elif tool=="finish": r=Result(self.phase==Phase.PUBLIC_COMPLETE,"FINISHED" if self.phase==Phase.PUBLIC_COMPLETE else "NOT_COMPLETE")
        else: r=Result(False,"TOOL_REJECTED")
        self.history.append({"tool":tool,"code":r.code,"rank":self.rank(),"fingerprint":fingerprint(r.diagnostic)}); self.persist("tool-actions",{"action":action,"result":r.code,"state":self.state()}); return r
    def plateau(self, minimum=8, window=5):
        h=self.history
        if len(h)<minimum or len(h)<window:return False
        ranks=[x["rank"] for x in h[-window:]]; return max(ranks)==min(ranks)
    def unfreeze(self, name, reason):
        """Exceptional supervisor-only transition; ordinary tools cannot unfreeze."""
        if name not in self.frozen or not reason: return Result(False,"UNFREEZE_REJECTED")
        self.frozen.remove(name); self.phase=Phase.PYTHON if name=="solution.py" else Phase.LEAN
        self.persist("attempts",{"event":"UNFREEZE","file":name,"reason":reason}); return Result(True,"UNFROZEN")

class EscalationPolicy:
    """Bounded model accounting; infrastructure failures are not model attempts."""
    def __init__(self, luna=1, terra=1, sol=0, infrastructure_retries=1):
        self.limits={"luna":min(1,luna),"terra":min(1,terra),"sol":0}
        self.calls={k:0 for k in self.limits}; self.infrastructure_retries={k:0 for k in self.limits}
        self.infrastructure_limit=infrastructure_retries; self.luna_progress=False; self.account_disabled=False
    def allow(self, model, plateau):
        if self.account_disabled or not plateau or self.calls.get(model,0)>=self.limits.get(model,0): return False
        return model=="luna" or (model=="terra" and self.calls["luna"] and not self.luna_progress)
    def infrastructure_retry(self, model):
        if self.infrastructure_retries.get(model, 0) >= self.infrastructure_limit: return False
        self.infrastructure_retries[model] += 1; return True
    def record(self, model, progress=False, account_error=False, substantive=True):
        if substantive and self.calls.get(model,0) < self.limits.get(model,0): self.calls[model]+=1
        if model=="luna" and progress:self.luna_progress=True
        if account_error:self.account_disabled=True
