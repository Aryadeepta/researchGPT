#!/usr/bin/env python3
"""ProofBench V9 restricted local-agent supervisor.

Private qualification objects never leave this process.  The model receives a
small public protocol and can only request one ProofEngine operation per turn.
"""
from __future__ import annotations
import argparse, datetime as dt, difflib, hashlib, json, os, random, shutil, signal, subprocess, sys, tempfile, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; PB=Path(__file__).resolve().parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.proofbench.proof_engine import ProofEngine, Phase, Result, sha, EscalationPolicy, run
from src.local_inference import LocalLLMProvider
from src.llm_gateway import LLMRequest
TASK_IDS=("H1","H2","H3","H4","H5")
REQUIRED={"H1":("solution_certificate_sound",),"H2":("solution_follows_sound","solution_certificate_sound"),"H3":("solution_translation_invariance","solution_reflection_invariance"),"H4":("solution_pair_sound",),"H5":("solution_certificate_sound","solution_sumSq_perm")}
TYPES={"H1":{"solution_certificate_sound":"∀ xs k target modulus residue idxs, H1.certBool xs k target modulus residue idxs = true → H1.CertProp xs k target modulus residue idxs"},"H2":{"solution_follows_sound":"∀ es start goal, H2.followsBool es start goal = true → H2.followsProp es start goal","solution_certificate_sound":"∀ edges start goal maxSteps budget idxs, H2.certBool edges start goal maxSteps budget idxs = true → H2.CertProp edges start goal maxSteps budget idxs"},"H3":{"solution_translation_invariance":"H3.TranslationInvariant","solution_reflection_invariance":"H3.ReflectionInvariant"},"H4":{"solution_pair_sound":"∀ sequence k i j, H4.pairBool sequence k i j = true → H4.PairProp sequence k i j"},"H5":{"solution_certificate_sound":"∀ xs k target modulus checksum idxs, H5.certBool xs k target modulus checksum idxs = true → H5.CertProp xs k target modulus checksum idxs","solution_sumSq_perm":"H5.SumSqPermutationInvariant"}}
ALLOWED={"status","read","diagnostic","diff","check","patch","revert","finish"}
PLAN_ACTIONS=("status","inspect_task","inspect_candidate","diagnostic","check","request_edit","revert","finish")

class V9BoundedIdentityReducer:
    """V9 prompts are already PUBLIC-only and deliberately bounded by the controller.

    This is intentionally not a change to the generic research ContextReducer.
    """
    def __init__(self, maximum=12000): self.maximum=maximum
    def reduce(self, prompt):
        text=str(prompt)
        if len(text)>self.maximum: raise ValueError("V9_PUBLIC_PROMPT_TOO_LARGE")
        return text

def v9_local_provider(provider=None):
    provider=provider or LocalLLMProvider(fallback_provider=None)
    provider.context_reducer=V9BoundedIdentityReducer()
    return provider

PLAN_SCHEMA={"type":"object","properties":{"action":{"type":"string","enum":list(PLAN_ACTIONS)}},"required":["action"],"additionalProperties":False}
MAX_REPLACEMENT_CHARS=6000
# Keep the semantic size limit in supervisor code rather than JSON grammar.
# llama.cpp turns maxLength=N into char{0,N}; large N can exceed its grammar limit.
EDIT_SCHEMA={"type":"object","properties":{"replacement":{"type":"string"}},"required":["replacement"],"additionalProperties":False}

def append(path,obj):
    with Path(path).open("a",encoding="utf-8") as f:f.write(json.dumps(obj,sort_keys=True)+"\n")
def result_root(): return Path.home()/"proofbench-results"/("v9-"+dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
REMOTE_CODES={"REMOTE_DISABLED","REMOTE_MODEL_REJECTED","REMOTE_CODEX_NOT_FOUND","REMOTE_ACCOUNT_FAILURE","REMOTE_TIMEOUT","REMOTE_CODEX_PROCESS_FAILURE","REMOTE_NO_CHANGE","REMOTE_PATCH_REJECTED","REMOTE_VALIDATOR_NO_PROGRESS","REMOTE_PROGRESS","REMOTE_INFRA_UNAVAILABLE","REMOTE_BUDGET_EXHAUSTED","REMOTE_NOT_NEEDED"}
def _safe(text,n=1200):
    # Diagnostics are public process output only; keep records bounded and remove
    # accidental private-looking markers without attempting semantic repair.
    return str(text).replace("hidden","[redacted]")[-n:]
def preflight(lean, task_ids=TASK_IDS):
    try:
        return all(
            run(
                [lean, ROOT / "tools/proofbench/public" / t / "Spec.lean"],
                ROOT,
                30,
            ).returncode == 0
            for t in task_ids
        )
    except OSError:
        return False

def public_cases(task):
    # These are deliberately fixed public examples, not a disguised held-out suite.
    return {"H1":[{"xs":[2,4,7,9],"k":2,"target":11,"modulus":3,"residue":0}],
      "H2":[{"n_states":3,"start":0,"goal":2,"max_steps":2,"budget":5,"edges":[[0,1,2],[1,2,3]]}],
      "H3":[{"n":3,"distances":[2,4,2]}], "H4":[{"n":3,"anchor_parity":0}],
      "H5":[{"xs":[1,2,3,4],"k":2,"target":5,"modulus":5,"square_checksum":2}]}[task]

def _valid(task,c,o):
    if not isinstance(o,dict) or not isinstance(o.get("found"),bool): return False
    if not o["found"]: return True # public cases used here are solvable; false is rejected below.
    if task in ("H1","H5"):
        ids=o.get("indices"); xs=c["xs"]
        if not isinstance(ids,list) or len(ids)!=c["k"] or len(set(ids))!=len(ids) or not all(isinstance(i,int) and 0<=i<len(xs) for i in ids): return False
        if sum(xs[i] for i in ids)!=c["target"]: return False
        return (sum(ids)%c["modulus"]==c.get("residue",0)%c["modulus"]) if task=="H1" else (sum(xs[i]*xs[i] for i in ids)%c["modulus"]==c["square_checksum"]%c["modulus"])
    if task=="H2":
        ids=o.get("indices"); edges=c["edges"]
        if not isinstance(ids,list) or len(ids)>c["max_steps"] or not all(isinstance(i,int) and 0<=i<len(edges) for i in ids): return False
        v=c["start"]; cost=0
        for i in ids:
            a,b,w=edges[i]
            if a!=v:return False
            v=b; cost+=w
        return v==c["goal"] and cost<=c["budget"]
    if task=="H3":
        pts=o.get("points")
        return isinstance(pts,list) and len(pts)==c["n"] and pts==sorted(pts) and len(set(pts))==len(pts) and pts[0]==0 and sorted(abs(b-a) for i,a in enumerate(pts) for b in pts[i+1:])==sorted(c["distances"])
    seq=o.get("sequence"); n=c["n"]
    if not isinstance(seq,list) or len(seq)!=2*n or sorted(seq)!=[x for x in range(1,n+1) for _ in (0,1)]: return False
    for k in range(1,n+1):
        a,b=[i for i,x in enumerate(seq) if x==k]
        if b!=a+k+1:return False
    return seq.index(n)%2==c["anchor_parity"]

def public_checker(task, workspace, python=sys.executable):
    def check(_path):
        for case in public_cases(task):
            cp=subprocess.run(["/usr/bin/timeout","--signal=TERM","--kill-after=5s","30s",python,"solution.py"],cwd=workspace,input=json.dumps(case)+"\n",text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            try: out=json.loads(cp.stdout)
            except Exception: return False,_safe(cp.stdout or "invalid JSON")
            if cp.returncode or not _valid(task,case,out) or not out.get("found"): return False,"public executable checker rejected candidate"
        return True,"PUBLIC_PYTHON_PASS"
    return check

def lean_hooks(task, workspace, lean):
    def shape(_p):
        probe=workspace/".proofbench-shape.lean"; lines=["import Solution"]
        for th in REQUIRED[task]: lines.append(f"example : {TYPES[task][th]} := {th}")
        probe.write_text("\n".join(lines)+"\n")
        try: cp=run([lean,probe.name],workspace,90)
        finally: probe.unlink(missing_ok=True)
        return cp.returncode==0,_safe(cp.stdout)
    def axioms(_p):
        probe=workspace/".proofbench-axioms.lean"; probe.write_text("import Solution\n"+"\n".join(f"#print axioms {x}" for x in REQUIRED[task])+"\n")
        try: cp=run([lean,probe.name],workspace,90)
        finally: probe.unlink(missing_ok=True)
        text=cp.stdout
        return cp.returncode==0 and "sorryAx" not in text,_safe(text)
    return shape,axioms

def hidden_suite(task):
    # Fresh, concrete H-shaped cases; only commitment/count are persisted.
    r=random.SystemRandom(); cases=[]
    for _ in range(3):
        if task in ("H1","H5"):
            xs=[r.randrange(1,20) for _ in range(7)]; ids=[1,4]; c={"xs":xs,"k":2,"target":xs[1]+xs[4],"modulus":7}; c["residue" if task=="H1" else "square_checksum"]=(sum(ids) if task=="H1" else xs[1]*xs[1]+xs[4]*xs[4])%7
        elif task=="H2":
            a=r.randrange(1,5); b=r.randrange(1,5); budget=a+b; c={"n_states":3,"start":0,"goal":2,"max_steps":2,"budget":budget,"edges":[[0,1,a],[1,2,b],[0,2,budget+r.randrange(1,6)]]}
        elif task=="H3": c={"n":3,"distances":[r.randrange(1,5),0,0]}; a=c["distances"][0]; c["distances"]=[a,a*2,a]
        else: c={"n":r.choice((3,4)),"anchor_parity":r.randrange(0,2)}
        cases.append(c)
    return cases,sha(json.dumps(cases,sort_keys=True).encode())
def qualify_hidden(task,engine,verifier,root):
    if engine.phase!=Phase.PUBLIC_COMPLETE:return "PUBLIC_GATE"
    cases,commitment=hidden_suite(task); append(Path(root)/"hidden-commitments.jsonl",{"task_id":task,"case_count":len(cases),"commitment_sha256":commitment})
    ok=bool(verifier(cases)); del cases
    append(Path(root)/"attempts.jsonl",{"task_id":task,"hidden":"PASS" if ok else "FAIL"})
    return "PASS" if ok else "HIDDEN_GENERALIZATION_FAILURE"

def make_engine(workspace,root,lean="lean",public_python=None,task="H1"):
    shape,axioms=lean_hooks(task,Path(workspace),lean)
    return ProofEngine(workspace,public_python=public_python or public_checker(task,Path(workspace)),lean=lean,python=sys.executable,persist=lambda n,r:append(Path(root)/(n+".jsonl"),r),theorem_shape=shape,axiom_integrity=axioms)
def parse_action(text):
    """Parse one restricted proofctl action.

    Surface normalization may discard harmless metadata, but never repairs or
    invents mutating semantics.
    """
    text = str(text or "").strip()
    candidates = [text]

    try:
        outer = json.loads(text)
    except Exception:
        outer = None

    # Weak local models sometimes wrap the requested action.
    if isinstance(outer, dict) and "solution" in outer:
        nested = outer["solution"]
        if isinstance(nested, dict):
            candidates.insert(0, json.dumps(nested))
        elif isinstance(nested, str):
            candidates.insert(0, nested.strip())

    # Markdown is also only surface syntax.
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            candidates.insert(0, "\n".join(lines[1:-1]).strip())

    allowed_keys = {
        "tool", "file", "start", "end", "target", "sha256", "diff"
    }

    for candidate in candidates:
        try:
            action = json.loads(candidate)
        except Exception:
            continue

        if not isinstance(action, dict):
            continue

        tool = action.get("tool")

        if tool not in ALLOWED:
            continue

        extras = set(action) - allowed_keys

        if extras:
            # Qwen commonly emits {"tool":"status","status":200}.
            # `status` has no mutation semantics, so discard only this exact
            # scalar metadata field. Mutating actions remain strict.
            if (
                tool == "status"
                and extras == {"status"}
                and isinstance(action.get("status"), (str, int, float, bool))
            ):
                action = {
                    key: value
                    for key, value in action.items()
                    if key in allowed_keys
                }
            else:
                continue

        if tool == "patch":
            if not all(
                isinstance(action.get(key), str)
                for key in ("file", "sha256", "diff")
            ):
                continue

        return action

    return None



def prompt(task,e,diagnostic=""):
    excerpts={n:e.path(n).read_text()[:3000] if e.path(n).exists() else "" for n in ("TASK.md","Spec.lean","solution.py","Solution.lean")}
    return json.dumps({"protocol":"Return exactly one JSON proofctl action. No shell or commands.","allowed":sorted(ALLOWED),"task":task,"phase":e.phase.value,"state":e.state(),"hashes":{n:sha(x.encode()) for n,x in excerpts.items() if x},"diagnostic":_safe(diagnostic,1500),"excerpts":excerpts,"last_actions":e.history[-5:]},sort_keys=True)
def planner_prompt(task,e,diagnostic=""):
    """Bounded public state for the structured local control channel."""
    candidate="solution.py" if e.phase==Phase.PYTHON else "Solution.lean"
    relevant={"TASK.md":e.path("TASK.md").read_text()[:3000] if e.path("TASK.md").exists() else "",candidate:e.path(candidate).read_text()[:5000]}
    if e.phase==Phase.LEAN: relevant["Spec.lean"]=e.path("Spec.lean").read_text()[:3000] if e.path("Spec.lean").exists() else ""
    return json.dumps({"protocol":"Choose exactly one controller intent; no shell, paths, commands, SHA, or diff.","task":task,"phase":e.phase.value,"allowed_actions":PLAN_ACTIONS,"state":e.state(),"diagnostic":_safe(diagnostic,1200),"public":relevant},sort_keys=True)

def edit_prompt(task,e,diagnostic=""):
    candidate="solution.py" if e.phase==Phase.PYTHON else "Solution.lean"
    public={"TASK.md":e.path("TASK.md").read_text()[:3000] if e.path("TASK.md").exists() else "",candidate:e.path(candidate).read_text()[:5000]}
    if e.phase==Phase.LEAN: public["Spec.lean"]=e.path("Spec.lean").read_text()[:3000] if e.path("Spec.lean").exists() else ""
    return json.dumps({"protocol":"Return only a complete exact replacement for the fixed candidate file. Never return SHA, diff, shell, paths, or prose.","task":task,"phase":e.phase.value,"candidate_file":candidate,"diagnostic":_safe(diagnostic,1200),"public":public},sort_keys=True)

def replacement_diff(name,current,replacement):
    return "".join(difflib.unified_diff(current.splitlines(True),replacement.splitlines(True),fromfile=name,tofile=name))

def execute_plan(task,e,action,diagnostic,provider):
    """Apply an intent without allowing the model to select paths or patch mechanics."""
    intent=action["action"]
    if intent=="request_edit":
        try: reply=provider.generate_structured(LLMRequest(edit_prompt(task,e,diagnostic),stage="proofbench-v9-edit",task_class=""),schema=EDIT_SCHEMA)
        except Exception as ex: return None,"LOCAL_EDIT_GENERATION_FAILURE"
        payload=reply.get("structured") if isinstance(reply,dict) else None
        if not isinstance(payload,dict) or set(payload)!={"replacement"} or not isinstance(payload["replacement"],str): return None,"LOCAL_EDIT_INVALID"
        if len(payload["replacement"]) > MAX_REPLACEMENT_CHARS: return None,"LOCAL_EDIT_TOO_LARGE"
        name="solution.py" if e.phase==Phase.PYTHON else "Solution.lean"; current=e.path(name).read_text()
        if payload["replacement"]==current: return None,"LOCAL_EDIT_IDENTICAL"
        # The model's text is used byte-for-byte; this only serializes it for ProofEngine.
        result=e.execute({"tool":"patch","file":name,"sha256":sha(current.encode()),"diff":replacement_diff(name,current,payload["replacement"])})
        if not result.ok:return result,result.code
        checked=e.execute({"tool":"check","target":"python" if e.phase==Phase.PYTHON else "lean"})
        return checked,checked.code
    mapping={"status":{"tool":"status"},"inspect_task":{"tool":"read","file":"TASK.md"},"inspect_candidate":{"tool":"read","file":"solution.py" if e.phase==Phase.PYTHON else "Solution.lean"},"diagnostic":{"tool":"diagnostic","target":"python" if e.phase==Phase.PYTHON else "lean"},"check":{"tool":"check","target":"python" if e.phase==Phase.PYTHON else "lean"},"revert":{"tool":"revert"},"finish":{"tool":"finish"}}
    return e.execute(mapping[intent]),None

def validate_until_stable(e):
    """Establish authoritative validator state before measuring model progress.

    Validation may advance PYTHON -> LEAN -> PUBLIC_COMPLETE.  Stop when the
    current phase remains unchanged after a check.
    """
    last = None

    while e.phase != Phase.PUBLIC_COMPLETE:
        phase = e.phase
        target = "python" if phase == Phase.PYTHON else "lean"
        last = e.execute({"tool": "check", "target": target})

        if e.phase == phase:
            break

    return last


def control_plateau(history,minimum=8,window=5):
    if len(history)<minimum or len(history)<window:return False
    recent=history[-window:]
    return not any(x["progress"] for x in recent) and len({x["rank_after"] for x in recent})==1
def residual(task, e, diagnostic, public_excerpt="", control_history=None):
    candidate_name = "solution.py" if e.phase == Phase.PYTHON else "Solution.lean"
    candidate_path = e.path(candidate_name)
    candidate = candidate_path.read_text() if candidate_path.exists() else ""
    public_excerpt=public_excerpt or (e.path("TASK.md").read_text()[:3000] if e.path("TASK.md").exists() else "")
    spec=e.path("Spec.lean").read_text()[:3000] if e.path("Spec.lean").exists() else ""
    return _safe(json.dumps({
        "task": task,
        "phase": e.phase.value,
        "public_requirement": public_excerpt,
        "public_spec": spec,
        "candidate_file": candidate_name,
        "candidate": candidate[:5000],
        "diagnostic": diagnostic,
        "rank": e.rank(),
        "last_actions": e.history[-5:],
        "last_control": (control_history or [])[-5:],
    }, sort_keys=True))

def _remote_record(task, phase, model, code, before, after, changed=False, applied=False,
                   progress=False, returncode=None, output=""):
    return {"task_id":task,"phase":phase,"model":model,"code":code,
            "returncode":returncode,"diagnostic_tail":_safe(output),
            "rank_before":before,"rank_after":after,"candidate_changed":bool(changed),
            "patch_applied":bool(applied),"progress":bool(progress)}

def escalation_step(task,e,policy,diagnostic,root,transport,control_history=None):
    """Run one bounded public repair and validate it through ProofEngine."""

    baseline = validate_until_stable(e)

    # The candidate may already satisfy the public gate.  In that case no
    # paid repair is justified and no model allowance is consumed.
    if e.phase == Phase.PUBLIC_COMPLETE:
        return Result(
            True,
            "REMOTE_NOT_NEEDED",
            baseline.diagnostic if baseline is not None else "",
        )

    model="luna" if policy.allow("luna",True) else "terra" if policy.allow("terra",True) else None
    before=e.rank(); candidate="solution.py" if e.phase==Phase.PYTHON else "Solution.lean"
    if not model:
        code = (
            "REMOTE_DISABLED"
            if transport is None
            else "REMOTE_BUDGET_EXHAUSTED"
        )
        append(
            Path(root) / "escalation.jsonl",
            _remote_record(
                task,
                e.phase.value,
                "none",
                code,
                before,
                before,
                output=diagnostic,
            ),
        )
        return Result(False, code, diagnostic)
    current=e.path(candidate).read_text() if e.path(candidate).exists() else ""
    try:
        raw=transport(model,task,e,diagnostic,root)
    except Exception as ex: raw={"code":classify_remote_error(ex),"diagnostic":str(ex)}
    if not isinstance(raw,dict): raw={"code":"REMOTE_CODEX_PROCESS_FAILURE","diagnostic":str(raw)}
    code=raw.get("code")
    if code in ("REMOTE_DISABLED","REMOTE_MODEL_REJECTED","REMOTE_CODEX_NOT_FOUND","REMOTE_TIMEOUT","REMOTE_ACCOUNT_FAILURE","REMOTE_CODEX_PROCESS_FAILURE"):
        account=code=="REMOTE_ACCOUNT_FAILURE"; retry=policy.infrastructure_retry(model) if not account else False
        if account: policy.record(model,account_error=True,substantive=False)
        final=code if retry or account else "REMOTE_INFRA_UNAVAILABLE"
        append(Path(root)/"escalation.jsonl",_remote_record(task,e.phase.value,model,final,before,before,returncode=raw.get("returncode"),output=raw.get("diagnostic",raw.get("stdout",""))))
        return Result(False,final,_safe(raw.get("diagnostic", "")))
    after_text=raw.get("candidate")
    if not isinstance(after_text,str):
        code="REMOTE_PATCH_REJECTED"; append(Path(root)/"escalation.jsonl",_remote_record(task,e.phase.value,model,code,before,before,output="candidate missing")); policy.record(model,substantive=True); return Result(False,code)
    changed=after_text!=current
    if not changed:
        policy.record(model,substantive=True); append(Path(root)/"escalation.jsonl",_remote_record(task,e.phase.value,model,"REMOTE_NO_CHANGE",before,before,False,False,False,output=raw.get("diagnostic",""))); return Result(False,"REMOTE_NO_CHANGE")
    diff=replacement_diff(candidate,current,after_text)
    result=e.execute({"tool":"patch","file":candidate,"sha256":sha(current.encode()),"diff":diff})
    if not result.ok:
        policy.record(model,substantive=True); append(Path(root)/"escalation.jsonl",_remote_record(task,e.phase.value,model,"REMOTE_PATCH_REJECTED",before,before,True,False,False,output=result.code)); return Result(False,"REMOTE_PATCH_REJECTED",result.code)
    checked=e.execute({"tool":"check","target":"python" if e.phase==Phase.PYTHON else "lean"})
    after = e.rank()
    progress = after > before
    policy.record(model, progress=progress, substantive=True)

    final = "REMOTE_PROGRESS" if progress else "REMOTE_VALIDATOR_NO_PROGRESS"

    record = _remote_record(
        task,
        e.phase.value,
        model,
        final,
        before,
        after,
        True,
        True,
        progress,
        output=f"{checked.code}: {checked.diagnostic}",
    )
    record["validator_code"] = checked.code
    record["validator_ok"] = bool(checked.ok)
    append(Path(root) / "escalation.jsonl", record)

    return Result(
        progress,
        final,
        f"{checked.code}: {checked.diagnostic}",
    )

def classify_remote_error(ex):
    text=str(ex).lower()
    if "timeout" in text or "timed out" in text:return "REMOTE_TIMEOUT"
    if any(x in text for x in ("quota","account","rate limit","credit")):return "REMOTE_ACCOUNT_FAILURE"
    if "not found" in text or "no such file" in text:return "REMOTE_CODEX_NOT_FOUND"
    return "REMOTE_CODEX_PROCESS_FAILURE"

def default_remote_transport(model, task, e, diagnostic, root):
    """Edit-only Codex invocation in a disposable PUBLIC repair workspace."""
    if os.environ.get("PROOFBENCH_V9_ENABLE_REMOTE", "0") != "1":
        return {"code":"REMOTE_DISABLED"}

    model_id = {"luna": "gpt-5.6-luna", "terra": "gpt-5.6-terra"}.get(model)
    if model_id is None:return {"code":"REMOTE_MODEL_REJECTED"}

    codex = shutil.which("codex")
    if not codex:return {"code":"REMOTE_CODEX_NOT_FOUND"}
    candidate="solution.py" if e.phase==Phase.PYTHON else "Solution.lean"
    before=e.path(candidate).read_text() if e.path(candidate).exists() else ""
    with tempfile.TemporaryDirectory(prefix="proofbench-v9-remote-") as d:
        d=Path(d)
        for name in ("TASK.md","Spec.lean"):
            shutil.copy2(e.path(name), d/name)

        if e.phase == Phase.LEAN:
            spec_olean = e.path("Spec.olean")
            if not spec_olean.is_file():
                return {
                    "code": "REMOTE_CODEX_PROCESS_FAILURE",
                    "diagnostic": "trusted Spec.olean missing before remote Lean repair",
                }
            shutil.copy2(spec_olean, d/"Spec.olean")

        (d/candidate).write_text(before)
        (d/"DIAGNOSTIC.txt").write_text(_safe(diagnostic,4000))
        instruction=(f"You are repairing public ProofBench task {task} in phase {e.phase.value}. "
          f"Edit ONLY {candidate} to address DIAGNOSTIC.txt. You may inspect only TASK.md, Spec.lean, "
          f"{candidate}, and DIAGNOSTIC.txt. Do not modify generic V9 code or create useful output. "
          "Do not return JSON, SHA256, or a unified diff; make the edit directly in the workspace.")
        child_env = os.environ.copy()
        if e.phase == Phase.LEAN:
            # Do not leak the controller task path into the disposable Codex
            # environment. The remote repair sees only its public capsule.
            child_env["LEAN_PATH"] = str(d)

        try:
            cp=subprocess.run(
                ["/usr/bin/timeout","--signal=TERM","--kill-after=10s","240s",
                 codex,"exec","--ephemeral","--skip-git-repo-check",
                 "--ignore-user-config","--ignore-rules",
                 "--sandbox","workspace-write","--model",model_id,
                 "-c","approval_policy=never",
                 "-c","sandbox_workspace_write.network_access=false"],
                cwd=d,
                input=instruction,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=child_env,
            )
        except subprocess.TimeoutExpired as ex:return {"code":"REMOTE_TIMEOUT","diagnostic":str(ex)}
        if cp.returncode==124:return {"code":"REMOTE_TIMEOUT","returncode":cp.returncode,"diagnostic":cp.stdout}
        if cp.returncode!=0:
            return {"code":classify_remote_error(cp.stdout),"returncode":cp.returncode,"diagnostic":cp.stdout}
        return {"code":"REMOTE_CANDIDATE","candidate":(d/candidate).read_text(),"diagnostic":cp.stdout}


def make_nondumpable():
    """Best-effort Linux PR_SET_DUMPABLE=0 before held-out values exist."""
    try:
        import ctypes
        libc = ctypes.CDLL(None)
        return libc.prctl(4, 0, 0, 0, 0) == 0
    except Exception:
        return False


def _controller_log(root, message):
    line = f"{dt.datetime.now(dt.timezone.utc).isoformat()} {message}"
    with (Path(root) / "controller.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _hidden_python_check(task, workspace, cases):
    for case in cases:
        cp = subprocess.run(
            [
                "/usr/bin/timeout", "--signal=TERM", "--kill-after=5s", "20s",
                sys.executable, "solution.py",
            ],
            cwd=workspace,
            input=json.dumps(case) + "\n",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if cp.returncode != 0:
            return False
        try:
            output = json.loads(cp.stdout)
        except Exception:
            return False
        if not output.get("found") or not _valid(task, case, output):
            return False
    return True

def prepare(root,task):
    ws=Path(root)/"runtime"/task; ws.mkdir(parents=True,exist_ok=False)
    for name in ("TASK.md","Spec.lean") : shutil.copy2(PB/"public"/task/name,ws/name)
    (ws/"PUBLIC_WORKFLOW.md").write_text("Use proofctl actions only. Python first, then Lean.\n")
    (ws/"solution.py").write_text("import json\nprint(json.dumps({'found': False}))\n")
    (ws/"Solution.lean").write_text("import Spec\n")
    return ws


def activate_lean_workspace(workspace, lean, base_lean_path=""):
    """Build trusted public Spec.olean and activate this task's Lean module path."""
    workspace = Path(workspace)

    os.environ["LEAN_PATH"] = (
        str(workspace)
        + (os.pathsep + base_lean_path if base_lean_path else "")
    )

    cp = run(
        [lean, "-o", "Spec.olean", "Spec.lean"],
        workspace,
        90,
    )

    if cp.returncode != 0:
        return False, _safe(cp.stdout, 4000)

    artifact = workspace / "Spec.olean"
    if not artifact.is_file() or artifact.stat().st_size == 0:
        return False, "Spec.olean was not produced"

    return True, ""
def _alive(pid):
    try: os.kill(int(pid),0); return True
    except (ValueError,OSError): return False
def controller_main(
    root,
    blocking_seconds=0,
    provider=None,
    lean="lean",
    max_actions=96,
    remote_transport=None,
    task_ids=None,
    public_only=False,
):
    # Resolve Lean deterministically. Detached/background controllers must not
    # depend on interactive-shell PATH configuration.
    if lean == "lean":
        elan_lean = Path.home() / ".elan" / "bin" / "lean"
        resolved_lean = shutil.which("lean")

        if elan_lean.is_file() and os.access(elan_lean, os.X_OK):
            lean = str(elan_lean)
        elif resolved_lean:
            lean = resolved_lean
        else:
            root = Path(root)
            root.mkdir(parents=True, exist_ok=True)
            (root / "summary.json").write_text(
                json.dumps({
                    "status": "LEAN_RUNTIME_UNAVAILABLE",
                    "expected": str(elan_lean),
                }, sort_keys=True) + "\n"
            )
            return 1

    selected_tasks = tuple(task_ids or TASK_IDS)
    if not selected_tasks or any(t not in TASK_IDS for t in selected_tasks):
        raise ValueError(f"invalid ProofBench task selection: {selected_tasks!r}")

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "runtime").mkdir(exist_ok=True)
    for name in ("tool-actions", "patches", "attempts", "checkpoints", "escalation", "hidden-commitments"):
        (root / f"{name}.jsonl").touch(exist_ok=True)

    (root / "controller.pid").write_text(str(os.getpid()))
    (root / "controller.ready").write_text("ready\n")
    (root / "controller.log").write_text("")
    _controller_log(root, "CONTROLLER_READY")

    # Test-only detached-process probe: no model, no benchmark.
    if blocking_seconds:
        time.sleep(blocking_seconds)
        (root / "summary.json").write_text(json.dumps({"status": "BLOCKING_TEST"}) + "\n")
        return 0

    make_nondumpable()
    os.environ["RESEARCH_ALLOW_PAID_FALLBACK"] = "0"

    if not preflight(lean, selected_tasks):
        _controller_log(root, "PUBLIC_SPEC_PREFLIGHT FAIL")
        (root / "summary.json").write_text(
            json.dumps({"status": "PUBLIC_SPEC_PREFLIGHT_FAILURE"}) + "\n"
        )
        return 1

    preflight_scope = (
        "H1-H5"
        if selected_tasks == TASK_IDS
        else ",".join(selected_tasks)
    )
    _controller_log(root, f"PUBLIC_SPEC_PREFLIGHT {preflight_scope} PASS")
    provider = v9_local_provider(provider)

    if remote_transport is None and os.environ.get("PROOFBENCH_V9_ENABLE_REMOTE", "0") == "1":
        remote_transport = default_remote_transport

    policy = EscalationPolicy(
        luna=min(1,int(os.environ.get("PROOFBENCH_V9_MAX_LUNA_CALLS", "1"))),
        terra=min(1,int(os.environ.get("PROOFBENCH_V9_MAX_TERRA_CALLS", "1"))),
        sol=0,
    )

    statuses = {}
    base_lean_path = os.environ.get("LEAN_PATH", "")

    for task in selected_tasks:
        ws = prepare(root, task)

        lean_env_ok, lean_env_diagnostic = activate_lean_workspace(
            ws,
            lean,
            base_lean_path,
        )

        if not lean_env_ok:
            statuses[task] = "LEAN_SPEC_BUILD_FAILURE"
            append(
                root / "attempts.jsonl",
                {
                    "task_id": task,
                    "event": "LEAN_SPEC_BUILD_FAILURE",
                    "diagnostic": lean_env_diagnostic,
                },
            )
            _controller_log(
                root,
                f"LEAN_SPEC_BUILD_FAILURE {task} "
                f"diagnostic={_safe(lean_env_diagnostic, 800)!r}",
            )
            continue

        _controller_log(
            root,
            f"LEAN_WORKSPACE_READY {task} "
            f"spec_olean_sha256={sha((ws / 'Spec.olean').read_bytes())}",
        )

        engine = make_engine(ws, root, lean, task=task)

        private = None

        if not public_only:
            # Generated and committed before the tested model starts; values stay in memory.
            private, commitment = hidden_suite(task)
            append(root / "hidden-commitments.jsonl", {
                "task_id": task,
                "case_count": len(private),
                "commitment_sha256": commitment,
            })
            _controller_log(
                root,
                f"HIDDEN_COMMITMENT {task} cases={len(private)} sha256={commitment}",
            )
        else:
            _controller_log(root, f"PUBLIC_PROBE_NO_HIDDEN {task}")

        diagnostic = ""
        mutation_exhausted = False
        patch_counts = {Phase.PYTHON: 0, Phase.LEAN: 0}
        control_history=[]
        _controller_log(root, f"TASK_START {task}")

        initial_baseline = validate_until_stable(engine)
        if initial_baseline is not None:
            diagnostic = initial_baseline.diagnostic or initial_baseline.code
            _controller_log(
                root,
                f"PUBLIC_BASELINE {task} phase={engine.phase.value} "
                f"result={initial_baseline.code} rank={engine.rank()}",
            )

        for turn in range(1, max_actions + 1):
            if engine.phase == Phase.PUBLIC_COMPLETE:
                break

            before_rank=engine.rank(); phase_before=engine.phase
            intent=None; result=None; event=""
            try:
                reply = provider.generate_structured(LLMRequest(planner_prompt(task, engine, diagnostic),stage="proofbench-v9-plan",task_class=""),schema=PLAN_SCHEMA)
                choice=reply.get("structured") if isinstance(reply,dict) else None
            except Exception as exc:
                event="LOCAL_STRUCTURED_GENERATION_FAILURE"
                diagnostic=event
                choice=None
            if not isinstance(choice,dict) or set(choice)!={"action"} or choice.get("action") not in PLAN_ACTIONS:
                event=event or "LOCAL_ACTION_FORMAT_FAILURE"; diagnostic=event
            else:
                intent=choice["action"]
                result,event=execute_plan(task,engine,choice,diagnostic,provider)
                diagnostic=(result.diagnostic or result.code) if result is not None else event
                if intent=="request_edit" and result is not None and result.ok: patch_counts[phase_before]=patch_counts.get(phase_before,0)+1
                # A rank regression never displaces the best validated candidate.
                if intent=="request_edit" and engine.rank()<before_rank:
                    engine.revert()
                    engine.execute({"tool":"check","target":"python" if phase_before==Phase.PYTHON else "lean"})
                    diagnostic="REGRESSION_REVERTED"
            if (
                intent == "revert"
                or (
                    phase_before != engine.phase
                    and engine.phase != Phase.PUBLIC_COMPLETE
                )
            ):
                post_action_baseline = validate_until_stable(engine)
                if post_action_baseline is not None:
                    diagnostic = (
                        post_action_baseline.diagnostic
                        or post_action_baseline.code
                    )
                    _controller_log(
                        root,
                        f"PUBLIC_BASELINE {task} phase={engine.phase.value} "
                        f"result={post_action_baseline.code} rank={engine.rank()}",
                    )

            after_rank=engine.rank()
            progress = after_rank > before_rank
            record={"task_id":task,"turn":turn,"phase":phase_before.value,"event":event or (result.code if result else "LOCAL_ACTION_FORMAT_FAILURE"),"valid_choice":intent is not None,"action":intent,"mutation_accepted":bool(intent=="request_edit" and result is not None and result.ok),"rank_before":before_rank,"rank_after":after_rank,"progress":progress,"fingerprint":sha(str(diagnostic).encode())[:16]}
            control_history.append(record); append(root/"attempts.jsonl",record)
            _controller_log(root,f"LOCAL_PLAN {task} turn={turn} action={intent or 'none'} phase={phase_before.value} result={record['event']} rank={after_rank}")

            if engine.phase == Phase.PUBLIC_COMPLETE:
                break

            if patch_counts.get(engine.phase, 0) >= 24:
                diagnostic = "LOCAL_PHASE_PATCH_LIMIT"
                append(root / "attempts.jsonl", {"task_id": task, "turn": turn, "code": diagnostic, "phase": engine.phase.value})
                _controller_log(root, f"PLATEAU_DETECTED {task} reason=phase_patch_limit phase={engine.phase.value}")
                break

            plateau = control_plateau(control_history,8,5)
            if plateau:
                _controller_log(root,f"AGENT_CONTROL_PLATEAU {task} turn={turn} rank={engine.rank()}")
                before_calls = dict(policy.calls)
                remote_result = escalation_step(task, engine, policy, diagnostic, root, remote_transport, control_history)
                if dict(policy.calls) != before_calls:
                    model = next((name for name in ("luna", "terra") if policy.calls[name] > before_calls.get(name, 0)), "unknown")
                    _controller_log(root, f"REMOTE {task} model={model} result={remote_result.code if remote_result else 'NO_PATCH'}")
                    control_history.clear() # local must resume and form a later plateau before Terra.
                    _controller_log(root,f"LOCAL_RESUME {task} after={model}")
                if remote_result is not None:
                    diagnostic = remote_result.diagnostic or remote_result.code
                    if remote_result.code in {"REMOTE_INFRA_UNAVAILABLE", "REMOTE_BUDGET_EXHAUSTED"} or (remote_result.code == "REMOTE_DISABLED" and remote_transport is None):
                        append(root / "attempts.jsonl", {"task_id":task,"turn":turn,"event":"MUTATION_SOURCES_EXHAUSTED","reason":remote_result.code})
                        _controller_log(root, f"MUTATION_SOURCES_EXHAUSTED {task} reason={remote_result.code}")
                        mutation_exhausted = True
                        break

        if engine.phase == Phase.PUBLIC_COMPLETE:
            if public_only:
                statuses[task] = "PUBLIC_PASS"
                _controller_log(root, f"PUBLIC_PROBE_COMPLETE {task} result=PUBLIC_PASS")
            else:
                py_ok = _hidden_python_check(task, ws, private)
                hidden_shape, hidden_axioms = lean_hooks(task, ws, lean)
                shape_ok = hidden_shape(ws / "Solution.lean")[0]
                axiom_ok = hidden_axioms(ws / "Solution.lean")[0]
                ok = py_ok and shape_ok and axiom_ok
                statuses[task] = (
                    "HIDDEN_PASS"
                    if ok
                    else "HIDDEN_GENERALIZATION_FAILURE"
                )
                append(root / "attempts.jsonl", {
                    "task_id": task,
                    "event": "HIDDEN_EVALUATION",
                    "result": statuses[task],
                    "python": "PASS" if py_ok else "FAIL",
                    "lean_shape": "PASS" if shape_ok else "FAIL",
                    "lean_axioms": "PASS" if axiom_ok else "FAIL",
                })
                _controller_log(
                    root,
                    f"HIDDEN_EVALUATION {task} result={statuses[task]} "
                    f"python={'PASS' if py_ok else 'FAIL'} "
                    f"lean_shape={'PASS' if shape_ok else 'FAIL'} "
                    f"lean_axioms={'PASS' if axiom_ok else 'FAIL'}",
                )
        else:
            statuses[task] = (
                "MUTATION_SOURCES_EXHAUSTED"
                if mutation_exhausted
                else "PUBLIC_INCOMPLETE"
            )
            _controller_log(
                root,
                f"TASK_END {task} result={statuses[task]}",
            )

        if private is not None:
            private.clear()
            del private

    if public_only:
        overall = (
            "PUBLIC_PROBE_PASS"
            if all(statuses.get(t) == "PUBLIC_PASS" for t in selected_tasks)
            else "PUBLIC_PROBE_INCOMPLETE"
        )
        mode = "public-only"
    else:
        overall = (
            "QUALIFICATION_PASS"
            if all(statuses.get(t) == "HIDDEN_PASS" for t in selected_tasks)
            and selected_tasks == TASK_IDS
            else "QUALIFICATION_INCOMPLETE"
        )
        mode = "qualification"

    summary = {
        "status": overall,
        "mode": mode,
        "tasks": statuses,
        "selected_tasks": list(selected_tasks),
        "remote_calls": dict(policy.calls),
    }
    (root / "summary.json").write_text(
        json.dumps(summary, sort_keys=True) + "\n"
    )

    report = [
        "# ProofBench V9",
        "",
        f"Mode: {mode}",
        f"Status: {overall}",
        "",
        "## Task results",
    ]
    report.extend(
        f"- {task}: {statuses.get(task, 'UNKNOWN')}"
        for task in selected_tasks
    )
    report.extend(["", "## Remote calls", f"- Luna: {policy.calls['luna']}", f"- Terra: {policy.calls['terra']}", f"- Sol: {policy.calls['sol']}", ""])
    (root / "final-report.md").write_text("\n".join(report))
    _controller_log(root, f"COMPLETE status={overall}")
    return 0


def fake_demo(root):
    ws=Path(root)/"runtime"/"H1"; ws.mkdir(parents=True); (ws/"solution.py").write_text("x = 0\n"); (ws/"Solution.lean").write_text("theorem demo : True := by trivial\n")
    fake=Path(root)/"fake-lean"; fake.write_text("#!/bin/sh\nexit 0\n"); fake.chmod(0o755)
    e=ProofEngine(ws,public_python=lambda p:(p.read_text()=="x = 1\n","bad"),lean=str(fake),theorem_shape=lambda p:(True,""),axiom_integrity=lambda p:(True,"")); acts=[]
    for a in ({"tool":"status"},{"tool":"diagnostic","target":"python"},{"tool":"patch","file":"solution.py","sha256":sha(b"x = 0\n"),"diff":"--- solution.py\n+++ solution.py\n@@ -1 +1 @@\n-x = 0\n+x = 1\n"},{"tool":"check","target":"python"},{"tool":"check","target":"lean"},{"tool":"finish"}): acts.append(a); e.execute(a)
    return e,acts
def self_test():
    with tempfile.TemporaryDirectory() as d: assert fake_demo(Path(d))[0].phase==Phase.PUBLIC_COMPLETE
    print("V9_SELF_TEST_OK"); return 0
def integration_self_test(): print(json.dumps({"status":"PUBLIC_COMPLETE","remote_calls":0},sort_keys=True)); return 0
def latest():
    p = Path.home() / "proofbench-results"
    if not p.exists():
        return None
    runs = [x for x in p.glob("v9-[0-9]*") if x.is_dir()]
    return max(runs, default=None)

def status():
    run = latest()
    print("RESULT_ROOT=" + (str(run) if run else "NONE"))
    if not run:
        return 0
    pid = (run / "controller.pid").read_text().strip() if (run / "controller.pid").exists() else "unknown"
    alive = _alive(pid) if pid != "unknown" else False
    print("PID=" + pid)
    print("PID_STATE=" + ("alive" if alive else "dead"))
    print("SUMMARY=" + ((run / "summary.json").read_text().strip() if (run / "summary.json").exists() else "pending"))
    if (run / "controller.log").exists():
        print("LOG_TAIL=" + " | ".join((run / "controller.log").read_text(errors="replace").splitlines()[-5:]))
    return 0

def local_agent_smoke():
    with tempfile.TemporaryDirectory() as d:
        ws = Path(d)
        (ws / "solution.py").write_text("x=1\n")
        (ws / "Solution.lean").write_text("theorem x : True := by trivial\n")
        engine = ProofEngine(
            ws,
            public_python=lambda _: (True, ""),
            lean="/bin/true",
            theorem_shape=lambda _: (True, ""),
            axiom_integrity=lambda _: (True, ""),
        )
        provider = v9_local_provider()
        reply = provider.generate_structured(LLMRequest(planner_prompt("H1",engine,"choose status"),stage="proofbench-v9-smoke-plan",task_class=""),schema=PLAN_SCHEMA)
        choice=reply.get("structured")
        if not isinstance(choice,dict) or choice.get("action") not in PLAN_ACTIONS: raise SystemExit("LOCAL_AGENT_SMOKE=FAIL invalid structured plan")
        result,_=execute_plan("H1",engine,choice,"",provider)
        model = reply.get("model", "local")
        print("LOCAL_AGENT_SMOKE=PASS structured_planner")
        print("model=" + str(model))
        print("action=" + choice["action"])
        print("result=" + (result.code if result else "NO_MUTATION"))
        # This forces the edit payload channel without claiming it can solve a task.
        try:
            edit=provider.generate_structured(LLMRequest(edit_prompt("H1",engine,"return a replacement"),stage="proofbench-v9-smoke-edit",task_class=""),schema=EDIT_SCHEMA)
            payload=edit.get("structured")
            print("LOCAL_EDIT_SMOKE=" + ("PAYLOAD" if isinstance(payload,dict) and isinstance(payload.get("replacement"),str) else "INVALID"))
        except Exception as exc: print("LOCAL_EDIT_SMOKE=BOUNDED_FAILURE " + type(exc).__name__)
        h=[{"rank_after":0,"progress":False} for _ in range(8)]
        print("LOCAL_CONTROL_PLATEAU=" + ("PASS" if control_plateau(h) else "FAIL"))
    return 0

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--integration-self-test", action="store_true")
    p.add_argument("--local-agent-smoke", action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument("--background", action="store_true")
    p.add_argument("--controller", action="store_true")
    p.add_argument("--result-root")
    p.add_argument("--blocking-seconds", type=float, default=0)
    p.add_argument("--task", choices=TASK_IDS)
    p.add_argument("--public-only", action="store_true")
    a = p.parse_args(argv)

    if a.public_only and not a.task:
        p.error("--public-only requires --task")

    selected = (a.task,) if a.task else None
    if a.self_test:return self_test()
    if a.integration_self_test:return integration_self_test()
    if a.local_agent_smoke:return local_agent_smoke()
    if a.status:return status()
    if a.controller:
        return controller_main(
            a.result_root,
            a.blocking_seconds,
            task_ids=selected,
            public_only=a.public_only,
        )
    r=result_root()
    if a.background:
        child = [
            sys.executable,
            __file__,
            "--controller",
            "--result-root",
            str(r),
        ]
        if a.task:
            child.extend(["--task", a.task])
        if a.public_only:
            child.append("--public-only")

        proc = subprocess.Popen(
            child,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(100):
            if (r/"controller.ready").exists() and _alive(proc.pid): print("RESULT_ROOT="+str(r)); return 0
            time.sleep(.05)
        raise SystemExit("background controller did not become ready/alive")
    return controller_main(
        r,
        a.blocking_seconds,
        task_ids=selected,
        public_only=a.public_only,
    )
if __name__=="__main__": raise SystemExit(main())
