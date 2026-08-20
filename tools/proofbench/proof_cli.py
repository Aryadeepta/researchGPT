"""Human/test CLI facade; never exposes a shell tool."""
from __future__ import annotations
import argparse, hashlib, json
from .proof_engine import ProofEngine
def main(argv=None, engine=None):
 p=argparse.ArgumentParser(prog="proofctl"); p.add_argument("--workspace",default="."); s=p.add_subparsers(dest="cmd",required=True)
 s.add_parser("status"); r=s.add_parser("read"); r.add_argument("file"); r.add_argument("range",nargs="?")
 for n in ("diagnostic","check"): x=s.add_parser(n); x.add_argument("target",choices=("python","lean","public"))
 x=s.add_parser("patch"); x.add_argument("file"); x.add_argument("sha256"); x.add_argument("diff")
 s.add_parser("diff"); s.add_parser("revert"); s.add_parser("finish"); a=p.parse_args(argv); e=engine or ProofEngine(a.workspace)
 if a.cmd=="read": start,end=(map(int,a.range.split(":")) if a.range else (None,None)); q={"tool":"read","file":a.file,"start":start,"end":end}
 elif a.cmd=="patch": q={"tool":"patch","file":a.file,"sha256":a.sha256,"diff":a.diff}
 elif a.cmd in ("check","diagnostic"): q={"tool":a.cmd,"target":a.target}
 elif a.cmd=="diff": q={"tool":"status"}
 else:q={"tool":a.cmd}
 z=e.execute(q); print(json.dumps({"ok":z.ok,"code":z.code,"diagnostic":z.diagnostic,"checkpoint":z.checkpoint})); return 0 if z.ok else 2
if __name__=="__main__": raise SystemExit(main())
