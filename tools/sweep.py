"""Benchmark sweep over a directory of MPS/QPS files, with HiGHS (or published) references.

    python tools/sweep.py instances/bench/netlib --time-limit 5 --configs default pdhg-gpu
    python tools/sweep.py instances/bench/netlib --configs "pdhg-gpu:--pdhg-tol 1e-8" --tag netlib_1e8

Each config is NAME or NAME:EXTRA-ARGS. Built-in names:
  default   nirnay solve FILE                      (auto: presolve + IPM, simplex fallback)
  simplex   --method simplex
  ipm       --method ipm
  pdhg-gpu  --method pdhg --device gpu
  pdhg-cpu  --method pdhg --device cpu

Reference objective: instances' published value if the directory has EXPECTED.txt, else HiGHS
(via SciPy, cached in <dir>/highs_ref.json). A run "agrees" if it is OPTIMAL and its objective is
within --agree (default 1e-6 relative; PDHG configs use their own tolerance x 10).
Writes results/sweep_<tag>.json and prints a summary.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXE = os.environ.get("NIRNAY_EXE") or os.path.join(ROOT, "bin", "nirnay.exe" if os.name == "nt" else "nirnay")
BUILTIN = {
    "default": [],
    "simplex": ["--method", "simplex"],
    "ipm": ["--method", "ipm"],
    "pdhg-gpu": ["--method", "pdhg", "--device", "gpu"],
    "pdhg-cpu": ["--method", "pdhg", "--device", "cpu"],
}


def highs_refs(d, files, tl):
    cache = os.path.join(d, "highs_ref.json")
    refs = json.load(open(cache)) if os.path.exists(cache) else {}
    todo = [f for f in files if os.path.basename(f) not in refs]
    for f in todo:
        code = ("import json,sys; sys.path.insert(0, r'%s'); from baseline import solve_highs; "
                "print(json.dumps(solve_highs(r'%s', %f, 1e-9), default=str))" % (HERE, f, tl))
        try:
            cp = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=tl + 60)
            r = json.loads(cp.stdout.strip().splitlines()[-1])
        except Exception as e:  # noqa: BLE001
            r = {"status": "ERROR", "message": str(e)[:200]}
        refs[os.path.basename(f)] = {"status": r.get("status"), "objective": r.get("objective"), "time": r.get("time")}
        json.dump(refs, open(cache, "w"), indent=1)
    return refs


def run(f, extra, tl):
    out = os.path.join(ROOT, "results", "_sweep.json")
    if os.path.exists(out):
        os.remove(out)
    t0 = time.perf_counter()
    try:
        cp = subprocess.run([EXE, "solve", f, "--quiet", "--json", out, "--time-limit", str(tl)] + extra,
                            capture_output=True, text=True, timeout=tl * 3 + 30)
    except subprocess.TimeoutExpired:
        return {"status": "HARD_TIMEOUT", "wall": time.perf_counter() - t0}
    wall = time.perf_counter() - t0
    if not os.path.exists(out):
        return {"status": "CRASH", "wall": wall, "err": (cp.stderr or cp.stdout)[-300:]}
    r = json.load(open(out))
    return {"status": r["status"], "objective": r["objective"], "time": r["time_total"], "wall": wall,
            "iterations": r["iterations"], "method": r["method"],
            "kkt": [r["kkt_primal_inf"], r["kkt_dual_inf"], r["kkt_rel_gap"]] if r["kkt_available"] else None,
            "viol": max(r["max_row_violation"], r["max_bound_violation"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--time-limit", type=float, default=5)
    ap.add_argument("--configs", nargs="+", default=["default", "pdhg-gpu"])
    ap.add_argument("--agree", type=float, default=1e-6)
    ap.add_argument("--tag")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    # case-insensitive filesystems return the same file for *.mps and *.MPS: de-duplicate
    files = sorted({os.path.normcase(f): f for pat in ("*.mps", "*.MPS", "*.qps", "*.QPS")
                    for f in glob.glob(os.path.join(args.dir, pat))}.values())
    if args.only:
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0].lower() in {o.lower() for o in args.only}]
    exp = {}
    ep = os.path.join(args.dir, "EXPECTED.txt")
    if os.path.exists(ep):
        for line in open(ep):
            if line.strip() and not line.startswith("#"):
                a, b = line.split()[:2]
                exp[a.lower()] = float(b)
    meta = os.path.join(args.dir, "metadata.json")
    if os.path.exists(meta):
        for k, v in json.load(open(meta)).items():
            if v.get("OPT") is not None:
                exp[k.lower()] = float(v["OPT"])
                exp[k.lower().replace("-", "").replace("_", "")] = float(v["OPT"])
    refs = {} if exp else highs_refs(args.dir, files, 60)
    tag = args.tag or os.path.basename(os.path.normpath(args.dir))
    rows = []
    for f in files:
        base = os.path.basename(f)
        name = os.path.splitext(base)[0]
        key = name.lower() if name.lower() in exp else name.lower().replace("-", "").replace("_", "")
        if key in exp:
            ref, rsrc = exp[key], "published"
        else:
            h = refs.get(base, {})
            ref, rsrc = (h.get("objective"), "HiGHS") if h.get("status") == "OPTIMAL" else (None, "HiGHS:" + str(h.get("status")))
        row = {"instance": name, "reference": ref, "ref_source": rsrc, "highs_time": refs.get(base, {}).get("time")}
        line = f"{name:12s} ref {ref if ref is not None else float('nan'):+.8e} |"
        for cfg in args.configs:
            cname, _, extra = cfg.partition(":")
            r = run(f, BUILTIN.get(cname, []) + extra.split(), args.time_limit)
            tol = args.agree
            if cname.startswith("pdhg"):
                m = re.search(r"--pdhg-tol\s+(\S+)", extra)
                tol = 10 * float(m.group(1)) if m else 1e-3
            err = abs(r["objective"] - ref) / max(1.0, abs(ref)) if (ref is not None and r.get("objective") is not None) else None
            r["relerr"] = err
            r["agree"] = r["status"] == "OPTIMAL" and err is not None and err <= tol
            row[cfg] = r
            line += f" {cfg[:14]:14s} {str(r['status'])[:10]:10s} {'OK ' if r['agree'] else '-- '} " \
                    f"{(err if err is not None else float('nan')):7.1e} {r.get('time') or r.get('wall') or 0:7.3f}s |"
        rows.append(row)
        print(line, flush=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(rows, open(os.path.join(ROOT, "results", f"sweep_{tag}.json"), "w"), indent=1)
    print(f"\n{len(rows)} instances ({args.time_limit:g} s limit)")
    for cfg in args.configs:
        rs = [r[cfg] for r in rows]
        opt = sum(1 for r in rs if r["status"] == "OPTIMAL")
        agr = sum(1 for r in rs if r["agree"])
        wrong = [row["instance"] for row in rows if row[cfg]["status"] == "OPTIMAL" and not row[cfg]["agree"] and row["reference"] is not None]
        print(f"  {cfg:28s} optimal {opt:3d}/{len(rs)}   agree with reference {agr:3d}   optimal-but-disagree: {wrong}")


if __name__ == "__main__":
    main()
