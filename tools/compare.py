"""Validate NIRNAY against IBM ILOG CPLEX and HiGHS on the bundled instances.

    python tools/compare.py [--time-limit 120] [--gap 1e-4]

Solvers (each in its own process, same MPS file, same time limit and MIP gap):
  * NIRNAY   bin/nirnay.exe, default settings (--method auto)
  * CPLEX    cplex.exe interactive optimizer (CPLEX Studio Community: <=1000 rows and <=1000 cols)
  * HiGHS    via SciPy (LP/MILP only; SciPy's HiGHS interface has no QP)

Reference objective per instance:
  * Netlib: the published optimal value (instances/netlib/EXPECTED.txt)
  * otherwise: the value on which CPLEX and HiGHS agree (rel. 1e-6), else whichever
    of them proved optimality.
Writes results/compare.json and results/compare.csv.
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NIRNAY = os.path.join(ROOT, "bin", "nirnay.exe" if os.name == "nt" else "nirnay")
CPLEX = next(iter(glob.glob(r"C:\Program Files\IBM\ILOG\CPLEX_Studio*\cplex\bin\x64_win64\cplex.exe")), "cplex")


def run_nirnay(path, tl, gap):
    out = os.path.join(ROOT, "results", "_cmp.json")
    if os.path.exists(out):
        os.remove(out)
    try:
        subprocess.run([NIRNAY, "solve", path, "--quiet", "--json", out, "--time-limit", str(tl), "--gap", str(gap)],
                       capture_output=True, text=True, timeout=tl + 60)
    except subprocess.TimeoutExpired:
        return dict(status="TIMEOUT")
    if not os.path.exists(out):
        return dict(status="CRASH")
    r = json.load(open(out))
    return dict(status=r["status"], obj=r["objective"], time=r["time_total"], method=r["method"],
                kkt=max(r["kkt_primal_inf"], r["kkt_dual_inf"], r["kkt_rel_gap"]) if r["kkt_available"] else None,
                viol=max(r["max_row_violation"], r["max_bound_violation"], r["max_int_violation"]))


def run_cplex(path, tl, gap):
    cmds = [f"read {path}", f"set timelimit {tl}", f"set mip tolerances mipgap {gap}", "set threads 0", "optimize", "quit"]
    try:
        cp = subprocess.run([CPLEX, "-c", *cmds], capture_output=True, text=True, timeout=tl + 60)
    except subprocess.TimeoutExpired:
        return dict(status="TIMEOUT")
    o = cp.stdout
    if "Problem size limits exceeded" in o or "1016" in o:
        return dict(status="SIZE_LIMIT")
    m = re.search(r"^(.*?)\s-\s(.*?):\s+Objective\s*=\s*([-+0-9.eE]+)", o, re.M)
    t = re.search(r"Solution time\s*=\s*([0-9.]+)", o)
    if not m:
        st = re.search(r"^(.*? - .*?)\.?\s*$", o[o.rfind("optimize"):], re.M)
        return dict(status="NO_SOLUTION", note=(st.group(1) if st else o[-200:]).strip())
    desc = m.group(2).lower()
    status = "OPTIMAL" if ("optimal" in desc and "limit" not in desc) else ("TIME_LIMIT" if "limit" in desc else desc.upper())
    return dict(status=status, obj=float(m.group(3)), time=float(t.group(1)) if t else None, method=m.group(1).strip())


def run_highs(path, tl, gap):
    code = ("import json,sys; sys.path.insert(0, r'%s'); from baseline import solve_highs; "
            "print(json.dumps(solve_highs(r'%s', %f, %g), default=str))" % (HERE, path, tl, gap))
    try:
        cp = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=tl + 60)
        r = json.loads(cp.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return dict(status="ERROR")
    return dict(status=r.get("status"), obj=r.get("objective"), time=r.get("time"))


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=120)
    ap.add_argument("--gap", type=float, default=1e-4)
    args = ap.parse_args()

    items = []
    ref_netlib = {}
    for line in open(os.path.join(ROOT, "instances", "netlib", "EXPECTED.txt")):
        if line.strip() and not line.startswith("#"):
            n, v = line.split()[:2]
            ref_netlib[n] = float(v)
            items.append((n, "netlib", "LP", os.path.join(ROOT, "instances", "netlib", n + ".mps")))
    man = json.load(open(os.path.join(ROOT, "instances", "manifest.json")))
    for n, m in man.items():
        if m["category"] != "gpu":
            items.append((n, m["category"], m["cls"], os.path.join(ROOT, m["path"])))

    rows = []
    for name, cat, cls, path in items:
        res = {"NIRNAY": run_nirnay(path, args.time_limit, args.gap),
               "CPLEX": run_cplex(path, args.time_limit, args.gap),
               "HiGHS": run_highs(path, args.time_limit, args.gap) if cls != "QP" else dict(status="N/A (no QP)")}
        if name in ref_netlib:
            ref, src = ref_netlib[name], "Netlib published"
        else:
            c, h = res["CPLEX"], res["HiGHS"]
            if c.get("status") == "OPTIMAL" and h.get("status") == "OPTIMAL" and rel(c["obj"], h["obj"]) <= 1e-6:
                ref, src = c["obj"], "CPLEX = HiGHS"
            elif c.get("status") == "OPTIMAL":
                ref, src = c["obj"], "CPLEX"
            elif h.get("status") == "OPTIMAL":
                ref, src = h["obj"], "HiGHS"
            else:
                ref, src = None, "none"
        row = dict(instance=name, category=cat, cls=cls, reference=ref, ref_source=src)
        for s, r in res.items():
            row[f"{s}_status"] = r.get("status")
            row[f"{s}_obj"] = r.get("obj")
            row[f"{s}_time"] = r.get("time")
            row[f"{s}_relerr"] = rel(r["obj"], ref) if (ref is not None and r.get("obj") is not None) else None
        row["NIRNAY_method"] = res["NIRNAY"].get("method")
        row["NIRNAY_kkt"] = res["NIRNAY"].get("kkt")
        row["NIRNAY_viol"] = res["NIRNAY"].get("viol")
        rows.append(row)
        f = lambda v, s: "-" if v is None else format(v, s)
        print(f"{name:20s} {cls:4s} ref {f(ref, '+.10e'):>18s} | " + " | ".join(
            f"{s} {str(row[s + '_status'])[:10]:10s} err {f(row[s + '_relerr'], '.1e'):>7s} t {f(row[s + '_time'], '.3f'):>7s}"
            for s in res), flush=True)

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(rows, open(os.path.join(ROOT, "results", "compare.json"), "w"), indent=1)
    with open(os.path.join(ROOT, "results", "compare.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
