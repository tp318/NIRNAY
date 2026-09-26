"""NIRNAY benchmark harness.

Runs every instance in instances/manifest.json (or a directory of .mps/.mps.gz
files) through NIRNAY and, as an external baseline, HiGHS (via SciPy, separate
process). Writes results/benchmark.{csv,json} and results/report.html.

    python tools/bench.py                       # base suite
    python tools/bench.py --suite gpu           # large LPs: PDHG CPU vs GPU
    python tools/bench.py --dir path/to/mps     # any directory (e.g. MIPLIB / Netlib files)

Nothing here is tuned per instance; all runs use default NIRNAY options unless
stated in the 'config' column.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
import math
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXE = os.path.join(ROOT, "bin", "nirnay.exe" if os.name == "nt" else "nirnay")


def run_nirnay(path, args, timeout):
    out_json = os.path.join(ROOT, "results", "_last.json")
    if os.path.exists(out_json):
        os.remove(out_json)
    cmd = [EXE, path, "--quiet", "--json", out_json] + args
    t0 = time.perf_counter()
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return dict(status="TIMEOUT", wall=timeout)
    wall = time.perf_counter() - t0
    if not os.path.exists(out_json):
        return dict(status="CRASH", wall=wall, stderr=cp.stderr[-400:])
    with open(out_json) as f:
        r = json.load(f)
    r["wall"] = wall
    return r


def run_highs(path, timeout, gap):
    cmd = [sys.executable, "-c",
           "import json,sys; sys.path.insert(0, r'%s'); from baseline import solve_highs; "
           "print(json.dumps(solve_highs(r'%s', %f, %g), default=str))" % (HERE, path, timeout, gap)]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 60)
        return json.loads(cp.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        return dict(status="TIMEOUT", time=timeout)
    except Exception as e:  # noqa: BLE001
        return dict(status="ERROR", message=str(e))


def rel_diff(a, b):
    if a is None or b is None:
        return None
    return abs(a - b) / max(1.0, abs(b))


def sgm(values, shift=1.0):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return math.exp(sum(math.log(v + shift) for v in vals) / len(vals)) - shift


def configs_for(cls, suite):
    if suite == "gpu":
        return [("pdhg-gpu", ["--method", "pdhg", "--device", "gpu"]),
                ("pdhg-cpu", ["--method", "pdhg", "--device", "cpu"])]
    if cls == "LP":
        return [("ipm", ["--method", "ipm"]), ("dual-simplex", ["--method", "simplex"])]
    if cls == "QP":
        return [("ipm", ["--method", "ipm", "--presolve", "off"])]
    return [("branch-and-cut", [])]


def collect(args):
    items = []
    if args.dir:
        for fn in sorted(os.listdir(args.dir)):
            if fn.endswith(".mps") or fn.endswith(".mps.gz") or fn.endswith(".qps"):
                p = os.path.join(args.dir, fn)
                if fn.endswith(".gz"):
                    tmp = os.path.join(ROOT, "build", fn[:-3])
                    with gzip.open(p, "rb") as fi, open(tmp, "wb") as fo:
                        shutil.copyfileobj(fi, fo)
                    p = tmp
                items.append(dict(name=fn.split(".")[0], path=p, category="external", cls=args.cls,
                                  description=f"external file {fn}"))
        return items
    with open(os.path.join(ROOT, "instances", "manifest.json")) as f:
        man = json.load(f)
    for name, m in man.items():
        if args.suite == "gpu" and m["category"] != "gpu":
            continue
        if args.suite == "base" and m["category"] == "gpu":
            continue
        items.append(dict(name=name, path=os.path.join(ROOT, m["path"]), **{k: v for k, v in m.items() if k != "path"}))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="base", choices=["base", "gpu", "all"])
    ap.add_argument("--dir")
    ap.add_argument("--cls", default="MILP", help="problem class for --dir files (LP/MILP/QP)")
    ap.add_argument("--time-limit", type=float, default=300)
    ap.add_argument("--gap", type=float, default=1e-4)
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or ("external" if args.dir else args.suite)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    rows = []
    for it in collect(args):
        base = None if args.no_baseline else run_highs(it["path"], args.time_limit, args.gap)
        for cfg, extra in configs_for(it["cls"], "gpu" if it["category"] == "gpu" else "base"):
            r = run_nirnay(it["path"], extra + ["--time-limit", str(args.time_limit), "--gap", str(args.gap)],
                           args.time_limit + 60)
            row = dict(instance=it["name"], category=it["category"], cls=it["cls"], config=cfg,
                       rows=r.get("rows", it.get("rows")), cols=r.get("cols", it.get("cols")),
                       nnz=r.get("nnz", it.get("nnz")), integers=r.get("int_cols", it.get("integers")),
                       status=r.get("status"), objective=r.get("objective"), bound=r.get("bound"),
                       mip_gap=r.get("mip_gap"), time=r.get("time_total"), wall=r.get("wall"),
                       iterations=r.get("iterations"), nodes=r.get("nodes"), cuts=r.get("cuts"),
                       kkt_primal=r.get("kkt_primal_inf") if r.get("kkt_available") else None,
                       kkt_dual=r.get("kkt_dual_inf") if r.get("kkt_available") else None,
                       kkt_gap=r.get("kkt_rel_gap") if r.get("kkt_available") else None,
                       max_viol=max(r.get("max_row_violation") or 0, r.get("max_bound_violation") or 0),
                       note=r.get("note", ""), description=it.get("description", ""))
            if base:
                row.update(highs_status=base.get("status"), highs_objective=base.get("objective"),
                           highs_time=base.get("time"))
                row["obj_rel_diff"] = rel_diff(row["objective"], base.get("objective")) if row["status"] in (
                    "OPTIMAL", "FEASIBLE") else None
                if row["time"] is not None and base.get("time"):
                    row["time_ratio"] = row["time"] / max(base["time"], 1e-3)
            rows.append(row)
            ht = row.get("highs_time")
            print(f"{it['name']:24s} {cfg:15s} {str(row['status']):16s} obj={row['objective']!s:>22s} "
                  f"t={row['time'] if row['time'] is not None else float('nan'):8.3f}s | HiGHS "
                  f"{row.get('highs_objective')!s:>22s} t={ht if ht is not None else float('nan'):8.3f}s "
                  f"| rel.diff {row.get('obj_rel_diff')}", flush=True)
    out = os.path.join(ROOT, "results", f"benchmark_{tag}")
    with open(out + ".json", "w") as f:
        json.dump(rows, f, indent=1, default=str)
    keys = sorted({k for r in rows for k in r})
    with open(out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    write_report(tag, rows)
    print("wrote", out + ".csv", "and results/report_" + tag + ".html")


def fmt(v, spec=".3g"):
    if v is None or v == "":
        return "–"
    if isinstance(v, float):
        return format(v, spec)
    return html.escape(str(v))


def write_report(tag, rows):
    agree = [r for r in rows if r.get("obj_rel_diff") is not None]
    ok = sum(1 for r in agree if r["obj_rel_diff"] <= 1e-6 or (r["config"].startswith("pdhg") and r["obj_rel_diff"] <= 1e-3)
             or (r["cls"] == "MILP" and r["obj_rel_diff"] <= 1e-4))
    solved = sum(1 for r in rows if r["status"] == "OPTIMAL")
    by_cfg = {}
    for r in rows:
        by_cfg.setdefault(r["config"], []).append(r)
    summary = []
    for cfg, rs in by_cfg.items():
        both = [r for r in rs if r.get("time") is not None and r.get("highs_time") is not None and r["status"] == "OPTIMAL"]
        summary.append((cfg, len(rs), sum(1 for r in rs if r["status"] == "OPTIMAL"),
                        sgm([r["time"] for r in both]), sgm([r["highs_time"] for r in both])))
    head = ["instance", "class", "config", "rows", "cols", "int", "status", "NIRNAY obj", "HiGHS obj", "rel.diff",
            "NIRNAY s", "HiGHS s", "iters", "nodes", "KKT p/d/gap", "note"]
    trs = []
    for r in rows:
        kkt = "–" if r.get("kkt_primal") is None else f"{r['kkt_primal']:.0e} / {r['kkt_dual']:.0e} / {r['kkt_gap']:.0e}"
        diff = r.get("obj_rel_diff")
        cls = "good" if diff is not None and diff <= 1e-6 else ("warn" if diff is not None and diff <= 1e-3 else ("bad" if diff is not None else ""))
        tr = r.get("time_ratio")
        tcls = "" if tr is None else ("good" if tr <= 1 else ("warn" if tr <= 10 else "bad"))
        trs.append("<tr>" + "".join([
            f"<td>{fmt(r['instance'])}</td>", f"<td>{fmt(r['cls'])}</td>", f"<td>{fmt(r['config'])}</td>",
            f"<td class=n>{fmt(r['rows'])}</td>", f"<td class=n>{fmt(r['cols'])}</td>", f"<td class=n>{fmt(r['integers'])}</td>",
            f"<td>{fmt(r['status'])}</td>", f"<td class=n>{fmt(r['objective'], '.10g')}</td>",
            f"<td class=n>{fmt(r.get('highs_objective'), '.10g')}</td>", f"<td class='n {cls}'>{fmt(diff, '.1e')}</td>",
            f"<td class='n {tcls}'>{fmt(r['time'], '.3f')}</td>", f"<td class=n>{fmt(r.get('highs_time'), '.3f')}</td>",
            f"<td class=n>{fmt(r['iterations'])}</td>", f"<td class=n>{fmt(r['nodes'])}</td>", f"<td class=n>{kkt}</td>",
            f"<td class=note>{fmt(r.get('note'))}</td>"]) + "</tr>")
    srows = "".join(f"<tr><td>{c}</td><td class=n>{n}</td><td class=n>{s}</td><td class=n>{fmt(a, '.3f')}</td><td class=n>{fmt(b, '.3f')}</td></tr>"
                    for c, n, s, a, b in summary)
    page = f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>NIRNAY Benchmark</title><style>
:root{{--bg:#fbfbf9;--fg:#1d1d1b;--mut:#6b6b66;--line:#e3e2dc;--good:#1f7a4d;--warn:#9a6700;--bad:#b42318;--card:#fff}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#151514;--fg:#ecebe6;--mut:#a3a29c;--line:#34332f;--good:#5fc48f;--warn:#e0b44c;--bad:#f07167;--card:#1d1d1b}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,Segoe UI,sans-serif;margin:0;padding:24px 16px}}
main{{max-width:1400px;margin:auto}} h1{{font-size:22px;margin:0 0 4px}} p{{color:var(--mut);margin:4px 0 16px}}
.wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--card)}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}} th,td{{padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap;text-align:left}}
th{{font-weight:600;color:var(--mut);font-size:12px}} td.n{{text-align:right}} td.note{{white-space:normal;min-width:240px;color:var(--mut);font-size:12px}}
.good{{color:var(--good)}} .warn{{color:var(--warn)}} .bad{{color:var(--bad)}} .kpi{{display:flex;gap:24px;flex-wrap:wrap;margin:12px 0 20px}}
.kpi div{{border:1px solid var(--line);border-radius:8px;padding:10px 14px;background:var(--card)}} .kpi b{{font-size:20px;display:block}}
</style></head><body><main>
<h1>NIRNAY benchmark — {html.escape(tag)}</h1>
<p>Generated {time.strftime('%Y-%m-%d %H:%M')}. Baseline: HiGHS via SciPy {html.escape(scipy_version())}, run in a separate process on the same MPS file. Times are solver-reported (excluding file reading). Machine: single laptop; numbers are indicative, not a formal benchmark.</p>
<div class=kpi><div><b>{len(rows)}</b>runs</div><div><b>{solved}</b>reported optimal</div><div><b>{ok}/{len(agree)}</b>objective agrees with HiGHS</div></div>
<h2 style="font-size:16px">Summary by configuration (shifted geometric mean, shift 1 s, over runs both solved)</h2>
<div class=wrap><table><tr><th>config</th><th>runs</th><th>optimal</th><th>NIRNAY SGM s</th><th>HiGHS SGM s</th></tr>{srows}</table></div>
<h2 style="font-size:16px">All runs</h2>
<div class=wrap><table><tr>{''.join(f'<th>{h}</th>' for h in head)}</tr>{''.join(trs)}</table></div>
<p>rel.diff = |obj<sub>NIRNAY</sub> − obj<sub>HiGHS</sub>| / max(1, |obj<sub>HiGHS</sub>|). KKT = relative primal infeasibility / dual infeasibility / duality gap evaluated by NIRNAY on the original model (LP/QP). PDHG is a first-order method run to 1e-4 relative accuracy, so objective differences near 1e-4 are expected.</p>
</main></body></html>"""
    with open(os.path.join(ROOT, "results", f"report_{tag}.html"), "w", encoding="utf-8") as f:
        f.write(page)


def scipy_version():
    try:
        import scipy
        return scipy.__version__
    except Exception:  # noqa: BLE001
        return "?"


if __name__ == "__main__":
    main()
