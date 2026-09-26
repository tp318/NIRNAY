"""Reference solve with HiGHS (through SciPy) — used ONLY as an external baseline.

NIRNAY does not link or call any third-party solver; this script runs HiGHS in a
separate Python process on the same MPS file for comparison and cross-validation.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

sys.path.insert(0, os.path.dirname(__file__))
from mpsio import read_mps  # noqa: E402


def solve_highs(path: str, time_limit: float = 600.0, mip_gap: float = 1e-4) -> dict:
    t0 = time.perf_counter()
    mdl = read_mps(path)
    t_read = time.perf_counter() - t0
    if mdl.Q is not None and mdl.Q.nnz:
        return dict(status="SKIPPED_QP", objective=None, time=None)
    sgn = -1.0 if mdl.maximize else 1.0
    c = sgn * mdl.c
    cons = [LinearConstraint(mdl.A, mdl.rl, mdl.ru)] if mdl.m else []
    t1 = time.perf_counter()
    res = milp(c, constraints=cons, integrality=mdl.integrality, bounds=Bounds(mdl.lb, mdl.ub),
               options=dict(time_limit=time_limit, mip_rel_gap=mip_gap, disp=False))
    t_solve = time.perf_counter() - t1
    out = dict(status={0: "OPTIMAL", 1: "LIMIT", 2: "INFEASIBLE", 3: "UNBOUNDED", 4: "OTHER"}.get(res.status, str(res.status)),
               message=res.message, time=t_solve, time_read=t_read)
    if res.x is not None:
        out["objective"] = float(mdl.c @ res.x + mdl.offset)
    else:
        out["objective"] = None
    if getattr(res, "mip_dual_bound", None) is not None and mdl.integrality.any():
        db = res.mip_dual_bound
        out["bound"] = float(sgn * db + mdl.offset) if db is not None and math.isfinite(db) else None
        out["mip_gap"] = res.mip_gap
        out["nodes"] = res.mip_node_count
    return out


if __name__ == "__main__":
    print(json.dumps(solve_highs(sys.argv[1]), indent=1, default=str))
