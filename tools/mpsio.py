"""Minimal MPS/QPS reader + writer used by the NIRNAY benchmark harness.

The harness uses its own reader so the baseline solver (HiGHS via SciPy) is fed
exactly the same model NIRNAY reads, without linking any solver code into NIRNAY.
"""
from __future__ import annotations

import gzip
import math
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

INF = math.inf
NL = chr(10)


@dataclass
class LPModel:
    name: str = "model"
    c: np.ndarray = None           # (n,)
    A: sp.csr_matrix = None        # (m, n)
    rl: np.ndarray = None          # row lower
    ru: np.ndarray = None          # row upper
    lb: np.ndarray = None
    ub: np.ndarray = None
    integrality: np.ndarray = None  # 1 = integer
    Q: sp.csc_matrix | None = None  # full symmetric, objective 0.5 x'Qx
    offset: float = 0.0
    maximize: bool = False
    col_names: list = field(default_factory=list)
    row_names: list = field(default_factory=list)

    @property
    def m(self):
        return self.A.shape[0]

    @property
    def n(self):
        return self.A.shape[1]


def _num(s):
    v = float(s)
    if v >= 1e30:
        return INF
    if v <= -1e30:
        return -INF
    return v


def read_mps(path: str) -> LPModel:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        lines = f.readlines()
    sec = None
    obj_row = None
    rows, rtype = {}, []
    free_rows = set()
    cols = {}
    col_names, row_names = [], []
    c, lb, ub, integ = [], [], [], []
    ti, tj, tv = [], [], []
    qi, qj, qv = [], [], []
    rhs, rng, has_rng = [], [], []
    in_int = False
    maximize = False
    offset = 0.0
    name = "model"
    lo_set = []
    quad_full = False

    def col(nm):
        if nm not in cols:
            cols[nm] = len(col_names)
            col_names.append(nm)
            c.append(0.0); lb.append(0.0); ub.append(INF); integ.append(1 if in_int else 0); lo_set.append(False)
        return cols[nm]

    for raw in lines:
        if not raw.strip() or raw.startswith("*"):
            continue
        header = not raw[0].isspace()
        tok = raw.split()
        if header:
            h = tok[0]
            if h == "NAME":
                name = tok[1] if len(tok) > 1 else name
                sec = None
            elif h in ("OBJSENSE", "OBJSENCE"):
                sec = "OBJSENSE"
                if len(tok) > 1:
                    maximize = tok[1].startswith("MAX"); sec = None
            elif h in ("ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS"):
                sec = h
            elif h in ("QUADOBJ", "QSECTION"):
                sec = "QUAD"; quad_full = False
            elif h == "QMATRIX":
                sec = "QUAD"; quad_full = True
            elif h == "ENDATA":
                break
            elif h.startswith("MAX"):
                maximize = True
            continue
        if sec == "OBJSENSE":
            maximize = tok[0].startswith("MAX")
        elif sec == "ROWS":
            t, nm = tok[0].upper(), tok[1]
            if t == "N":
                if obj_row is None:
                    obj_row = nm
                else:
                    free_rows.add(nm)
            else:
                rows[nm] = len(rtype); rtype.append(t); row_names.append(nm)
                rhs.append(0.0); rng.append(0.0); has_rng.append(False)
        elif sec == "COLUMNS":
            if len(tok) >= 3 and tok[1] == "'MARKER'":
                if tok[2] == "'INTORG'":
                    in_int = True
                elif tok[2] == "'INTEND'":
                    in_int = False
                continue
            j = col(tok[0])
            for k in range(1, len(tok) - 1, 2):
                r, v = tok[k], _num(tok[k + 1])
                if r == obj_row:
                    c[j] += v
                elif r in free_rows:
                    continue
                else:
                    ti.append(rows[r]); tj.append(j); tv.append(v)
        elif sec in ("RHS", "RANGES"):
            start = 1 if len(tok) % 2 == 1 else 0
            for k in range(start, len(tok) - 1, 2):
                r, v = tok[k], _num(tok[k + 1])
                if r == obj_row:
                    if sec == "RHS":
                        offset = -v
                    continue
                if r in free_rows:
                    continue
                i = rows[r]
                if sec == "RHS":
                    rhs[i] = v
                else:
                    rng[i] = v; has_rng[i] = True
        elif sec == "BOUNDS":
            bt = tok[0]
            needs = bt not in ("FR", "MI", "PL", "BV")
            if needs:
                cn, v = (tok[2], _num(tok[3])) if len(tok) >= 4 else (tok[1], _num(tok[2]))
            else:
                cn, v = (tok[2] if len(tok) >= 3 else tok[1]), 0.0
            j = cols[cn]
            if bt == "UP":
                ub[j] = v
                if v < 0 and lb[j] == 0.0 and not lo_set[j]:
                    lb[j] = -INF
            elif bt == "LO":
                lb[j] = v; lo_set[j] = True
            elif bt == "FX":
                lb[j] = v; ub[j] = v; lo_set[j] = True
            elif bt == "FR":
                lb[j] = -INF; ub[j] = INF; lo_set[j] = True
            elif bt == "MI":
                lb[j] = -INF; lo_set[j] = True
            elif bt == "PL":
                ub[j] = INF
            elif bt == "BV":
                lb[j] = 0.0; ub[j] = 1.0; integ[j] = 1; lo_set[j] = True
            elif bt == "LI":
                lb[j] = v; integ[j] = 1; lo_set[j] = True
            elif bt == "UI":
                ub[j] = v; integ[j] = 1
                if v < 0 and lb[j] == 0.0 and not lo_set[j]:
                    lb[j] = -INF
            else:
                raise ValueError(f"unsupported bound {bt}")
        elif sec == "QUAD":
            a, b, v = cols[tok[0]], cols[tok[1]], _num(tok[2])
            qi.append(a); qj.append(b); qv.append(v)
            if not quad_full and a != b:
                qi.append(b); qj.append(a); qv.append(v)

    m, n = len(rtype), len(col_names)
    A = sp.csr_matrix((tv, (ti, tj)), shape=(m, n))
    A.sum_duplicates()
    rl = np.empty(m); ru = np.empty(m)
    for i, t in enumerate(rtype):
        r, R = rhs[i], rng[i]
        if t == "E":
            if not has_rng[i]:
                rl[i] = ru[i] = r
            elif R >= 0:
                rl[i], ru[i] = r, r + R
            else:
                rl[i], ru[i] = r + R, r
        elif t == "L":
            ru[i] = r; rl[i] = r - abs(R) if has_rng[i] else -INF
        else:
            rl[i] = r; ru[i] = r + abs(R) if has_rng[i] else INF
    Q = sp.csc_matrix((qv, (qi, qj)), shape=(n, n)) if qv else None
    return LPModel(name, np.array(c, float), A, rl, ru, np.array(lb, float), np.array(ub, float),
                   np.array(integ, int), Q, offset, maximize, col_names, row_names)


def _fmt(v):
    return repr(float(v))


def write_mps(model: LPModel, path: str):
    A = sp.csc_matrix(model.A)
    m, n = A.shape
    cn = model.col_names or [f"x{j}" for j in range(n)]
    rn = model.row_names or [f"r{i}" for i in range(m)]
    f = open(path, "w", buffering=1 << 20)

    class _Out:
        def append(self, line):
            f.write(line + NL)

        def __iadd__(self, lines):
            for line in lines:
                self.append(line)
            return self

    out = _Out()
    out.append(f"NAME {model.name}")
    if model.maximize:
        out += ["OBJSENSE", "    MAX"]
    out += ["ROWS", " N  OBJ"]
    for i in range(m):
        lo, up = model.rl[i], model.ru[i]
        t = "E" if lo == up else ("G" if math.isfinite(lo) else "L")
        out.append(f" {t}  {rn[i]}")
    out.append("COLUMNS")
    in_int = False
    mk = 0
    integ = model.integrality if model.integrality is not None else np.zeros(n, int)
    for j in range(n):
        isint = bool(integ[j])
        if isint != in_int:
            out.append(f"    M{mk}  'MARKER'  '{'INTORG' if isint else 'INTEND'}'"); mk += 1
            in_int = isint
        wrote = False
        if model.c[j] != 0:
            out.append(f"    {cn[j]}  OBJ  {_fmt(model.c[j])}"); wrote = True
        for p in range(A.indptr[j], A.indptr[j + 1]):
            out.append(f"    {cn[j]}  {rn[A.indices[p]]}  {_fmt(A.data[p])}"); wrote = True
        if not wrote:
            out.append(f"    {cn[j]}  OBJ  0")
    if in_int:
        out.append(f"    M{mk}  'MARKER'  'INTEND'")
    out.append("RHS")
    if model.offset:
        out.append(f"    RHS  OBJ  {_fmt(-model.offset)}")
    for i in range(m):
        lo, up = model.rl[i], model.ru[i]
        r = lo if (lo == up or math.isfinite(lo)) else up
        if r != 0 and math.isfinite(r):
            out.append(f"    RHS  {rn[i]}  {_fmt(r)}")
    out.append("RANGES")
    for i in range(m):
        lo, up = model.rl[i], model.ru[i]
        if lo != up and math.isfinite(lo) and math.isfinite(up):
            out.append(f"    RNG  {rn[i]}  {_fmt(up - lo)}")
    out.append("BOUNDS")
    for j in range(n):
        lo, up = model.lb[j], model.ub[j]
        if lo == up:
            out.append(f" FX BND  {cn[j]}  {_fmt(lo)}"); continue
        if not math.isfinite(lo) and not math.isfinite(up):
            out.append(f" FR BND  {cn[j]}"); continue
        if not math.isfinite(lo):
            out.append(f" MI BND  {cn[j]}")
        elif lo != 0:
            out.append(f" LO BND  {cn[j]}  {_fmt(lo)}")
        if math.isfinite(up):
            out.append(f" UP BND  {cn[j]}  {_fmt(up)}")
        elif integ[j]:
            out.append(f" PL BND  {cn[j]}")
    if model.Q is not None and model.Q.nnz:
        out.append("QUADOBJ")
        Q = sp.csc_matrix(model.Q)
        for j in range(n):
            for p in range(Q.indptr[j], Q.indptr[j + 1]):
                i = Q.indices[p]
                if i >= j and Q.data[p] != 0:
                    out.append(f"    {cn[j]}  {cn[i]}  {_fmt(Q.data[p])}")
    out.append("ENDATA")
    f.close()
