"""Generate the NIRNAY test/benchmark instance suite.

All instances are synthetic and reproducible (fixed seeds). The refinery and
power instances are *reconstructions of standard open-literature formulations*
(linear blending-index gasoline blending, multi-period refinery planning with
unit operating modes, crude-oil unloading & tank scheduling with dedicated
tanks, unit commitment); the data are realistic in magnitude but invented.
They are NOT MRPL data.

    python tools/generate.py            # writes instances/*/*.mps + instances/manifest.json
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(__file__))
from mpsio import LPModel, write_mps  # noqa: E402

INF = math.inf
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "instances")


class Builder:
    """Small row-wise model builder."""

    def __init__(self, name):
        self.name = name
        self.c, self.lb, self.ub, self.integ, self.cn = [], [], [], [], []
        self.ri, self.rj, self.rv = [], [], []
        self.rl, self.ru, self.rn = [], [], []
        self.q = {}
        self.offset = 0.0
        self.maximize = False

    def var(self, name, lb=0.0, ub=INF, cost=0.0, integer=False):
        self.cn.append(name); self.c.append(cost); self.lb.append(lb); self.ub.append(ub)
        self.integ.append(1 if integer else 0)
        return len(self.cn) - 1

    def row(self, name, coefs, lo=-INF, up=INF):
        i = len(self.rn)
        self.rn.append(name); self.rl.append(lo); self.ru.append(up)
        for j, v in coefs:
            if v != 0:
                self.ri.append(i); self.rj.append(j); self.rv.append(float(v))
        return i

    def quad(self, i, j, v):
        """add v * x_i x_j to 0.5 x'Qx (i.e. Q_ij += v, Q_ji += v for i != j; Q_ii += v)."""
        if i == j:
            self.q[(i, i)] = self.q.get((i, i), 0.0) + v
        else:
            self.q[(i, j)] = self.q.get((i, j), 0.0) + v
            self.q[(j, i)] = self.q.get((j, i), 0.0) + v

    def build(self):
        m, n = len(self.rn), len(self.cn)
        A = sp.csr_matrix((self.rv, (self.ri, self.rj)), shape=(m, n))
        A.sum_duplicates()
        Q = None
        if self.q:
            keys = list(self.q.keys())
            Q = sp.csc_matrix(([self.q[k] for k in keys], ([k[0] for k in keys], [k[1] for k in keys])), shape=(n, n))
        return LPModel(self.name, np.array(self.c, float), A, np.array(self.rl, float), np.array(self.ru, float),
                       np.array(self.lb, float), np.array(self.ub, float), np.array(self.integ, int), Q,
                       self.offset, self.maximize, self.cn, self.rn)


# --------------------------------------------------------------------------- LP
def transport(ns, nd, seed):
    rng = np.random.default_rng(seed)
    b = Builder(f"transport_{ns}x{nd}")
    supply = rng.integers(50, 150, ns).astype(float)
    demand = rng.dirichlet(np.ones(nd)) * supply.sum()
    demand = np.floor(demand); demand[-1] += supply.sum() - demand.sum()  # balanced -> degenerate
    cost = rng.integers(1, 30, (ns, nd))
    x = [[b.var(f"x_{i}_{j}", cost=float(cost[i, j])) for j in range(nd)] for i in range(ns)]
    for i in range(ns):
        b.row(f"sup_{i}", [(x[i][j], 1) for j in range(nd)], up=supply[i])
    for j in range(nd):
        b.row(f"dem_{j}", [(x[i][j], 1) for i in range(ns)], lo=demand[j])
    return b.build()


def assignment(n, seed):
    rng = np.random.default_rng(seed)
    b = Builder(f"assign_{n}")
    cost = rng.integers(1, 100, (n, n))
    x = [[b.var(f"x_{i}_{j}", cost=float(cost[i, j])) for j in range(n)] for i in range(n)]
    for i in range(n):
        b.row(f"a_{i}", [(x[i][j], 1) for j in range(n)], 1, 1)
    for j in range(n):
        b.row(f"t_{j}", [(x[i][j], 1) for i in range(n)], 1, 1)
    return b.build()


def multicommodity(gw, gh, K, seed, name=None):
    """Min-cost multicommodity flow on a directed grid with shared arc capacities."""
    rng = np.random.default_rng(seed)
    nodes = gw * gh
    arcs = []
    for yy in range(gh):
        for xx in range(gw):
            u = yy * gw + xx
            if xx + 1 < gw:
                arcs += [(u, u + 1), (u + 1, u)]
            if yy + 1 < gh:
                arcs += [(u, u + gw), (u + gw, u)]
    E = len(arcs)
    cost = rng.uniform(1, 10, E)
    cap = rng.uniform(20, 60, E)
    src = rng.integers(0, nodes, K)
    dst = rng.integers(0, nodes, K)
    dst = np.where(dst == src, (dst + nodes // 2) % nodes, dst)
    dem = rng.uniform(5, 15, K)
    n = E * K
    # variables f[k,e] index k*E + e
    rows, cols, vals = [], [], []
    rl, ru = [], []
    r = 0
    tail = np.array([a[0] for a in arcs]); head = np.array([a[1] for a in arcs])
    for k in range(K):
        # flow conservation: out - in = supply
        base = k * E
        rows += list(r + tail); cols += list(base + np.arange(E)); vals += [1.0] * E
        rows += list(r + head); cols += list(base + np.arange(E)); vals += [-1.0] * E
        rhs = np.zeros(nodes); rhs[src[k]] = dem[k]; rhs[dst[k]] = -dem[k]
        rl += list(rhs); ru += list(rhs)
        r += nodes
    for e in range(E):
        rows += [r + e] * K; cols += [k * E + e for k in range(K)]; vals += [1.0] * K
    rl += [-INF] * E; ru += list(cap)
    r += E
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r, n))
    c = np.tile(cost, K)
    # slack "unmet demand" is not modelled: capacities are generous enough to stay feasible
    return LPModel(name or f"mcf_{gw}x{gh}_k{K}", c, A, np.array(rl), np.array(ru), np.zeros(n), np.full(n, INF),
                   np.zeros(n, int), None, 0.0, False, [f"f{j}" for j in range(n)], [f"r{i}" for i in range(r)])


def multicommodity_stream(gw, gh, K, seed, path):
    """Same model as multicommodity() but written column-by-column straight to disk
    (constant memory) for the very large GPU instances."""
    rng = np.random.default_rng(seed)
    nodes = gw * gh
    arcs = []
    for yy in range(gh):
        for xx in range(gw):
            u = yy * gw + xx
            if xx + 1 < gw:
                arcs += [(u, u + 1), (u + 1, u)]
            if yy + 1 < gh:
                arcs += [(u, u + gw), (u + gw, u)]
    E = len(arcs)
    cost = rng.uniform(1, 10, E)
    cap = rng.uniform(20, 60, E)
    src = rng.integers(0, nodes, K)
    dst = rng.integers(0, nodes, K)
    dst = np.where(dst == src, (dst + nodes // 2) % nodes, dst)
    dem = rng.uniform(5, 15, K)
    nl = chr(10)
    with open(path, "w", buffering=1 << 22) as f:
        f.write(f"NAME mcf_{gw}x{gh}_k{K}{nl}ROWS{nl} N  OBJ{nl}")
        for k in range(K):
            for v in range(nodes):
                f.write(f" E  n{k}_{v}{nl}")
        for e in range(E):
            f.write(f" L  c{e}{nl}")
        f.write("COLUMNS" + nl)
        for k in range(K):
            for e, (a, b) in enumerate(arcs):
                nm = f"f{k}_{e}"
                f.write(f"    {nm}  OBJ  {float(cost[e])!r}  n{k}_{a}  1{nl}    {nm}  n{k}_{b}  -1  c{e}  1{nl}")
        f.write("RHS" + nl)
        for k in range(K):
            f.write(f"    RHS  n{k}_{src[k]}  {float(dem[k])!r}{nl}    RHS  n{k}_{dst[k]}  {-float(dem[k])!r}{nl}")
        for e in range(E):
            f.write(f"    RHS  c{e}  {float(cap[e])!r}{nl}")
        f.write("ENDATA" + nl)
    return dict(rows=K * nodes + E, cols=K * E, nnz=3 * K * E, integers=0)


def ill_conditioned(m, n, seed):
    """Sparse LP made ill-conditioned: rows/cols scaled over 8 orders of magnitude and
    near-parallel row pairs (a_i and a_i + 1e-7 * noise). Feasible and bounded by construction."""
    rng = np.random.default_rng(seed)
    dens = min(1.0, 6.0 / n)
    A = sp.random(m, n, density=dens, random_state=seed, data_rvs=lambda k: rng.uniform(-1, 1, k)).tocsr()
    # ensure each column has an entry
    A = A + sp.csr_matrix((rng.uniform(0.5, 1, n), (rng.integers(0, m, n), np.arange(n))), shape=(m, n))
    A = A.tolil()
    for i in range(0, m - 1, 5):  # near-parallel pairs
        A[i + 1, :] = A[i, :] * 1.0
        nz = A[i + 1, :].nonzero()[1]
        if len(nz):
            j = nz[0]
            A[i + 1, j] = A[i + 1, j] * (1 + 1e-7)
    A = A.tocsr()
    R = 10.0 ** rng.uniform(-4, 4, m)
    C = 10.0 ** rng.uniform(-3, 3, n)
    A = sp.diags(R) @ A @ sp.diags(C)
    x0 = rng.uniform(0, 1, n) / C
    ax = A @ x0
    slack = np.abs(ax) * 0.05 + R * 1e-3
    rl = ax - slack
    ru = ax + slack
    typ = rng.integers(0, 3, m)
    rl = np.where(typ == 1, -INF, rl)
    ru = np.where(typ == 2, INF, ru)
    lb = np.zeros(n)
    ub = 2.0 / C
    c = rng.normal(0, 1, n) / C * 10.0 ** rng.uniform(-2, 2, n)
    return LPModel(f"illcond_{m}x{n}", c, sp.csr_matrix(A), rl, ru, lb, ub, np.zeros(n, int), None, 0.0, False,
                   [f"x{j}" for j in range(n)], [f"r{i}" for i in range(m)])


def degenerate_lp(m, n, seed):
    """Highly primal-degenerate LP: a known vertex x0 at which most rows are active with zero slack."""
    rng = np.random.default_rng(seed)
    A = sp.random(m, n, density=min(1.0, 8.0 / n), random_state=seed + 1,
                  data_rvs=lambda k: rng.integers(-5, 6, k).astype(float)).tocsr()
    A.eliminate_zeros()
    x0 = np.where(rng.random(n) < 0.7, 0.0, rng.integers(1, 4, n).astype(float))
    ax = A @ x0
    rl = ax.copy()                      # every row active at x0 -> massive degeneracy
    ru = np.full(m, INF)
    c = np.abs(rng.integers(-3, 10, n)).astype(float) + 1.0
    ub = np.full(n, 10.0)
    return LPModel(f"degen_{m}x{n}", c, A, rl, ru, np.zeros(n), ub, np.zeros(n, int), None, 0.0, False,
                   [f"x{j}" for j in range(n)], [f"r{i}" for i in range(m)])


# ------------------------------------------------------------------------- MILP
def multiknapsack(n, m, seed):
    rng = np.random.default_rng(seed)
    b = Builder(f"mknap_{n}x{m}")
    b.maximize = True
    w = rng.integers(10, 100, (m, n))
    p = (w.sum(0) / m + rng.integers(0, 20, n)).astype(float)
    x = [b.var(f"x{j}", 0, 1, p[j], True) for j in range(n)]
    for i in range(m):
        b.row(f"cap{i}", [(x[j], w[i, j]) for j in range(n)], up=float(0.5 * w[i].sum()))
    return b.build()


def facility_location(nf, nc, seed):
    """Capacitated facility location, *aggregated* (weak) big-M linking: sum_j x_ij <= cap_i y_i."""
    rng = np.random.default_rng(seed)
    b = Builder(f"cfl_{nf}x{nc}")
    fx = rng.uniform(0, 100, (nf, 2)); cx = rng.uniform(0, 100, (nc, 2))
    dem = rng.integers(5, 35, nc).astype(float)
    cap = rng.integers(80, 220, nf).astype(float)
    while cap.sum() < 1.5 * dem.sum():
        cap *= 1.2
    fixed = rng.uniform(300, 700, nf)
    y = [b.var(f"open_{i}", 0, 1, float(fixed[i]), True) for i in range(nf)]
    x = [[b.var(f"x_{i}_{j}", 0, INF, float(np.linalg.norm(fx[i] - cx[j]) * dem[j] / 10)) for j in range(nc)]
         for i in range(nf)]
    # x_ij = fraction of demand j served by i
    for j in range(nc):
        b.row(f"dem_{j}", [(x[i][j], 1) for i in range(nf)], 1, 1)
    for i in range(nf):
        b.row(f"cap_{i}", [(x[i][j], dem[j]) for j in range(nc)] + [(y[i], -cap[i])], up=0)
    return b.build()


def set_cover(m, n, seed):
    rng = np.random.default_rng(seed)
    b = Builder(f"setcover_{m}x{n}")
    cost = rng.integers(1, 100, n).astype(float)
    x = [b.var(f"s{j}", 0, 1, cost[j], True) for j in range(n)]
    for i in range(m):
        k = rng.integers(2, max(3, n // 20))
        cols = rng.choice(n, k, replace=False)
        b.row(f"e{i}", [(x[j], 1) for j in cols], lo=1)
    return b.build()


def unit_commitment(G, T, seed):
    """Thermal unit commitment (power-system dispatch): on/off, start-up, 3-segment
    piecewise-linear fuel cost, ramping, spinning reserve, minimum up time."""
    rng = np.random.default_rng(seed)
    b = Builder(f"uc_{G}g_{T}h")
    pmax = rng.uniform(100, 500, G).round()
    pmin = (pmax * rng.uniform(0.25, 0.45, G)).round()
    mc = rng.uniform(15, 60, G)  # marginal cost base
    noload = rng.uniform(100, 600, G)
    startc = rng.uniform(500, 4000, G)
    ramp = pmax * rng.uniform(0.3, 0.7, G)
    minup = rng.integers(2, 5, G)
    base = 0.55 * pmax.sum()
    hours = np.arange(T)
    demand = base * (0.75 + 0.25 * np.sin((hours - 6) / 24 * 2 * np.pi)) + rng.normal(0, 0.01 * base, T)
    reserve = 0.08 * demand
    u = [[b.var(f"u_{g}_{t}", 0, 1, float(noload[g]), True) for t in range(T)] for g in range(G)]
    v = [[b.var(f"v_{g}_{t}", 0, 1, float(startc[g]), True) for t in range(T)] for g in range(G)]
    seg = 3
    p = [[[b.var(f"p_{g}_{t}_{s}", 0, float((pmax[g] - pmin[g]) / seg), float(mc[g] * (1 + 0.15 * s)))
           for s in range(seg)] for t in range(T)] for g in range(G)]
    for g in range(G):
        for t in range(T):
            # segment output only if on
            for s in range(seg):
                b.row(f"seg_{g}_{t}_{s}", [(p[g][t][s], 1), (u[g][t], -(pmax[g] - pmin[g]) / seg)], up=0)
            b.row(f"start_{g}_{t}", [(v[g][t], 1), (u[g][t], -1)] + ([(u[g][t - 1], 1)] if t else []), lo=0)
            if t:
                out_t = [(p[g][t][s], 1) for s in range(seg)] + [(u[g][t], pmin[g])]
                out_p = [(p[g][t - 1][s], -1) for s in range(seg)] + [(u[g][t - 1], -pmin[g])]
                b.row(f"rup_{g}_{t}", out_t + out_p, up=float(ramp[g] + pmin[g]))
                b.row(f"rdn_{g}_{t}", out_t + out_p, lo=float(-ramp[g] - pmin[g]))
            # minimum up time: sum_{tau=t-L+1..t} v <= u_t
            L = int(minup[g])
            b.row(f"minup_{g}_{t}", [(v[g][tau], 1) for tau in range(max(0, t - L + 1), t + 1)] + [(u[g][t], -1)], up=0)
    for t in range(T):
        gen = []
        cap = []
        for g in range(G):
            gen += [(p[g][t][s], 1) for s in range(seg)] + [(u[g][t], pmin[g])]
            cap += [(u[g][t], pmax[g])]
        b.row(f"demand_{t}", gen, float(demand[t]), float(demand[t]))
        b.row(f"reserve_{t}", cap, lo=float(demand[t] + reserve[t]))
    return b.build()


# --------------------------------------------------------------- refinery cases
CRUDES = {
    # name: (API, sulfur wt%, price $/bbl, yields LPG, naphtha, kero, diesel, VGO, residue)
    "CR_LIGHT_SWEET": (40.0, 0.2, 84.0, [0.03, 0.26, 0.14, 0.27, 0.20, 0.10]),
    "CR_ARAB_LIGHT":  (33.0, 1.8, 80.5, [0.02, 0.19, 0.13, 0.25, 0.24, 0.17]),
    "CR_ARAB_HEAVY":  (28.0, 2.9, 76.0, [0.02, 0.15, 0.10, 0.22, 0.25, 0.26]),
    "CR_OFFSHORE_IN": (39.0, 0.15, 83.0, [0.03, 0.24, 0.15, 0.28, 0.20, 0.10]),
    "CR_HIGH_TAN":    (22.0, 0.6, 72.0, [0.01, 0.10, 0.09, 0.24, 0.28, 0.28]),
    "CR_MEDIUM_SOUR": (31.0, 2.2, 78.5, [0.02, 0.18, 0.12, 0.24, 0.25, 0.19]),
}
CUTS = ["LPG", "NAP", "KERO", "DSL", "VGO", "RES"]


def refinery_planning(T, seed, name=None):
    """Multi-period refinery production planning (MILP).

    Crude cargo selection (integer cargoes), CDU with crude-dependent yields and a
    feed sulfur/API window (linear crude blending), naphtha reformer, FCC with two
    operating modes (binary mode selection + switch-over cost), diesel hydrotreater,
    gasoline blending with linear octane blending, diesel sulfur spec, product demand
    windows, inventories, a planned FCC maintenance period. Objective: max gross margin.
    """
    rng = np.random.default_rng(seed)
    b = Builder(name or f"refplan_{T}p")
    b.maximize = True
    crudes = list(CRUDES)
    cargo = 500.0              # kbbl per cargo
    cdu_cap = 1100.0           # kbbl per period (~10 days @ 110 kbpd)
    ref_cap, fcc_cap, dht_cap = 180.0, 320.0, 420.0
    fcc_maint = T // 2         # FCC at 40% capacity in this period
    modes = {"MAXGAS": dict(GAS=0.55, LCO=0.20, LPG=0.15, SLURRY=0.10, opex=2.2),
             "MAXDIST": dict(GAS=0.42, LCO=0.33, LPG=0.12, SLURRY=0.13, opex=2.0)}
    price = dict(LPG=55.0, GASOLINE=110.0, DIESEL=112.0, ATF=108.0, FO=62.0)
    pmult = 1 + 0.03 * rng.standard_normal((T, len(price)))
    dmin = dict(LPG=20, GASOLINE=150, DIESEL=320, ATF=80, FO=40)
    dmax = dict(LPG=90, GASOLINE=420, DIESEL=600, ATF=200, FO=400)
    ron = dict(NAP=66.0, REFORMATE=98.0, FCCGAS=91.5, BUTANE=93.0)

    n_cargo, feed, inv = {}, {}, {}
    for t in range(T):
        for c in crudes:
            api, sul, pr, _ = CRUDES[c]
            prc = pr * (1 + 0.02 * rng.standard_normal())
            n_cargo[c, t] = b.var(f"cargo_{c}_{t}", 0, 2, -prc * cargo, True)
            feed[c, t] = b.var(f"cdu_{c}_{t}", 0, cdu_cap)
            inv[c, t] = b.var(f"cinv_{c}_{t}", 0, 1200.0, -0.15)  # holding cost
    for t in range(T):
        b.row(f"jetty_{t}", [(n_cargo[c, t], 1) for c in crudes], up=3)
        for c in crudes:
            prev = [(inv[c, t - 1], 1)] if t else []
            init = 250.0 if t == 0 and c in ("CR_ARAB_LIGHT", "CR_OFFSHORE_IN") else 0.0
            b.row(f"cbal_{c}_{t}", prev + [(n_cargo[c, t], cargo), (feed[c, t], -1), (inv[c, t], -1)], -init, -init)
        tot = [(feed[c, t], 1) for c in crudes]
        b.row(f"cducap_{t}", tot, lo=0.6 * cdu_cap, up=cdu_cap)
        # CDU feed quality window (linear crude blending by volume)
        b.row(f"cdu_sulfur_{t}", [(feed[c, t], CRUDES[c][1] - 2.0) for c in crudes], up=0)
        b.row(f"cdu_api_lo_{t}", [(feed[c, t], CRUDES[c][0] - 29.0) for c in crudes], lo=0)
        # cut production
        cut = {}
        for k, nm in enumerate(CUTS):
            cut[nm] = b.var(f"{nm}_{t}")
            b.row(f"yield_{nm}_{t}", [(feed[c, t], CRUDES[c][3][k]) for c in crudes] + [(cut[nm], -1)], 0, 0)
        # naphtha: to reformer or gasoline pool directly
        nap_ref = b.var(f"nap_to_ref_{t}", 0, ref_cap, -3.5)
        nap_gas = b.var(f"nap_to_gas_{t}")
        nap_exp = b.var(f"nap_export_{t}", 0, 60.0, 68.0 * pmult[t, 0])
        b.row(f"nap_split_{t}", [(cut["NAP"], 1), (nap_ref, -1), (nap_gas, -1), (nap_exp, -1)], 0, 0)
        reformate = b.var(f"reformate_{t}")
        b.row(f"reformer_{t}", [(nap_ref, 0.84), (reformate, -1)], 0, 0)
        # FCC with operating modes
        fcap = fcc_cap * (0.4 if t == fcc_maint else 1.0)
        vgo_fcc = {}
        fprod = {p: b.var(f"fcc_{p}_{t}") for p in ("GAS", "LCO", "LPG", "SLURRY")}
        for mname, md in modes.items():
            z = b.var(f"mode_{mname}_{t}", 0, 1, 0.0, True)
            vgo_fcc[mname] = b.var(f"vgo_{mname}_{t}", 0, INF, -md["opex"])
            b.row(f"fcccap_{mname}_{t}", [(vgo_fcc[mname], 1), (z, -fcap)], up=0)
            b.vars_mode = getattr(b, "vars_mode", {})
            b.vars_mode[mname, t] = z
        b.row(f"onemode_{t}", [(b.vars_mode[mm, t], 1) for mm in modes], 1, 1)
        for p in fprod:
            b.row(f"fccyield_{p}_{t}", [(vgo_fcc[mm], modes[mm][p]) for mm in modes] + [(fprod[p], -1)], 0, 0)
        vgo_fo = b.var(f"vgo_to_fo_{t}")
        b.row(f"vgo_split_{t}", [(cut["VGO"], 1)] + [(vgo_fcc[mm], -1) for mm in modes] + [(vgo_fo, -1)], 0, 0)
        # hydrotreater: kero/diesel/LCO -> ULSD ; kero may go straight to ATF
        kero_atf = b.var(f"kero_atf_{t}")
        kero_dht = b.var(f"kero_dht_{t}")
        b.row(f"kero_split_{t}", [(cut["KERO"], 1), (kero_atf, -1), (kero_dht, -1)], 0, 0)
        dht_feed = [(kero_dht, 1), (cut["DSL"], 1), (fprod["LCO"], 1)]
        b.row(f"dhtcap_{t}", dht_feed, up=dht_cap)
        # products (sales variables with period prices)
        pv = {}
        for k, pn in enumerate(price):
            pv[pn] = b.var(f"sell_{pn}_{t}", dmin[pn], dmax[pn], price[pn] * pmult[t, k])
        b.row(f"atf_{t}", [(kero_atf, 1), (pv["ATF"], -1)], 0, 0)
        b.row(f"diesel_{t}", [(kero_dht, 0.985), (cut["DSL"], 0.985), (fprod["LCO"], 0.985), (pv["DIESEL"], -1)], lo=0)
        b.row(f"lpg_{t}", [(cut["LPG"], 1), (fprod["LPG"], 1), (pv["LPG"], -1)], lo=0)
        b.row(f"fo_{t}", [(cut["RES"], 1), (fprod["SLURRY"], 1), (vgo_fo, 1), (pv["FO"], -1)], lo=0)
        # gasoline pool: blend volume balance + octane (linear blending index)
        butane = b.var(f"butane_blend_{t}", 0, 25.0)
        b.row(f"gas_vol_{t}", [(nap_gas, 1), (reformate, 1), (fprod["GAS"], 1), (butane, 1), (pv["GASOLINE"], -1)], 0, 0)
        b.row(f"gas_ron_{t}", [(nap_gas, ron["NAP"] - 91), (reformate, ron["REFORMATE"] - 91),
                               (fprod["GAS"], ron["FCCGAS"] - 91), (butane, ron["BUTANE"] - 91)], lo=0)
        b.row(f"gas_nap_limit_{t}", [(nap_gas, 1), (pv["GASOLINE"], -0.15)], up=0)
    # FCC mode switch-over cost
    for t in range(1, T):
        for mname in modes:
            s = b.var(f"switch_{mname}_{t}", 0, 1, -150.0)
            b.row(f"swdef_{mname}_{t}", [(s, 1), (b.vars_mode[mname, t], -1), (b.vars_mode[mname, t - 1], 1)], lo=0)
    return b.build()


def crude_scheduling(H, seed, name=None):
    """Crude-oil unloading, storage and CDU charging schedule (MILP).

    Discrete time (H periods of 8 h). Vessels (arrival, crude, parcel) unload at a
    single berth into dedicated storage tanks; at most two tanks charge the CDU in a
    period with min/max charging rates; a tank cannot receive and charge in the same
    period; CDU feed sulfur spec (linear since tanks are crude-dedicated); CDU rate
    fixed per period. Cost = demurrage + tank changeovers + inventory holding.
    """
    rng = np.random.default_rng(seed)
    b = Builder(name or f"crudesched_{H}")
    tanks = ["T_SWEET", "T_SOUR", "T_HEAVY", "T_OFFSH"]
    tank_sulfur = dict(T_SWEET=0.2, T_SOUR=2.2, T_HEAVY=2.9, T_OFFSH=0.15)
    tank_cap = dict(T_SWEET=600.0, T_SOUR=700.0, T_HEAVY=600.0, T_OFFSH=500.0)
    inv0 = dict(T_SWEET=220.0, T_SOUR=300.0, T_HEAVY=200.0, T_OFFSH=180.0)
    nv = max(3, H // 5)
    arrivals = sorted(rng.integers(0, max(1, H - 6), nv))
    vcrude = [tanks[k] for k in rng.integers(0, len(tanks), nv)]
    parcel = rng.uniform(150, 300, nv).round()
    urate = 110.0        # kbbl per period unloading
    cdu_rate = 36.0      # kbbl per period
    fmin, fmax = 8.0, 30.0
    s_spec = 1.6
    demurrage = 25.0     # k$ per period
    changeover = 5.0
    w, un, done = {}, {}, {}
    for v in range(nv):
        for t in range(H):
            if t < arrivals[v]:
                continue
            w[v, t] = b.var(f"unl_{v}_{t}", 0, 1, 0.0, True)
            un[v, t] = b.var(f"uvol_{v}_{t}", 0, urate)
            b.row(f"urate_{v}_{t}", [(un[v, t], 1), (w[v, t], -urate)], up=0)
            done[v, t] = b.var(f"done_{v}_{t}", 0, 1, -demurrage, True)  # reward finishing early
        ts = [t for t in range(arrivals[v], H)]
        b.row(f"parcel_{v}", [(un[v, t], 1) for t in ts], parcel[v], parcel[v])
        for t in ts:
            # done_{v,t} = 1 only if the parcel is fully discharged by t
            b.row(f"donedef_{v}_{t}", [(un[v, tt], 1) for tt in ts if tt <= t] + [(done[v, t], -parcel[v])], lo=0)
            if t > arrivals[v]:
                b.row(f"donemono_{v}_{t}", [(done[v, t], 1), (done[v, t - 1], -1)], lo=0)
            b.row(f"nounl_after_{v}_{t}", [(w[v, t], 1), (done[v, t - 1], 1)] if t > arrivals[v] else [(w[v, t], 1)], up=1)
        b.offset += demurrage * len(ts)
    for t in range(H):
        b.row(f"berth_{t}", [(w[v, t], 1) for v in range(nv) if (v, t) in w], up=1)
    u, f, inv, chg = {}, {}, {}, {}
    for s in tanks:
        for t in range(H):
            u[s, t] = b.var(f"feed_on_{s}_{t}", 0, 1, 0.0, True)
            f[s, t] = b.var(f"feed_{s}_{t}", 0, fmax)
            inv[s, t] = b.var(f"inv_{s}_{t}", 0.1 * tank_cap[s], tank_cap[s], 0.02)
            b.row(f"fmax_{s}_{t}", [(f[s, t], 1), (u[s, t], -fmax)], up=0)
            b.row(f"fmin_{s}_{t}", [(f[s, t], 1), (u[s, t], -fmin)], lo=0)
            rec = [(un[v, t], 1) for v in range(nv) if vcrude[v] == s and (v, t) in un]
            prev = [(inv[s, t - 1], 1)] if t else []
            b.row(f"tbal_{s}_{t}", prev + rec + [(f[s, t], -1), (inv[s, t], -1)], -inv0[s] if t == 0 else 0,
                  -inv0[s] if t == 0 else 0)
            for v in range(nv):
                if vcrude[v] == s and (v, t) in w:
                    b.row(f"norecfeed_{s}_{v}_{t}", [(u[s, t], 1), (w[v, t], 1)], up=1)
            if t:
                chg[s, t] = b.var(f"chg_{s}_{t}", 0, 1, changeover)
                b.row(f"chgdef_{s}_{t}", [(chg[s, t], 1), (u[s, t], -1), (u[s, t - 1], 1)], lo=0)
    for t in range(H):
        b.row(f"cdu_{t}", [(f[s, t], 1) for s in tanks], cdu_rate, cdu_rate)
        b.row(f"twotanks_{t}", [(u[s, t], 1) for s in tanks], up=2)
        b.row(f"sulfur_{t}", [(f[s, t], tank_sulfur[s] - s_spec) for s in tanks], up=0)
    return b.build()


def gasoline_blending(T, seed, name=None):
    """Multi-period gasoline blending LP with linear blending indices.

    Components: butane, light naphtha, reformate, FCC gasoline, alkylate, isomerate,
    ethanol. Products: 91 RON regular, 95 RON premium. Specs: RON (volumetric
    blending), RVP via the RVP^1.25 blending index, sulfur, benzene, aromatics,
    olefins, ethanol cap. Component inventories and production rates per period.
    """
    rng = np.random.default_rng(seed)
    b = Builder(name or f"gasblend_{T}p")
    b.maximize = True
    comp = {
        #            RON   RVP(psi) S(ppm) Bz%  Aro% Ole% cost  prod/period
        "BUTANE":   (93.0, 52.0,  10,   0.0,  0,   0,   40.0, 12),
        "LSRN":     (68.0, 11.0,  30,   1.2,  3,   1,   62.0, 60),
        "REFORMATE": (98.0, 4.5,  1,    3.2,  62,  1,   88.0, 90),
        "FCCGAS":   (91.5, 7.5,   40,   0.9,  28,  28,  80.0, 110),
        "ALKYLATE": (95.0, 5.5,   5,    0.0,  0,   0,   95.0, 35),
        "ISOMERATE": (87.0, 13.0, 1,    0.0,  0,   0,   80.0, 30),
        "ETHANOL":  (109.0, 18.0, 5,    0.0,  0,   0,   85.0, 25),
    }
    prods = {"REG91": (91.0, 9.0, 10, 1.0, 35, 18, 0.10, 104.0, (120, 260)),
             "PREM95": (95.0, 8.5, 10, 1.0, 35, 18, 0.10, 112.0, (40, 120))}
    inv = {}
    for t in range(T):
        blend = {}
        for p, spec in prods.items():
            ronmin, rvpmax, smax, bzmax, aromax, olemax, ethmax, pr, (dlo, dhi) = spec
            prc = pr * (1 + 0.03 * rng.standard_normal())
            sale = b.var(f"sale_{p}_{t}", dlo, dhi, prc)
            terms = []
            for cname in comp:
                blend[cname, p] = b.var(f"b_{cname}_{p}_{t}")
                terms.append((blend[cname, p], 1))
            b.row(f"vol_{p}_{t}", terms + [(sale, -1)], 0, 0)
            b.row(f"ron_{p}_{t}", [(blend[cn, p], comp[cn][0] - ronmin) for cn in comp], lo=0)
            b.row(f"rvp_{p}_{t}", [(blend[cn, p], comp[cn][1] ** 1.25 - rvpmax ** 1.25) for cn in comp], up=0)
            b.row(f"sul_{p}_{t}", [(blend[cn, p], comp[cn][2] - smax) for cn in comp], up=0)
            b.row(f"bz_{p}_{t}", [(blend[cn, p], comp[cn][3] - bzmax) for cn in comp], up=0)
            b.row(f"aro_{p}_{t}", [(blend[cn, p], comp[cn][4] - aromax) for cn in comp], up=0)
            b.row(f"ole_{p}_{t}", [(blend[cn, p], comp[cn][5] - olemax) for cn in comp], up=0)
            b.row(f"eth_{p}_{t}", [(blend[cn, p], (1 - ethmax) if cn == "ETHANOL" else -ethmax) for cn in comp], up=0)
        for cn, data in comp.items():
            prod = data[7] * (1 + 0.1 * rng.standard_normal())
            buy = b.var(f"buy_{cn}_{t}", 0, 40.0 if cn in ("BUTANE", "ETHANOL", "ALKYLATE") else 0.0, -data[6] * 1.08)
            sell = b.var(f"sellcomp_{cn}_{t}", 0, INF, data[6] * 0.9)
            inv[cn, t] = b.var(f"inv_{cn}_{t}", 0, 3 * data[7], -0.2)
            prev = [(inv[cn, t - 1], 1)] if t else []
            init = 0.5 * data[7] if t == 0 else 0.0
            b.row(f"cbal_{cn}_{t}", prev + [(buy, 1), (inv[cn, t], -1), (sell, -1)]
                  + [(blend[cn, p], -1) for p in prods], -prod - init, -prod - init)
    return b.build()


# ---------------------------------------------------------------------------- QP
def portfolio_qp(n, seed):
    """Markowitz mean-variance portfolio (convex QP) with a factor covariance, sector limits."""
    rng = np.random.default_rng(seed)
    b = Builder(f"portfolio_{n}")
    k = max(3, n // 20)
    F = rng.normal(0, 0.1, (n, k))
    d = rng.uniform(0.01, 0.05, n)
    Sigma = F @ F.T + np.diag(d)
    mu = rng.uniform(0.02, 0.15, n)
    gamma = 2.0
    x = [b.var(f"w{j}", 0, 0.1, -mu[j]) for j in range(n)]
    for i in range(n):
        for j in range(i, n):
            val = gamma * Sigma[i, j]
            if abs(val) > 1e-12:
                if i == j:
                    b.quad(i, i, val)
                else:
                    b.quad(i, j, val)
    b.row("budget", [(x[j], 1) for j in range(n)], 1, 1)
    sectors = rng.integers(0, 8, n)
    for s in range(8):
        idx = [j for j in range(n) if sectors[j] == s]
        if idx:
            b.row(f"sector{s}", [(x[j], 1) for j in idx], up=0.3)
    return b.build()


def blend_tracking_qp(T, seed):
    """Blending with quadratic penalty on deviation from target product qualities
    (a common way refinery schedulers regularise give-away)."""
    rng = np.random.default_rng(seed)
    b = Builder(f"blendqp_{T}p")
    comps = 6
    q = rng.uniform(80, 100, comps)
    s = rng.uniform(1, 50, comps)
    cost = rng.uniform(60, 95, comps)
    for t in range(T):
        x = [b.var(f"x_{i}_{t}", 0, 60, cost[i]) for i in range(comps)]
        dev_q = b.var(f"devq_{t}", -INF, INF)
        dev_s = b.var(f"devs_{t}", -INF, INF)
        b.row(f"vol_{t}", [(xi, 1) for xi in x], 100, 100)
        b.row(f"qdef_{t}", [(x[i], q[i]) for i in range(comps)] + [(dev_q, -1)], 92 * 100, 92 * 100)
        b.row(f"sdef_{t}", [(x[i], s[i]) for i in range(comps)] + [(dev_s, -1)], 10 * 100, 10 * 100)
        b.quad(dev_q, dev_q, 2.0)
        b.quad(dev_s, dev_s, 0.5)
        if t:
            for i in range(comps):
                # smooth recipe changes: penalise (x_it - x_i,t-1)^2
                j0 = (t - 1) * (comps + 2) + i
                j1 = t * (comps + 2) + i
                b.quad(j0, j0, 0.3); b.quad(j1, j1, 0.3); b.quad(j0, j1, -0.3)
    return b.build()


def bounded_least_squares(m, n, seed):
    rng = np.random.default_rng(seed)
    b = Builder(f"bls_{m}x{n}")
    C = sp.random(m, n, density=min(1.0, 5.0 / n), random_state=seed, data_rvs=lambda k: rng.normal(0, 1, k)).tocsr()
    C = C + sp.eye(m, n) * 0.5
    dvec = rng.normal(0, 1, m)
    Q = (C.T @ C).tocoo()
    lin = -(C.T @ dvec)
    for j in range(n):
        b.var(f"x{j}", -0.5, 0.5, float(lin[j]))
    for i, j, v in zip(Q.row, Q.col, Q.data):
        if i <= j and abs(v) > 1e-14:
            b.quad(int(i), int(j), float(v))
    b.row("sum", [(j, 1) for j in range(n)], lo=-1, up=1)
    b.offset = 0.5 * float(dvec @ dvec)
    return b.build()


SUITE = [
    # (category, class, generator, args, description)
    ("robustness", "LP", transport, (40, 60, 1), "Balanced transportation LP (primal degenerate)"),
    ("robustness", "LP", assignment, (60, 2), "Assignment LP, 3600 vars (massively degenerate)"),
    ("robustness", "LP", degenerate_lp, (400, 800, 3), "Every row active at a known vertex (primal degeneracy)"),
    ("robustness", "LP", ill_conditioned, (300, 500, 4), "Row/col scaling over 8 decades + near-parallel rows"),
    ("robustness", "LP", ill_conditioned, (1200, 2000, 5), "Larger ill-conditioned LP"),
    ("scale", "LP", multicommodity, (12, 12, 8, 6), "Multicommodity flow on 12x12 grid, 8 commodities"),
    ("scale", "LP", multicommodity, (25, 25, 16, 7), "Multicommodity flow, 25x25 grid, 16 commodities"),
    ("milp", "MILP", multiknapsack, (60, 5, 8), "Multi-dimensional knapsack"),
    ("milp", "MILP", facility_location, (15, 60, 9), "Capacitated facility location, weak aggregated big-M"),
    ("milp", "MILP", set_cover, (200, 120, 10), "Set covering"),
    ("industrial", "MILP", unit_commitment, (10, 24, 11), "Power dispatch: unit commitment, 10 units x 24 h"),
    ("industrial", "LP", gasoline_blending, (6, 12), "Refinery: multi-period gasoline blending (RON/RVP/S/Bz/Aro/Ole)"),
    ("industrial", "MILP", refinery_planning, (4, 13), "Refinery: 4-period planning, crude cargoes + FCC modes"),
    ("industrial", "MILP", refinery_planning, (8, 14, "refplan_8p"), "Refinery: 8-period planning"),
    ("industrial", "MILP", crude_scheduling, (15, 15), "Refinery: crude unloading & tank scheduling (5 days, 8h slots)"),
    ("qp", "QP", portfolio_qp, (200, 16), "Markowitz portfolio QP, factor covariance, 200 assets"),
    ("qp", "QP", blend_tracking_qp, (12, 17), "Blending with quadratic quality-deviation and recipe-smoothing penalties"),
    ("qp", "QP", bounded_least_squares, (400, 300, 18), "Bounded sparse least squares"),
]

GPU_SUITE = [
    ("gpu", "LP", multicommodity, (40, 40, 24, 21, "mcf_40x40_k24"), "Large multicommodity flow (~180k vars)"),
    ("gpu", "LP", multicommodity, (60, 60, 32, 22, "mcf_60x60_k32"), "Large multicommodity flow (~450k vars)"),
    ("gpu", "LP", multicommodity, (90, 90, 40, 23, "mcf_90x90_k40"), "Very large multicommodity flow (~1.3M vars)"),
]


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "base"
    suite = SUITE if which == "base" else GPU_SUITE if which == "gpu" else SUITE + GPU_SUITE
    manifest_path = os.path.join(ROOT, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    for cat, cls, gen, args, desc in suite:
        if gen is multicommodity and cat == "gpu":
            name = args[4]
            d = os.path.join(ROOT, cat)
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, name + ".mps")
            info = multicommodity_stream(*args[:4], path)
            rel = os.path.relpath(path, os.path.dirname(ROOT)).replace("\\", "/")
            manifest[name] = dict(path=rel, category=cat, cls=cls, description=desc,
                                  source="synthetic (NIRNAY generator)", **info)
            print(f"{name:28s} {cls:5s} rows={info['rows']:7d} cols={info['cols']:8d} nnz={info['nnz']:9d}")
            continue
        model = gen(*args)
        d = os.path.join(ROOT, cat)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, model.name + ".mps")
        write_mps(model, path)
        rel = os.path.relpath(path, os.path.dirname(ROOT)).replace("\\", "/")
        manifest[model.name] = dict(path=rel, category=cat, cls=cls, description=desc,
                                    rows=int(model.m), cols=int(model.n), nnz=int(model.A.nnz),
                                    integers=int(model.integrality.sum()), source="synthetic (NIRNAY generator)")
        print(f"{model.name:28s} {cls:5s} rows={model.m:7d} cols={model.n:8d} nnz={model.A.nnz:9d} int={int(model.integrality.sum())}")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1)


if __name__ == "__main__":
    main()
