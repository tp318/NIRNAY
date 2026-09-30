"""Refinery stress test: a crude-blending / scheduling LP built to break naive solvers.

    python tools/stress_refinery.py            -> instances/stress/refinery_stress_30d.mps

Structure (per day t = 1..T):
  * 5 crudes: purchase b[c,t], charge to the crude unit p[c,t], tank inventory I[c,t]
  * 3 products (BS-VI petrol, BS-VI diesel, fuel oil): make y[k,t] from crude yields,
    sell s[k,t], product inventory J[k,t]
  * sulfur and density specifications on every product, as linear blending rows
Deliberate numerical traps:
  * mixed units: flows in tonnes (~46,000 t/day, MRPL scale) next to sulfur limits of
    10 ppm (1e-5) -> matrix coefficients span ~1e-6 .. 1e5
  * degeneracy: the crude-unit capacity row appears three times (t, kt and Mt units),
    two crudes have identical margins and properties (ties -> non-unique optimum), and
    capacity, tankage and demand are sized so several constraints bind at one vertex
Kept under 1000 rows / columns so the size-limited CPLEX Community edition can solve it too.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from generate import Builder  # noqa: E402
from mpsio import write_mps  # noqa: E402

INF = math.inf


def build(T=30):
    b = Builder("refinery_stress_30d")
    crudes = [  # name, price $/t, sulfur (mass fraction), density t/m3, yields (petrol, diesel, fuel oil)
        ("arab_light", 520.0, 0.0180, 0.855, (0.22, 0.34, 0.38)),
        ("basrah_medium", 505.0, 0.0290, 0.874, (0.18, 0.31, 0.46)),
        ("murban", 548.0, 0.0079, 0.824, (0.28, 0.37, 0.30)),
        ("murban_b", 548.0, 0.0079, 0.824, (0.28, 0.37, 0.30)),   # identical twin: tie -> degenerate optimum
        ("mumbai_high", 560.0, 0.0017, 0.827, (0.30, 0.38, 0.27)),
    ]
    prods = [  # name, price $/t, sulfur spec (fraction), max density, demand t/day
        ("petrol_bs6", 880.0, 10e-6, 0.775, 9800.0),
        ("diesel_bs6", 840.0, 10e-6, 0.845, 15500.0),
        ("fuel_oil", 470.0, 0.0350, 0.991, 1e9),
    ]
    # desulfurisation leaves a crude-specific residual: sulfur in product = f_k * crude sulfur
    hdt = (0.00040, 0.00055, 1.0)
    cap_t = 46000.0             # crude unit, t/day (16.8 MMT / 365)
    tank_c, tank_k = 180000.0, 60000.0
    P, B, I, Y, S, J = {}, {}, {}, {}, {}, {}
    for t in range(T):
        for ci, (cn, price, *_rest) in enumerate(crudes):
            B[ci, t] = b.var(f"buy_{cn}_{t}", 0, 30000.0, price)
            P[ci, t] = b.var(f"run_{cn}_{t}", 0, 30000.0, 4.0)            # processing cost $/t
            I[ci, t] = b.var(f"inv_{cn}_{t}", 0, tank_c, 0.05)
        for ki, (kn, price, *_rest) in enumerate(prods):
            Y[ki, t] = b.var(f"make_{kn}_{t}", 0, INF, 0.0)
            S[ki, t] = b.var(f"sell_{kn}_{t}", 0, prods[ki][4], -price)
            J[ki, t] = b.var(f"pinv_{kn}_{t}", 0, tank_k, 0.08)
    for t in range(T):
        for ci, (cn, *_rest) in enumerate(crudes):
            prev = [(I[ci, t - 1], 1.0)] if t > 0 else []
            opening = 60000.0 if t == 0 else 0.0
            b.row(f"cbal_{cn}_{t}", prev + [(B[ci, t], 1.0), (P[ci, t], -1.0), (I[ci, t], -1.0)], -opening, -opening)
        # crude unit capacity, stated three times in different units (parallel rows -> degeneracy)
        run = [(P[ci, t], 1.0) for ci in range(len(crudes))]
        b.row(f"cdu_t_{t}", run, -INF, cap_t)
        b.row(f"cdu_kt_{t}", [(j, 1e-3) for j, _ in run], -INF, cap_t * 1e-3)
        b.row(f"cdu_Mt_{t}", [(j, 1e-6) for j, _ in run], -INF, cap_t * 1e-6)
        for ki, (kn, _price, sspec, dmax, _dem) in enumerate(prods):
            # yield balance: make = sum yield * run
            b.row(f"yield_{kn}_{t}", [(Y[ki, t], 1.0)] + [(P[ci, t], -crudes[ci][4][ki]) for ci in range(len(crudes))], 0, 0)
            prev = [(J[ki, t - 1], 1.0)] if t > 0 else []
            b.row(f"pbal_{kn}_{t}", prev + [(Y[ki, t], 1.0), (S[ki, t], -1.0), (J[ki, t], -1.0)], 0, 0)
            # sulfur spec in ppm territory: sum y_ck (f_k s_c - spec_k) run_c <= 0  (coefficients ~1e-6 .. 1e-2)
            b.row(f"sulfur_{kn}_{t}",
                  [(P[ci, t], crudes[ci][4][ki] * (hdt[ki] * crudes[ci][2] - sspec)) for ci in range(len(crudes))],
                  -INF, 0.0)
            # density spec on the cut: rho_ck = rho_c + shift_k (petrol lighter, fuel oil heavier than the crude);
            # sum y_ck (rho_ck - rho_max) run_c <= 0
            shift = (-0.110, -0.015, 0.105)[ki]
            b.row(f"density_{kn}_{t}",
                  [(P[ci, t], crudes[ci][4][ki] * (crudes[ci][3] + shift - dmax)) for ci in range(len(crudes))],
                  -INF, 0.0)
    return b.build()


def main():
    m = build()
    out = os.path.join(os.path.dirname(HERE), "instances", "stress")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "refinery_stress_30d.mps")
    write_mps(m, path)
    A = m.A.tocoo()
    import numpy as np
    av = np.abs(A.data[A.data != 0])
    print(f"{path}: {m.A.shape[0]} rows, {m.A.shape[1]} cols, {A.nnz} nnz; |coef| range {av.min():.1e} .. {av.max():.1e}"
          f" (ratio {av.max() / av.min():.1e})")


if __name__ == "__main__":
    main()
