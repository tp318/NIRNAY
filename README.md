# NIRNAY

LP / convex QP / MILP solver written from scratch in C++17 and CUDA: presolve, interior point,
dual simplex, branch-and-cut, and a GPU first-order method (PDHG) with simplex crossover.
No third-party solver or math library inside (no COIN-OR, HiGHS, cuSPARSE, cuBLAS, cuDSS).

## 1. How to run

```bash
docker run --rm ghcr.io/tp318/nirnay:latest solve afiro.mps
```

Or download a static binary from [Releases](https://github.com/tp318/NIRNAY/releases) and run
`nirnay solve model.mps`. Useful flags: `--method auto|ipm|simplex|pdhg`, `--device cpu|gpu`,
`--time-limit <s>`, `--sol out.sol`, `--json out.json`. Build from source: `sh build.sh` (Linux) or
`build.bat` (Windows, CUDA). Full instructions and expected output: [DEPLOY.md](DEPLOY.md).

## 2. Current results

Machine: laptop, NVIDIA RTX 3050 6 GB, 12 CPU threads. "Wrong" = reported optimal with an incorrect objective.

| Benchmark | NIRNAY | Wrong |
|---|---|---|
| Netlib LP, 91 instances, 5 s each | 86–88 optimal (edge cases vary run to run) | 0 |
| Maros–Mészáros QP, 138 instances, 5 s each | 116 optimal, 114 confirmed vs published optima | 0 |
| MIPLIB 3 MILP, 65 instances, 20 s each | 34 proven optimal | 0 |
| Refinery stress LP (coefficients 3e-8 .. 1, degenerate) | optimal, 0.011 s | 0 |

**vs IBM CPLEX 22.1 (Community edition, ≤1000 rows/cols):** all 18 models both could run agree
(worst relative difference 4.5e-9). NIRNAY is faster on small LPs, QPs and easy MILPs, and 5–10x
slower on harder MILPs (crudesched, refplan). Stress LP: CPLEX 0.03 s, NIRNAY 0.011 s.

**vs HiGHS 1.x (via SciPy):** same objectives on every instance both solved. MIPLIB 3 in 30 s:
HiGHS 44/65, NIRNAY 34/65 in 20 s.

**GPU PDHG:** 3–5x faster per iteration than the same algorithm on 12 CPU threads (0.45–1.28 M
variables); mixed precision adds 1.15–1.55x.

**Gurobi, FICO Xpress:** not benchmarked (no licence available here). No claims are made about them.

## 3. Future work and challenges

| Challenge | How we will solve it |
|---|---|
| Hard MILPs close slowly (HiGHS/CPLEX prove some in <1 s that take us >20 s) | MILP presolve with probing and bound tightening; more cut families (knapsack cover, MIR, flow cover); primal heuristics (RINS, feasibility pump); parallel tree search on CPU threads |
| Presolve removes too little | Add parallel-row detection, doubleton equations and dominated columns, and report the nonzero reduction for every model |
| Crossover is slow when the simplex needs many pivots | Faster dual simplex: hypersparse FTRAN/BTRAN, better LU update, bound-flipping tuning; smarter crossover basis from PDHG |
| IPM stalls on a few ill-conditioned LPs and QPs | Dynamic KKT regularisation, better starting points, and a first-order QP fallback |
| GPU PDHG handles LP only | Extend the GPU engine to convex QP (Halpern PDHG with a quadratic term) |
| High accuracy costs extra PDHG iterations on the largest LPs | Earlier handoff to crossover, and feasibility polishing before the final check |
| Integration into refinery workflows | `.lp` file reader, Python API (`import nirnay`), and a MINLP roadmap (spatial branch-and-bound, outer approximation) |

Benchmark scripts: `tools/sweep.py`, `tools/compare.py`, `tools/check_netlib.py`.
