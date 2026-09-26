# NIRNAY

LP / MILP / QP solver written from scratch in C++17: presolve, scaling, interior point (IPM),
dual simplex, branch-and-cut (GMI cuts, strong branching, diving heuristics), and a PDHG
first-order method with an optional CUDA backend.

## Build

- Linux/macOS: `make` (CPU-only, OpenMP), or `make CUDA=1` with nvcc.
- Windows: `build.bat` (MSVC 2019, CUDA if `nvcc` is on PATH) or `build.bat cpu`.

## Run

    bin/nirnay instances/industrial/gasblend_6p.mps --method ipm|simplex|pdhg [--device cpu|gpu] [--time-limit s] [--json out.json]

## Benchmarks

    pip install numpy scipy
    python tools/generate.py gpu          # regenerate the large instances/gpu/ LPs (not committed)
    python tools/bench.py --suite base    # vs HiGHS (via SciPy) -> results/report_base.html
    python tools/bench.py --suite gpu     # PDHG CPU vs GPU on large LPs

## Open issues

- IPM returns NUMERICAL_ERROR on the multicommodity flow LPs (scale/mcf_12x12_k8, scale/mcf_25x25_k16);
  dual simplex and PDHG solve them.
- MILP: refplan_8p hits the 120 s limit (HiGHS: 15 s); uc_10g_24h and refplan_4p are slower than HiGHS.
