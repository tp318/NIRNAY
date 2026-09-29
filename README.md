# NIRNAY

LP / MILP / QP solver written from scratch in C++17: presolve, scaling, interior point (IPM),
dual simplex, branch-and-cut (GMI cuts, strong branching, diving heuristics), and a PDHG
first-order method with an optional CUDA backend.

## Quick start (no build needed)

    docker run --rm ghcr.io/tp318/nirnay:latest solve afiro.mps

or download a static binary from Releases. See **[DEPLOY.md](DEPLOY.md)** for copy-paste
commands, the expected output, and how to verify it against the published Netlib optima.

## Build from source

- Linux/macOS: `make` (OpenMP), `make CUDA=1` (with nvcc), or `sh build.sh` for the static release binary.
- Windows: `build.bat` (MSVC, CUDA if `nvcc` is on PATH), `build.bat cpu`, or `build-release.bat` for the static exe.

## Run

    nirnay solve instances/netlib/afiro.mps [--method auto|ipm|simplex|pdhg] [--device cpu|gpu] [--time-limit s] [--json out.json]

## Benchmarks

    pip install numpy scipy
    python tools/generate.py gpu          # regenerate the large instances/gpu/ LPs (not committed)
    python tools/bench.py --suite base    # vs HiGHS (via SciPy) -> results/report_base.html
    python tools/bench.py --suite gpu     # PDHG CPU vs GPU on large LPs

## Open issues

- IPM returns NUMERICAL_ERROR on the multicommodity flow LPs (scale/mcf_12x12_k8, scale/mcf_25x25_k16);
  dual simplex and PDHG solve them.
- MILP: refplan_8p hits the 120 s limit (HiGHS: 15 s); uc_10g_24h and refplan_4p are slower than HiGHS.
