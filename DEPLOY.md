# Running NIRNAY: zero-setup guide

Two ways to go from nothing to a verified solve. Both need **no compiler, no libraries, no Python**.

| Path | Needs | Time |
|---|---|---|
| A. Docker image | Docker | ~30 s |
| B. Prebuilt binary | nothing (Linux x86_64 or Windows x86_64) | ~10 s |

Both ship the same solver. Eight Netlib LP instances plus three of the project's own models
are bundled so you can test without supplying a file.

---

## A. Docker

```bash
docker pull ghcr.io/tp318/nirnay:latest
docker run --rm ghcr.io/tp318/nirnay:latest solve afiro.mps
```

Bundled instances (in the image's working directory `/instances`):
`afiro.mps  sc50a.mps  sc105.mps  adlittle.mps  blend.mps  kb2.mps  share2b.mps  stocfor1.mps`
(Netlib LP), `gasblend_6p.mps` (LP), `setcover_200x120.mps` (MILP), `portfolio_200.mps` (QP).

Your own model file (mount the current directory at `/data`):

```bash
docker run --rm -v "$PWD:/data" ghcr.io/tp318/nirnay:latest solve /data/model.mps --time-limit 60
```

No registry access? Build the identical image from a clone (about a minute; the build itself
fails if the binary is not static or does not solve `afiro` correctly):

```bash
git clone https://github.com/tp318/NIRNAY.git && cd NIRNAY
docker build -t nirnay .
docker run --rm nirnay solve afiro.mps
```

The image is `FROM scratch`: one static binary plus the instance files, a few MB in total. No OS, no shell.

---

## B. Prebuilt binary

**Linux x86_64** (any distribution; the binary is fully static):

```bash
curl -fLO https://github.com/tp318/NIRNAY/releases/latest/download/nirnay-linux-x86_64
chmod +x nirnay-linux-x86_64
curl -fLO https://raw.githubusercontent.com/tp318/NIRNAY/main/instances/netlib/afiro.mps
./nirnay-linux-x86_64 solve afiro.mps
```

**Windows x86_64** (PowerShell; depends only on `KERNEL32.dll`, no Visual C++ redistributable):

```powershell
Invoke-WebRequest https://github.com/tp318/NIRNAY/releases/latest/download/nirnay-windows-x86_64.exe -OutFile nirnay.exe
Invoke-WebRequest https://raw.githubusercontent.com/tp318/NIRNAY/main/instances/netlib/afiro.mps -OutFile afiro.mps
.\nirnay.exe solve afiro.mps
```

Checksums are published next to the binaries in `SHA256SUMS.txt`.

---

## Expected output (Netlib `afiro`)

```
NIRNAY | model AFIRO: 27 rows, 32 cols, 83 nnz, 0 integer, minimize  (read 0.000s)
Presolve: 27 rows, 32 cols -> 25 rows, 32 cols  (singleton 2, forcing 0, redundant 0, fixed 0, empty 0) 0.000s
IPM: rows 25  cols 32  (augmented 74)  nnz(L) 152
 iter      primal obj        dual obj     p.inf     d.inf        mu   step
    0  2.57585124e+02 -1.37347041e+04  1.11e-01  2.77e-01  4.03e+01  0.000
  ...
    8 -4.64753143e+02 -4.64753143e+02  5.48e-16  3.48e-15  3.52e-12  1.000
Status        : OPTIMAL
Method        : ipm
Objective     : -4.647531428559e+02
Violations    : row 3.72e-14  bound 0.00e+00  integrality 0.00e+00
KKT (orig.)   : rel.primal 3.72e-14  rel.dual 5.55e-17  rel.gap 5.47e-12
Iterations    : 8
Time          : 0.002s (presolve 0.000s, solve 0.002s)
```

What to check:

- **Objective** `-4.647531428559e+02` matches the published Netlib optimum **−464.75314286**.
  Expect agreement to about 10 significant digits; the last digit or two may differ across platforms.
- **Certificate**: the `Violations` and `KKT (orig.)` lines are recomputed by the solver on the
  *original* model, not the presolved or scaled one. Primal feasibility, dual feasibility and
  the primal–dual gap should all be ≤ 1e-8, which proves optimality independently of the algorithm.
- **Time** will differ by machine.

Reference optima for all bundled Netlib instances: [`instances/netlib/EXPECTED.txt`](instances/netlib/EXPECTED.txt).
From a clone, check all of them in one command (needs Python 3):

```bash
python3 tools/check_netlib.py path/to/nirnay-linux-x86_64          # IPM (default)
python3 tools/check_netlib.py path/to/nirnay-linux-x86_64 --method simplex
```

Each line prints `PASS`/`FAIL` with the relative error against the published value.

---

## Useful options

```
nirnay solve <model.mps|.qps> [options]
  --method auto|ipm|simplex|pdhg   LP/QP algorithm (MILPs always use branch-and-cut)
  --time-limit <sec>               wall-clock limit
  --gap <rel>                      MILP relative gap (default 1e-4)
  --threads <n>                    CPU threads (Linux build; the Windows release exe is single-threaded)
  --json <file>                    machine-readable result
  --sol <file>                     primal (and dual) solution
  --quiet                          only the final summary
```

Supported input: fixed and free MPS, including RANGES, all bound types, integer markers,
OBJSENSE, and QUADOBJ/QMATRIX for convex QP. Files must be uncompressed (`gunzip model.mps.gz` first).

## System requirements

- **Linux**: x86_64, any kernel from the last 10+ years. Statically linked with musl, no
  runtime dependencies (`ldd` reports "not a dynamic executable").
- **Windows**: x86_64, Windows 10 or later. Static C runtime; only system `KERNEL32.dll`.
- **Docker**: any Docker engine on x86_64.
- The GPU (CUDA) PDHG backend is **not** in the release binaries. It needs a source build with
  `build.bat` (Windows) or `make CUDA=1` (Linux) plus the CUDA toolkit.

## Building the release artifacts yourself

| Target | Command | Output |
|---|---|---|
| Linux static binary | `sh build.sh` (uses Docker/Alpine; falls back to `make static`) | `dist/nirnay-linux-x86_64` |
| Windows static exe | `build-release.bat` (Visual Studio Build Tools) | `dist\nirnay-windows-x86_64.exe` |
| Docker image | `docker build -t nirnay .` | image `nirnay` |

Pushing a tag `v*` runs `.github/workflows/release.yml`, which builds both binaries, verifies
them against the Netlib reference values, attaches them to a GitHub Release, and pushes
`ghcr.io/tp318/nirnay:<version>` and `:latest`.

### Making it reachable for judges (repository owner)

The repository is private, so the links above work only for people with access. Before
sharing, either make the repository public (Settings → General → Danger Zone → Change
visibility), or keep it private and add the judges as collaborators. After the first release,
also make the container package public: GitHub profile → Packages → `nirnay` → Package
settings → Change visibility → Public.
