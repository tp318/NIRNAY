"""Solve every bundled Netlib instance and compare with the published optimal objective.

    python tools/check_netlib.py [path/to/nirnay] [--method ipm|simplex|pdhg]

Exit code 0 only if every instance is OPTIMAL and within 1e-6 relative of the
reference value (1e-3 for PDHG). Reference values: instances/netlib/EXPECTED.txt
(from the Netlib LP collection's published optimal values).
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NETLIB = os.path.join(ROOT, "instances", "netlib")


def main():
    args = sys.argv[1:]
    method = "auto"
    if "--method" in args:
        i = args.index("--method")
        method = args[i + 1]
        del args[i:i + 2]
    exe = args[0] if args else os.path.join(ROOT, "dist", "nirnay-windows-x86_64.exe" if os.name == "nt" else "nirnay-linux-x86_64")
    tol = 1e-3 if method == "pdhg" else 1e-6
    ok = True
    with open(os.path.join(NETLIB, "EXPECTED.txt")) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, ref = line.split()[:2]
            ref = float(ref)
            out = subprocess.run([exe, "solve", os.path.join(NETLIB, name + ".mps"), "--method", method],
                                 capture_output=True, text=True).stdout
            st = re.search(r"^Status\s*:\s*(\S+)", out, re.M)
            ob = re.search(r"^Objective\s*:\s*(\S+)", out, re.M)
            tm = re.search(r"^Time\s*:\s*(\S+)", out, re.M)
            status = st.group(1) if st else "NO_OUTPUT"
            obj = float(ob.group(1)) if ob else float("nan")
            err = abs(obj - ref) / max(1.0, abs(ref))
            good = status == "OPTIMAL" and err <= tol
            ok &= good
            print(f"{'PASS' if good else 'FAIL'}  {name:10s} {status:10s} obj {obj:+.10e}  ref {ref:+.10e}  "
                  f"rel.err {err:.1e}  time {tm.group(1) if tm else '?'}")
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
