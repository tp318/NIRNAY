#!/bin/sh
# Build the fully static Linux binary -> dist/nirnay-linux-x86_64
#
# Uses Docker (Alpine/musl) when available, which gives a truly static binary that
# runs on any x86_64 Linux. Without Docker it falls back to `make static` with the
# local toolchain (static glibc works but is less portable; install musl-gcc if possible).
set -eu
cd "$(dirname "$0")"
VERSION="${NIRNAY_VERSION:-0.1.0}"
mkdir -p dist
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    docker build --build-arg NIRNAY_VERSION="$VERSION" --target export --output dist .
else
    echo "docker not available; building with local toolchain" >&2
    make static NIRNAY_VERSION="$VERSION" || make static OMP=0 NIRNAY_VERSION="$VERSION"
fi
chmod +x dist/nirnay-linux-x86_64
file dist/nirnay-linux-x86_64 2>/dev/null || true
dist/nirnay-linux-x86_64 --version
