# NIRNAY container image.
#
#   docker build -t nirnay .
#   docker run --rm nirnay solve afiro.mps                       # bundled Netlib instance
#   docker run --rm -v "$PWD:/data" nirnay solve /data/model.mps  # your own file
#
# Stage 1 builds a fully static binary on Alpine (musl); the final image is `scratch`
# holding only that binary plus a few small benchmark instances.

FROM alpine:3.20 AS build
RUN apk add --no-cache g++ make
WORKDIR /src
COPY Makefile ./
COPY src ./src
ARG NIRNAY_VERSION=0.1.0
# OpenMP build first; fall back to single-threaded if static libgomp is unavailable.
RUN make static NIRNAY_VERSION=${NIRNAY_VERSION} \
 || (echo "!! static OpenMP link failed, building single-threaded" && make static OMP=0 NIRNAY_VERSION=${NIRNAY_VERSION})
COPY instances/netlib ./netlib
# Fail the build if the binary is not static or does not solve a known instance correctly.
RUN ! ldd dist/nirnay-linux-x86_64 2>/dev/null | grep -q '=>' \
 && ./dist/nirnay-linux-x86_64 --version \
 && ./dist/nirnay-linux-x86_64 solve netlib/afiro.mps | tee /tmp/out.txt \
 && grep -q "Status        : OPTIMAL" /tmp/out.txt \
 && grep -q "Objective     : -4.6475314" /tmp/out.txt

# `docker build --target export --output dist .` writes the bare binary to ./dist
FROM scratch AS export
COPY --from=build /src/dist/nirnay-linux-x86_64 /

FROM scratch
COPY --from=build /src/dist/nirnay-linux-x86_64 /usr/local/bin/nirnay
COPY instances/netlib/ /instances/
COPY instances/industrial/gasblend_6p.mps instances/milp/setcover_200x120.mps instances/qp/portfolio_200.mps /instances/
WORKDIR /instances
ENTRYPOINT ["/usr/local/bin/nirnay"]
CMD ["--help"]
