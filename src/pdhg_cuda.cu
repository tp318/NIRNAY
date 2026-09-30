// NIRNAY — CUDA backend for the PDHG engine.
//
// All kernels are written from scratch (no cuSPARSE / cuBLAS):
//   * CSR sparse mat-vec with a row-group size chosen from the average row length
//   * fused projected primal / proximal dual steps with on-the-fly reductions
//   * fused KKT-residual kernels (original-space weights) for termination checks
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "pdhg.hpp"

namespace nirnay {

namespace {

#define CUDA_OK(x)                                                                      \
    do {                                                                                \
        cudaError_t e_ = (x);                                                           \
        if (e_ != cudaSuccess) {                                                        \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_),   \
                         __FILE__, __LINE__);                                           \
            std::abort();                                                               \
        }                                                                               \
    } while (0)

constexpr int kBlock = 256;

__device__ inline double warp_sum(double v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// block-wide reduction of up to 4 values, result atomically added to out[0..k)
template <int K>
__device__ inline void block_reduce_add(double (&v)[K], double* out) {
    __shared__ double sh[K][kBlock / 32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
#pragma unroll
    for (int k = 0; k < K; ++k) {
        double s = warp_sum(v[k]);
        if (lane == 0) sh[k][wid] = s;
    }
    __syncthreads();
    if (wid == 0) {
#pragma unroll
        for (int k = 0; k < K; ++k) {
            double s = (lane < kBlock / 32) ? sh[k][lane] : 0.0;
            s = warp_sum(s);
            if (lane == 0) atomicAdd(out + k, s);
        }
    }
}

// block max of a non-negative double, atomically merged into *out (bit pattern order == value order)
__device__ inline void block_max_atomic(double v, unsigned long long* out) {
    for (int o = 16; o > 0; o >>= 1) v = fmax(v, __shfl_down_sync(0xffffffffu, v, o));
    __shared__ double sm[kBlock / 32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) sm[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = lane < kBlock / 32 ? sm[lane] : 0.0;
        for (int o = 16; o > 0; o >>= 1) v = fmax(v, __shfl_down_sync(0xffffffffu, v, o));
        if (lane == 0) atomicMax(out, (unsigned long long)__double_as_longlong(v));
    }
}

// CSR SpMV: TPR threads cooperate on one row.
template <int TPR>
__global__ void k_spmv(int nrows, const int* __restrict__ rp, const int* __restrict__ ci,
                       const double* __restrict__ v, const double* __restrict__ x, double* __restrict__ y) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = gid / TPR;
    const int lane = gid % TPR;
    double s = 0.0;
    if (row < nrows) {
        for (int p = rp[row] + lane; p < rp[row + 1]; p += TPR) s += v[p] * __ldg(x + ci[p]);
    }
    if (TPR > 1) {
#pragma unroll
        for (int o = TPR / 2; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o, TPR);
    }
    if (row < nrows && lane == 0) y[row] = s;
}

__global__ void k_primal_step(int n, double tau, const double* __restrict__ x, const double* __restrict__ c,
                              const double* __restrict__ KTy, const double* __restrict__ l,
                              const double* __restrict__ u, double* __restrict__ xn, double* sums) {
    double acc[1] = {0.0};
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x) {
        double v = x[j] - tau * (c[j] - KTy[j]);
        v = fmin(fmax(v, l[j]), u[j]);
        xn[j] = v;
        double t = v - x[j];
        acc[0] += t * t;
    }
    block_reduce_add<1>(acc, sums);
}

__global__ void k_dual_step(int m, double sigma, const double* __restrict__ y, const double* __restrict__ Kx,
                            const double* __restrict__ Kxn, const double* __restrict__ rl,
                            const double* __restrict__ ru, double* __restrict__ yn, double* sums) {
    double acc[2] = {0.0, 0.0};
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += gridDim.x * blockDim.x) {
        double kbar = 2.0 * Kxn[i] - Kx[i];
        double v = y[i] - sigma * kbar;
        double lo = rl[i], up = ru[i];
        double a = isinf(lo) ? 0.0 : fmax(v + sigma * lo, 0.0);
        double b = isinf(up) ? 0.0 : fmin(v + sigma * up, 0.0);
        double w = a + b;
        yn[i] = w;
        double t = w - y[i];
        acc[0] += t * t;
        acc[1] += t * (Kxn[i] - Kx[i]);
    }
    block_reduce_add<2>(acc, sums);
}

__global__ void k_axpy2(int n, double a, const double* __restrict__ x, double* __restrict__ s) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x) s[j] += a * x[j];
}

__global__ void k_scale(int n, double a, const double* __restrict__ x, double* __restrict__ out) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x) out[j] = a * x[j];
}

__global__ void k_dist2(int n, const double* __restrict__ a, const double* __restrict__ b, double* sums) {
    double acc[1] = {0.0};
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x) {
        double t = a[j] - b[j];
        acc[0] += t * t;
    }
    block_reduce_add<1>(acc, sums);
}

__global__ void k_kkt_rows(int m, const double* __restrict__ Kx, const double* __restrict__ y,
                           const double* __restrict__ rl, const double* __restrict__ ru,
                           const double* __restrict__ wr, double* sums, unsigned long long* pmax) {
    double acc[3] = {0.0, 0.0, 0.0};
    double vmax = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += gridDim.x * blockDim.x) {
        double k = Kx[i];
        double proj = fmin(fmax(k, rl[i]), ru[i]);
        double pr = k - proj;
        acc[0] += pr * pr;
        double po = pr * wr[i];
        acc[1] += po * po;
        if (pr != 0.0) vmax = fmax(vmax, fabs(po) / (1.0 + fabs(proj * wr[i])));  // per-row relative, original units
        double yi = y[i];
        if (yi > 0 && !isinf(rl[i])) acc[2] += yi * rl[i];
        else if (yi < 0 && !isinf(ru[i])) acc[2] += yi * ru[i];
    }
    block_reduce_add<3>(acc, sums);
    block_max_atomic(vmax, pmax);
}

__global__ void k_kkt_cols(int n, const double* __restrict__ x, const double* __restrict__ KTy,
                           const double* __restrict__ c, const double* __restrict__ l,
                           const double* __restrict__ u, const double* __restrict__ wc, double* sums,
                           unsigned long long* dmax) {
    double acc[4] = {0.0, 0.0, 0.0, 0.0};
    double vmax = 0.0;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x) {
        double lam = c[j] - KTy[j];
        bool fl = !isinf(l[j]), fu = !isinf(u[j]);
        double lp = (fl && fu) ? lam : fl ? fmax(lam, 0.0) : fu ? fmin(lam, 0.0) : 0.0;
        double dr = lam - lp;
        acc[0] += dr * dr;
        double dor = dr * wc[j];
        acc[1] += dor * dor;
        if (dr != 0.0) vmax = fmax(vmax, fabs(dor) / (1.0 + fabs(c[j] * wc[j])));  // per-column relative
        acc[2] += c[j] * x[j];
        if (lp > 0) acc[3] += lp * l[j];
        else if (lp < 0) acc[3] += lp * u[j];
    }
    block_reduce_add<4>(acc, sums);
    block_max_atomic(vmax, dmax);
}

// ------------------------------------------------------------------ Halpern PDHG kernels
//
// The Halpern path never stores the scaled matrix K = diag(Dr) A diag(Dc). It keeps the raw A in
// the smallest exact format and applies the diagonal scaling where it is free:
//   K v  = Dr .* (A  (Dc .* v))   -- Dc .* v is written by the kernel that produces v
//   K'w  = Dc .* (A' (Dr .* w))   -- Dr .* w likewise
// so an SpMV streams 4 bytes per nonzero for +-1 matrices (sign in the index's top bit) or 8 bytes
// for fp32-exact values, versus 12 bytes for double values.
enum MatFmt { kF64 = 0, kF32 = 1, kPat = 2 };

// Iteration kernels are templated on the vector type TV (double, or float in the first phase of
// mixed precision). Arithmetic is done in TV as well: on consumer GPUs FP64 math runs at 1/64 of the
// FP32 rate (RTX 3050: ~70 GFLOP/s), so float storage with double math just moves the bottleneck
// from memory bandwidth to the FP64 units (measured: primal kernel 546 us -> 408 us, not ~270 us).
template <int FMT, typename TV>
__device__ inline TV row_dot(int beg, int end, int lane, int TPR, const int* __restrict__ ci,
                             const double* __restrict__ v64, const float* __restrict__ v32,
                             const TV* __restrict__ x) {
    TV s = 0;
    for (int p = beg + lane; p < end; p += TPR) {
        if (FMT == kPat) {
            const unsigned c = (unsigned)ci[p];
            const TV xv = __ldg(x + (c & 0x7fffffffu));
            s += (c >> 31) ? -xv : xv;
        } else if (FMT == kF32) {
            s += (TV)v32[p] * __ldg(x + ci[p]);
        } else {
            s += (TV)v64[p] * __ldg(x + ci[p]);
        }
    }
    return s;
}

template <typename T>
__device__ inline T shfl_sum(T s, int TPR) {
    for (int o = TPR / 2; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o, TPR);
    return s;
}

template <int TPR, int FMT, typename TV>
__global__ void k_spmv_c(int nrows, const int* __restrict__ rp, const int* __restrict__ ci,
                         const double* __restrict__ v64, const float* __restrict__ v32,
                         const TV* __restrict__ x, const float* __restrict__ rowscale, TV* __restrict__ y) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = gid / TPR;
    const int lane = gid % TPR;
    TV s = row < nrows ? row_dot<FMT, TV>(rp[row], rp[row + 1], lane, TPR, ci, v64, v32, x) : (TV)0;
    if (TPR > 1) s = shfl_sum(s, TPR);
    if (row < nrows && lane == 0) y[row] = s * (TV)rowscale[row];
}

__global__ void k_set_counter(long long* dk, long long v) { *dk = v; }

template <typename R>
__device__ inline R clampv(R v, R lo, R up) { return v < lo ? lo : (v > up ? up : v); }

// Primal update for one coordinate, given (K'y)_j.   par[0] = tau, par[1] = sigma
template <bool MAJOR, typename TV>
__device__ inline void primal_update(int j, TV kty, TV tau, TV w, TV g, TV* __restrict__ x,
                                     const TV* __restrict__ x0, const TV* __restrict__ c, const TV* __restrict__ l,
                                     const TV* __restrict__ u, TV dcj, TV* __restrict__ xbar_s,
                                     TV* __restrict__ xh, TV* __restrict__ dxh) {
    const TV xj = x[j];
    const TV h = clampv<TV>(xj - tau * (c[j] - kty), l[j], u[j]);
    const TV r = (TV)2 * h - xj;
    xbar_s[j] = dcj * r;
    if (MAJOR) { xh[j] = h; dxh[j] = h - xj; }
    x[j] = w * (g * r + ((TV)1 - g) * xj) + ((TV)1 - w) * x0[j];
}

template <bool MAJOR, typename TV>
__device__ inline void dual_update(int i, TV kxbar, TV sigma, TV w, TV g, TV* __restrict__ y,
                                   const TV* __restrict__ y0, const TV* __restrict__ rl, const TV* __restrict__ ru,
                                   TV dri, TV* __restrict__ y_s, TV* __restrict__ yh, TV* __restrict__ dyh) {
    const TV yi = y[i];
    const TV v = yi - sigma * kxbar;
    const TV lo = rl[i], up = ru[i];
    const TV vl = v + sigma * lo, vu = v + sigma * up;
    const TV a = isinf(lo) ? (TV)0 : (vl > (TV)0 ? vl : (TV)0);
    const TV b = isinf(up) ? (TV)0 : (vu < (TV)0 ? vu : (TV)0);
    const TV h = a + b;
    const TV r = (TV)2 * h - yi;
    if (MAJOR) { yh[i] = h; dyh[i] = h - yi; }
    const TV yn = w * (g * r + ((TV)1 - g) * yi) + ((TV)1 - w) * y0[i];
    y[i] = yn;
    y_s[i] = dri * yn;
}

template <typename TV>
__device__ inline TV halpern_w(const long long* dk, int off) {
    const double k = (double)(*dk + off);
    return (TV)((k + 1.0) / (k + 2.0));
}

template <bool MAJOR, typename TV>
__global__ void k_h_primal(int n, const double* __restrict__ par, const long long* __restrict__ dk, int off, double g,
                           TV* __restrict__ x, const TV* __restrict__ x0, const TV* __restrict__ c,
                           const TV* __restrict__ KTy, const TV* __restrict__ l, const TV* __restrict__ u,
                           const float* __restrict__ Dc, TV* __restrict__ xbar_s, TV* __restrict__ xh,
                           TV* __restrict__ dxh) {
    const TV tau = (TV)par[0], w = halpern_w<TV>(dk, off);
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x)
        primal_update<MAJOR, TV>(j, KTy[j], tau, w, (TV)g, x, x0, c, l, u, (TV)Dc[j], xbar_s, xh, dxh);
}

template <bool MAJOR, typename TV>
__global__ void k_h_dual(int m, const double* __restrict__ par, const long long* __restrict__ dk, int off, double g,
                         TV* __restrict__ y, const TV* __restrict__ y0, const TV* __restrict__ Kxbar,
                         const TV* __restrict__ rl, const TV* __restrict__ ru, const float* __restrict__ Dr,
                         TV* __restrict__ y_s, TV* __restrict__ yh, TV* __restrict__ dyh) {
    const TV sigma = (TV)par[1], w = halpern_w<TV>(dk, off);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += gridDim.x * blockDim.x)
        dual_update<MAJOR, TV>(i, Kxbar[i], sigma, w, (TV)g, y, y0, rl, ru, (TV)Dr[i], y_s, yh, dyh);
}

// ---- fused SpMV + update (NIRNAY). One TPR-thread group owns column j (row j of A'): it forms
// (K'y)_j = Dc_j * sum_i A_ij (Dr_i y_i) and immediately applies the projected primal step, so
// K'y is never written to or re-read from memory, and the iteration needs 2 launches instead of 4.
template <int TPR, int FMT, bool MAJOR, typename TV>
__global__ void k_fused_primal(int n, const int* __restrict__ rp, const int* __restrict__ ci,
                               const double* __restrict__ v64, const float* __restrict__ v32,
                               const TV* __restrict__ ys, const double* __restrict__ par,
                               const long long* __restrict__ dk, int off, double g, TV* __restrict__ x,
                               const TV* __restrict__ x0, const TV* __restrict__ c, const TV* __restrict__ l,
                               const TV* __restrict__ u, const float* __restrict__ Dc, TV* __restrict__ xbar_s,
                               TV* __restrict__ xh, TV* __restrict__ dxh) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = gid / TPR, lane = gid % TPR;
    TV s = j < n ? row_dot<FMT, TV>(rp[j], rp[j + 1], lane, TPR, ci, v64, v32, ys) : (TV)0;
    if (TPR > 1) s = shfl_sum(s, TPR);
    if (j >= n || lane != 0) return;
    const TV dcj = (TV)Dc[j];
    primal_update<MAJOR, TV>(j, dcj * s, (TV)par[0], halpern_w<TV>(dk, off), (TV)g, x, x0, c, l, u, dcj, xbar_s, xh, dxh);
}

template <int TPR, int FMT, bool MAJOR, typename TV>
__global__ void k_fused_dual(int m, const int* __restrict__ rp, const int* __restrict__ ci,
                             const double* __restrict__ v64, const float* __restrict__ v32,
                             const TV* __restrict__ xbar_s, const double* __restrict__ par,
                             const long long* __restrict__ dk, int off, double g, TV* __restrict__ y,
                             const TV* __restrict__ y0, const TV* __restrict__ rl, const TV* __restrict__ ru,
                             const float* __restrict__ Dr, TV* __restrict__ y_s, TV* __restrict__ yh,
                             TV* __restrict__ dyh) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = gid / TPR, lane = gid % TPR;
    TV s = i < m ? row_dot<FMT, TV>(rp[i], rp[i + 1], lane, TPR, ci, v64, v32, xbar_s) : (TV)0;
    if (TPR > 1) s = shfl_sum(s, TPR);
    if (i >= m || lane != 0) return;
    const TV dri = (TV)Dr[i];
    dual_update<MAJOR, TV>(i, dri * s, (TV)par[1], halpern_w<TV>(dk, off), (TV)g, y, y0, rl, ru, dri, y_s, yh, dyh);
}

template <typename TA, typename TB>
__global__ void k_convert(int n, const TA* __restrict__ a, TB* __restrict__ b) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x) b[j] = (TB)a[j];
}

template <typename TV>
__global__ void k_mul(int n, const double* __restrict__ a, const TV* __restrict__ b, TV* __restrict__ out) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x)
        out[j] = (TV)(a[j] * (double)b[j]);
}

template <typename TV>
__global__ void k_dist2_t(int n, const TV* __restrict__ a, const TV* __restrict__ b, double* sums) {
    double acc[1] = {0.0};
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x) {
        double t = (double)a[j] - (double)b[j];
        acc[0] += t * t;
    }
    block_reduce_add<1>(acc, sums);
}

// sums[0] += |a|^2, sums[1] += a.b
__global__ void k_norm_dot(int n, const double* __restrict__ a, const double* __restrict__ b, double* sums) {
    double acc[2] = {0.0, 0.0};
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x) {
        const double t = a[j];
        acc[0] += t * t;
        if (b) acc[1] += t * b[j];
    }
    block_reduce_add<2>(acc, sums);
}

template <typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    CUDA_OK(cudaMalloc(&p, std::max<size_t>(1, n) * sizeof(T)));
    return p;
}
template <typename T>
T* dupload(const std::vector<T>& v) {
    T* p = dalloc<T>(v.size());
    if (!v.empty()) CUDA_OK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

class CudaBackend : public PdhgBackend {
public:
    ~CudaBackend() override {
        for (auto& kv : graphs_) cudaGraphExecDestroy(kv.second);
        if (ev0_) cudaEventDestroy(ev0_);
        if (ev1_) cudaEventDestroy(ev1_);
        if (st_) cudaStreamDestroy(st_);
        for (void* p : allocs_) cudaFree(p);
    }
    std::string name() const override { return "gpu(" + devname_ + ")"; }

    void load(const PdhgData& d) override {
        d_ = &d;
        m_ = d.m; n_ = d.n;
        cudaDeviceProp prop;
        CUDA_OK(cudaGetDeviceProperties(&prop, 0));
        devname_ = prop.name;
        sms_ = prop.multiProcessorCount;
        Kp_ = up(d.Kp); Ki_ = up(d.Ki); Kv_ = up(d.Kx);
        KTp_ = up(d.KTp); KTi_ = up(d.KTi); KTv_ = up(d.KTx);
        c_ = up(d.c); l_ = up(d.l); u_ = up(d.u); rl_ = up(d.rl); ru_ = up(d.ru);
        wr_ = up(d.wr); wc_ = up(d.wc);
        std::vector<double> x0(n_);
        for (int j = 0; j < n_; ++j) x0[j] = std::min(std::max(0.0, d.l[j]), d.u[j]);
        x_ = up(x0); xn_ = mk(n_); xs_ = mk(n_); xl_ = up(x0);
        y_ = mk(m_); yn_ = mk(m_); ys_ = mk(m_); yl_ = mk(m_);
        Kx_ = mk(m_); Kxn_ = mk(m_); KTy_ = mk(n_);
        ta_x_ = mk(n_); ta_y_ = mk(m_); ta_Kx_ = mk(m_); ta_KTy_ = mk(n_);
        sums_ = mk(16);
        CUDA_OK(cudaMemset(y_, 0, m_ * sizeof(double)));
        CUDA_OK(cudaMemset(yl_, 0, m_ * sizeof(double)));
        CUDA_OK(cudaMemset(xs_, 0, n_ * sizeof(double)));
        CUDA_OK(cudaMemset(ys_, 0, m_ * sizeof(double)));
        CUDA_OK(cudaMemset(KTy_, 0, n_ * sizeof(double)));
        tpr_K_ = pick_tpr(d.Kp.back(), m_);
        tpr_KT_ = pick_tpr(d.KTp.back(), n_);
        spmv(Kp_, Ki_, Kv_, m_, tpr_K_, x_, Kx_);
        esum_ = 0;
        load_compressed(d);
    }

    void configure(bool compress, bool graphs) override { want_compress_ = compress; want_graphs_ = graphs; }

    // ---------------------------------------------------------- Halpern API
    double op_norm(int max_iter, double tol) override {
        std::vector<double> v0(n_);
        unsigned long long st = 0x9E3779B97F4A7C15ull;
        for (int j = 0; j < n_; ++j) {  // same pseudo-random start as the CPU backend
            st ^= st << 13; st ^= st >> 7; st ^= st << 17;
            v0[j] = 0.5 + (double)(st >> 11) * (1.0 / 9007199254740992.0);
        }
        {
            double nv = 0;
            for (double t : v0) nv += t * t;
            nv = std::sqrt(nv);
            if (nv > 0) for (double& t : v0) t /= nv;
        }
        CUDA_OK(cudaMemcpy(d64_.x, v0.data(), n_ * sizeof(double), cudaMemcpyHostToDevice));
        double s = 0;
        for (int it = 0; it < max_iter; ++it) {
            Kmul(d64_.x, d64_.Kx);          // u = K v
            KTmul(d64_.Kx, d64_.KTy);       // w = K'u
            CUDA_OK(cudaMemsetAsync(sums_, 0, 2 * sizeof(double), st_));
            k_norm_dot<<<grid(n_), kBlock, 0, st_>>>(n_, d64_.KTy, nullptr, sums_);
            double h[2];
            CUDA_OK(cudaMemcpyAsync(h, sums_, 2 * sizeof(double), cudaMemcpyDeviceToHost, st_));
            CUDA_OK(cudaStreamSynchronize(st_));
            const double nw = std::sqrt(h[0]);
            if (nw == 0) return 0;
            const double s_new = std::sqrt(nw);
            k_scale<<<grid(n_), kBlock, 0, st_>>>(n_, 1.0 / nw, d64_.KTy, d64_.x);
            const bool done = std::fabs(s_new - s) <= tol * s_new;
            s = s_new;
            if (done) break;
        }
        return s;
    }
    void h_reset() override {
        const PdhgData& d = *d_;
        fp32_ = false;
        std::vector<double> x0(n_);
        for (int j = 0; j < n_; ++j) x0[j] = std::min(std::max(0.0, d.l[j]), d.u[j]);
        for (double* p : {d64_.x, d64_.x0, d64_.xh})
            CUDA_OK(cudaMemcpy(p, x0.data(), n_ * sizeof(double), cudaMemcpyHostToDevice));
        for (double* p : {d64_.y, d64_.y0, d64_.yh, d64_.ys, d64_.dyh}) CUDA_OK(cudaMemset(p, 0, m_ * sizeof(double)));
        CUDA_OK(cudaMemset(d64_.dxh, 0, n_ * sizeof(double)));
    }
    // Mixed precision (NIRNAY): switch the iterate storage between float and double in place.
    bool set_fp32(bool on) override {
        if (on == fp32_) return true;
        if (on) {
            if (!f32_.x) alloc_f32();
            copy_state(d64_, f32_);
        } else {
            copy_state(f32_, d64_);
            k_mul<double><<<grid(m_), kBlock, 0, st_>>>(m_, hDr_, d64_.y, d64_.ys);   // y_s at full precision
        }
        fp32_ = on;
        return true;
    }
    void h_steps(double tau, double sigma) override {
        double p[2] = {tau, sigma};
        CUDA_OK(cudaMemcpyAsync(par_, p, 2 * sizeof(double), cudaMemcpyHostToDevice, st_));
        CUDA_OK(cudaStreamSynchronize(st_));
    }
    void h_run(int iters, long long k0, double g) override {
        k_set_counter<<<1, 1, 0, st_>>>(dk_, k0);
        // Runtime autotune (NIRNAY): fused and unfused kernels produce bit-identical iterates, so the
        // first two sizeable batches are timed one each way and the faster variant is kept.
        const bool timing = tune_state_ < 2 && iters >= 32;
        if (timing) {
            fuse_ = tune_state_ == 0;
            CUDA_OK(cudaEventRecord(ev0_, st_));
        }
        launch_batch(iters, g);
        if (timing) {
            CUDA_OK(cudaEventRecord(ev1_, st_));
            CUDA_OK(cudaEventSynchronize(ev1_));
            float ms = 0;
            CUDA_OK(cudaEventElapsedTime(&ms, ev0_, ev1_));
            tune_ms_[tune_state_] = ms / iters;
            if (++tune_state_ == 2) fuse_ = tune_ms_[0] <= tune_ms_[1];
        }
    }
    void launch_batch(int iters, double g) {
        if (!want_graphs_ || iters < 4) { enqueue_any(iters, g); return; }
        const int key = ((iters * 2 + (fuse_ ? 1 : 0)) * 2) + (fp32_ ? 1 : 0);
        auto it = graphs_.find(key);
        if (it == graphs_.end() || gamma_graph_ != g) {
            if (gamma_graph_ != g) { for (auto& kv : graphs_) cudaGraphExecDestroy(kv.second); graphs_.clear(); }
            gamma_graph_ = g;
            cudaGraph_t graph;
            CUDA_OK(cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal));
            enqueue_any(iters, g);
            CUDA_OK(cudaStreamEndCapture(st_, &graph));
            cudaGraphExec_t exec;
            CUDA_OK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
            CUDA_OK(cudaGraphDestroy(graph));
            it = graphs_.emplace(key, exec).first;
        }
        CUDA_OK(cudaGraphLaunch(it->second, st_));
    }
    std::string matrix_format() const override {
        static const char* nm[] = {"csr-f64", "csr-f32(exact)", "csr-pattern(+-1)"};
        std::string s = nm[fmt_];
        if (tune_state_ >= 2) s += fuse_ ? ", fused" : ", unfused";
        if (f32_.x) s += ", fp32->fp64";
        return s;
    }
    double h_fixed_point(double omega, double eta) override {
        if (fp32_) {
            k_convert<float, double><<<grid(n_), kBlock, 0, st_>>>(n_, f32_.dxh, d64_.dxh);
            k_convert<float, double><<<grid(m_), kBlock, 0, st_>>>(m_, f32_.dyh, d64_.dyh);
        }
        KTmul(d64_.dyh, hKTy_tmp_);                                                   // K' dy
        CUDA_OK(cudaMemsetAsync(sums_, 0, 4 * sizeof(double), st_));
        k_norm_dot<<<grid(n_), kBlock, 0, st_>>>(n_, d64_.dxh, hKTy_tmp_, sums_);      // |dx|^2, dx.K'dy
        k_norm_dot<<<grid(m_), kBlock, 0, st_>>>(m_, d64_.dyh, nullptr, sums_ + 2);    // |dy|^2
        double h[4];
        CUDA_OK(cudaMemcpyAsync(h, sums_, 4 * sizeof(double), cudaMemcpyDeviceToHost, st_));
        CUDA_OK(cudaStreamSynchronize(st_));
        return std::sqrt(std::max(0.0, omega * h[0] + h[2] / omega + 2.0 * eta * h[1]));
    }
    void h_kkt(PdhgKkt& o) override {
        // KKT is always evaluated in double, also during the float phase
        if (fp32_) {
            k_convert<float, double><<<grid(n_), kBlock, 0, st_>>>(n_, f32_.xh, d64_.xh);
            k_convert<float, double><<<grid(m_), kBlock, 0, st_>>>(m_, f32_.yh, d64_.yh);
        }
        Kmul(d64_.xh, hKxc_);
        KTmul(d64_.yh, hKTyc_);
        kkt_on(d64_.xh, d64_.yh, hKxc_, hKTyc_, st_, o);
    }
    void h_restart(double& dist_x, double& dist_y) override {
        if (fp32_) restart_impl(f32_, dist_x, dist_y);
        else restart_impl(d64_, dist_x, dist_y);
    }
    void h_get(std::vector<double>& x, std::vector<double>& y) override {
        if (fp32_) {
            k_convert<float, double><<<grid(n_), kBlock, 0, st_>>>(n_, f32_.xh, d64_.xh);
            k_convert<float, double><<<grid(m_), kBlock, 0, st_>>>(m_, f32_.yh, d64_.yh);
        }
        CUDA_OK(cudaStreamSynchronize(st_));
        x.resize(n_); y.resize(m_);
        CUDA_OK(cudaMemcpy(x.data(), d64_.xh, n_ * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(y.data(), d64_.yh, m_ * sizeof(double), cudaMemcpyDeviceToHost));
    }

    void step(double tau, double sigma, double& dx2, double& dy2, double& inter) override {
        CUDA_OK(cudaMemsetAsync(sums_, 0, 3 * sizeof(double)));
        k_primal_step<<<grid(n_), kBlock>>>(n_, tau, x_, c_, KTy_, l_, u_, xn_, sums_);
        spmv(Kp_, Ki_, Kv_, m_, tpr_K_, xn_, Kxn_);
        k_dual_step<<<grid(m_), kBlock>>>(m_, sigma, y_, Kx_, Kxn_, rl_, ru_, yn_, sums_ + 1);
        double h[3];
        CUDA_OK(cudaMemcpy(h, sums_, 3 * sizeof(double), cudaMemcpyDeviceToHost));
        dx2 = h[0]; dy2 = h[1]; inter = h[2];
    }

    void accept(double eta) override {
        std::swap(x_, xn_); std::swap(y_, yn_); std::swap(Kx_, Kxn_);
        spmv(KTp_, KTi_, KTv_, n_, tpr_KT_, y_, KTy_);
        k_axpy2<<<grid(n_), kBlock>>>(n_, eta, x_, xs_);
        k_axpy2<<<grid(m_), kBlock>>>(m_, eta, y_, ys_);
        esum_ += eta;
    }

    void kkt(bool average, PdhgKkt& o) override {
        const PdhgData& d = *d_;
        double *x = x_, *y = y_, *Kx = Kx_, *KTy = KTy_;
        if (average && esum_ > 0) {
            make_average();
            x = ta_x_; y = ta_y_; Kx = ta_Kx_; KTy = ta_KTy_;
        }
        CUDA_OK(cudaMemsetAsync(sums_ + 3, 0, 7 * sizeof(double)));
        CUDA_OK(cudaMemsetAsync(sums_ + 14, 0, 2 * sizeof(double)));
        k_kkt_rows<<<grid(m_), kBlock>>>(m_, Kx, y, rl_, ru_, wr_, sums_ + 3, (unsigned long long*)(sums_ + 14));
        k_kkt_cols<<<grid(n_), kBlock>>>(n_, x, KTy, c_, l_, u_, wc_, sums_ + 6, (unsigned long long*)(sums_ + 15));
        double h[7];
        CUDA_OK(cudaMemcpy(h, sums_ + 3, 7 * sizeof(double), cudaMemcpyDeviceToHost));
        const double p2 = h[0], p2o = h[1], drow = h[2], d2 = h[3], d2o = h[4], pobj = h[5], dcol = h[6];
        const double dobj = drow + dcol;
        o.err_p2 = p2; o.err_d2 = d2; o.gap_s = std::fabs(pobj - dobj);
        o.pobj = pobj * d.obj_unscale; o.dobj = dobj * d.obj_unscale;
        o.rel_p = std::sqrt(p2o) / (1.0 + d.bnorm);
        o.rel_d = std::sqrt(d2o) / (1.0 + d.cnorm);
        o.rel_gap = std::fabs(o.pobj - o.dobj) / (1.0 + std::fabs(o.pobj) + std::fabs(o.dobj));
    }

    void restart(bool to_average, double& dist_x, double& dist_y) override {
        if (to_average && esum_ > 0) {
            make_average();
            CUDA_OK(cudaMemcpy(x_, ta_x_, n_ * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_OK(cudaMemcpy(y_, ta_y_, m_ * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_OK(cudaMemcpy(Kx_, ta_Kx_, m_ * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_OK(cudaMemcpy(KTy_, ta_KTy_, n_ * sizeof(double), cudaMemcpyDeviceToDevice));
        }
        CUDA_OK(cudaMemsetAsync(sums_ + 10, 0, 2 * sizeof(double)));
        k_dist2<<<grid(n_), kBlock>>>(n_, x_, xl_, sums_ + 10);
        k_dist2<<<grid(m_), kBlock>>>(m_, y_, yl_, sums_ + 11);
        double h[2];
        CUDA_OK(cudaMemcpy(h, sums_ + 10, 2 * sizeof(double), cudaMemcpyDeviceToHost));
        dist_x = std::sqrt(h[0]); dist_y = std::sqrt(h[1]);
        CUDA_OK(cudaMemcpy(xl_, x_, n_ * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_OK(cudaMemcpy(yl_, y_, m_ * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_OK(cudaMemset(xs_, 0, n_ * sizeof(double)));
        CUDA_OK(cudaMemset(ys_, 0, m_ * sizeof(double)));
        esum_ = 0;
    }

    void get(bool average, std::vector<double>& x, std::vector<double>& y) override {
        double *px = x_, *py = y_;
        if (average && esum_ > 0) { make_average(); px = ta_x_; py = ta_y_; }
        x.resize(n_); y.resize(m_);
        CUDA_OK(cudaMemcpy(x.data(), px, n_ * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(y.data(), py, m_ * sizeof(double), cudaMemcpyDeviceToHost));
    }

private:
    // ---- Halpern state
    template <typename T>
    struct HState {
        T *x = nullptr, *x0 = nullptr, *xh = nullptr, *dxh = nullptr, *xbs = nullptr, *KTy = nullptr;
        T *y = nullptr, *y0 = nullptr, *yh = nullptr, *dyh = nullptr, *ys = nullptr, *Kx = nullptr;
        const T *c = nullptr, *l = nullptr, *u = nullptr, *rl = nullptr, *ru = nullptr;   // scaled problem data
    };
    HState<double> d64_;
    HState<float> f32_;          // allocated on first use of mixed precision
    bool fp32_ = false;
    bool want_compress_ = true, want_graphs_ = true;
    int fmt_ = kF64;
    int *Ap_ = nullptr, *Ai_ = nullptr, *ATp_ = nullptr, *ATi_ = nullptr;
    double *Av64_ = nullptr, *ATv64_ = nullptr;
    float *Av32_ = nullptr, *ATv32_ = nullptr;
    double *hDr_ = nullptr, *hDc_ = nullptr, *hKTy_tmp_ = nullptr, *hKxc_ = nullptr, *hKTyc_ = nullptr;
    double *hvn_ = nullptr, *hvm_ = nullptr, *par_ = nullptr;
    float *fDr_ = nullptr, *fDc_ = nullptr;
    bool fuse_ = true;
    int tune_state_ = 0;
    float tune_ms_[2] = {0, 0};
    cudaEvent_t ev0_ = nullptr, ev1_ = nullptr;
    long long* dk_ = nullptr;
    cudaStream_t st_ = nullptr;
    std::map<int, cudaGraphExec_t> graphs_;
    double gamma_graph_ = -1;

    void load_compressed(const PdhgData& d) {
        CUDA_OK(cudaStreamCreate(&st_));
        const size_t nnz = d.Araw.size();
        bool pat = want_compress_, f32 = want_compress_;
        for (size_t p = 0; p < nnz && (pat || f32); ++p) {
            const double v = d.Araw[p];
            if (std::fabs(v) != 1.0) pat = false;
            if ((double)(float)v != v) f32 = false;
        }
        if (n_ >= (1 << 30) || m_ >= (1 << 30)) pat = false;
        fmt_ = pat ? kPat : (f32 ? kF32 : kF64);
        Ap_ = up(d.Kp); ATp_ = up(d.KTp);
        if (fmt_ == kPat) {
            std::vector<int> ci(nnz), cti(nnz);
            for (size_t p = 0; p < nnz; ++p) ci[p] = d.Ki[p] | (d.Araw[p] < 0 ? (int)0x80000000u : 0);
            for (size_t p = 0; p < nnz; ++p) cti[p] = d.KTi[p] | (d.ATraw[p] < 0 ? (int)0x80000000u : 0);
            Ai_ = up(ci); ATi_ = up(cti);
        } else {
            Ai_ = up(d.Ki); ATi_ = up(d.KTi);
            if (fmt_ == kF32) {
                std::vector<float> a(d.Araw.size()), at(d.ATraw.size());
                for (size_t p = 0; p < a.size(); ++p) a[p] = (float)d.Araw[p];
                for (size_t p = 0; p < at.size(); ++p) at[p] = (float)d.ATraw[p];
                Av32_ = up(a); ATv32_ = up(at);
            } else {
                Av64_ = up(d.Araw); ATv64_ = up(d.ATraw);
            }
        }
        hDr_ = up(d.Dr); hDc_ = up(d.Dc);
        {
            std::vector<float> fr(d.Dr.size()), fc(d.Dc.size());
            for (size_t i = 0; i < fr.size(); ++i) fr[i] = (float)d.Dr[i];
            for (size_t j = 0; j < fc.size(); ++j) fc[j] = (float)d.Dc[j];
            fDr_ = up(fr); fDc_ = up(fc);
        }
        fuse_ = std::getenv("NIRNAY_PDHG_NOFUSE") == nullptr;
        if (const char* f = std::getenv("NIRNAY_PDHG_FUSE")) { fuse_ = f[0] == '1'; tune_state_ = 2; }
        else if (!fuse_) tune_state_ = 2;
        CUDA_OK(cudaEventCreate(&ev0_));
        CUDA_OK(cudaEventCreate(&ev1_));
        for (double** p : {&d64_.x, &d64_.x0, &d64_.xh, &d64_.dxh, &d64_.xbs, &d64_.KTy, &hKTy_tmp_, &hKTyc_, &hvn_}) *p = mk(n_);
        for (double** p : {&d64_.y, &d64_.y0, &d64_.yh, &d64_.dyh, &d64_.ys, &d64_.Kx, &hKxc_, &hvm_}) *p = mk(m_);
        d64_.c = c_; d64_.l = l_; d64_.u = u_; d64_.rl = rl_; d64_.ru = ru_;
        par_ = mk(2);
        long long* k = nullptr;
        CUDA_OK(cudaMalloc(&k, sizeof(long long)));
        allocs_.push_back(k);
        dk_ = k;
    }
    float* mkf(size_t n) { float* p = dalloc<float>(n); allocs_.push_back(p); return p; }
    void alloc_f32() {
        for (float** p : {&f32_.x, &f32_.x0, &f32_.xh, &f32_.dxh, &f32_.xbs, &f32_.KTy}) *p = mkf(n_);
        for (float** p : {&f32_.y, &f32_.y0, &f32_.yh, &f32_.dyh, &f32_.ys, &f32_.Kx}) *p = mkf(m_);
        float *c = mkf(n_), *l = mkf(n_), *u = mkf(n_), *rl = mkf(m_), *ru = mkf(m_);
        k_convert<double, float><<<grid(n_), kBlock, 0, st_>>>(n_, c_, c);
        k_convert<double, float><<<grid(n_), kBlock, 0, st_>>>(n_, l_, l);
        k_convert<double, float><<<grid(n_), kBlock, 0, st_>>>(n_, u_, u);
        k_convert<double, float><<<grid(m_), kBlock, 0, st_>>>(m_, rl_, rl);
        k_convert<double, float><<<grid(m_), kBlock, 0, st_>>>(m_, ru_, ru);
        f32_.c = c; f32_.l = l; f32_.u = u; f32_.rl = rl; f32_.ru = ru;
    }
    template <typename A, typename B>
    void copy_state(const HState<A>& a, HState<B>& b) {
        for (int k = 0; k < 4; ++k) {
            const A* src[4] = {a.x, a.x0, a.xh, a.dxh};
            B* dst[4] = {b.x, b.x0, b.xh, b.dxh};
            k_convert<A, B><<<grid(n_), kBlock, 0, st_>>>(n_, src[k], dst[k]);
        }
        for (int k = 0; k < 5; ++k) {
            const A* src[5] = {a.y, a.y0, a.yh, a.dyh, a.ys};
            B* dst[5] = {b.y, b.y0, b.yh, b.dyh, b.ys};
            k_convert<A, B><<<grid(m_), kBlock, 0, st_>>>(m_, src[k], dst[k]);
        }
    }
    template <typename T>
    void restart_impl(HState<T>& S, double& dist_x, double& dist_y) {
        CUDA_OK(cudaMemsetAsync(sums_ + 10, 0, 2 * sizeof(double), st_));
        k_dist2_t<T><<<grid(n_), kBlock, 0, st_>>>(n_, S.xh, S.x0, sums_ + 10);
        k_dist2_t<T><<<grid(m_), kBlock, 0, st_>>>(m_, S.yh, S.y0, sums_ + 11);
        double h[2];
        CUDA_OK(cudaMemcpyAsync(h, sums_ + 10, 2 * sizeof(double), cudaMemcpyDeviceToHost, st_));
        CUDA_OK(cudaStreamSynchronize(st_));
        dist_x = std::sqrt(h[0]); dist_y = std::sqrt(h[1]);
        CUDA_OK(cudaMemcpyAsync(S.x0, S.xh, n_ * sizeof(T), cudaMemcpyDeviceToDevice, st_));
        CUDA_OK(cudaMemcpyAsync(S.x, S.xh, n_ * sizeof(T), cudaMemcpyDeviceToDevice, st_));
        CUDA_OK(cudaMemcpyAsync(S.y0, S.yh, m_ * sizeof(T), cudaMemcpyDeviceToDevice, st_));
        CUDA_OK(cudaMemcpyAsync(S.y, S.yh, m_ * sizeof(T), cudaMemcpyDeviceToDevice, st_));
        k_mul<T><<<grid(m_), kBlock, 0, st_>>>(m_, hDr_, S.y, S.ys);
    }
    // raw A (rows) or A' (rows of A') times a pre-scaled vector, scaled by `rowscale` on output
    template <typename TV>
    void spmv_c(bool transpose, const TV* xin, const float* rowscale, TV* out) {
        const int rows = transpose ? n_ : m_;
        const int* rp = transpose ? ATp_ : Ap_;
        const int* ci = transpose ? ATi_ : Ai_;
        const double* v64 = transpose ? ATv64_ : Av64_;
        const float* v32 = transpose ? ATv32_ : Av32_;
        const int tpr = transpose ? tpr_KT_ : tpr_K_;
        const int blocks = (int)(((long long)rows * tpr + kBlock - 1) / kBlock);
#define NIRNAY_SPMV(T)                                                                                                    \
    switch (fmt_) {                                                                                                       \
        case kPat: k_spmv_c<T, kPat, TV><<<blocks, kBlock, 0, st_>>>(rows, rp, ci, v64, v32, xin, rowscale, out); break; \
        case kF32: k_spmv_c<T, kF32, TV><<<blocks, kBlock, 0, st_>>>(rows, rp, ci, v64, v32, xin, rowscale, out); break; \
        default:   k_spmv_c<T, kF64, TV><<<blocks, kBlock, 0, st_>>>(rows, rp, ci, v64, v32, xin, rowscale, out); break; \
    }
        switch (tpr) {
            case 1: NIRNAY_SPMV(1) break;
            case 4: NIRNAY_SPMV(4) break;
            case 8: NIRNAY_SPMV(8) break;
            case 16: NIRNAY_SPMV(16) break;
            default: NIRNAY_SPMV(32) break;
        }
#undef NIRNAY_SPMV
    }
    void Kmul(const double* v, double* out) {       // out = K v   (double; checks and power iteration)
        k_mul<double><<<grid(n_), kBlock, 0, st_>>>(n_, hDc_, v, hvn_);
        spmv_c<double>(false, hvn_, fDr_, out);
    }
    void KTmul(const double* w, double* out) {      // out = K' w
        k_mul<double><<<grid(m_), kBlock, 0, st_>>>(m_, hDr_, w, hvm_);
        spmv_c<double>(true, hvm_, fDc_, out);
    }
    template <int TPR, int FMT, typename TV>
    void fused_primal(HState<TV>& S, int it, bool major, double g) {
        const int bp = (int)(((long long)n_ * TPR + kBlock - 1) / kBlock);
        if (major)
            k_fused_primal<TPR, FMT, true, TV><<<bp, kBlock, 0, st_>>>(n_, ATp_, ATi_, ATv64_, ATv32_, S.ys, par_, dk_, it, g,
                S.x, S.x0, S.c, S.l, S.u, fDc_, S.xbs, S.xh, S.dxh);
        else
            k_fused_primal<TPR, FMT, false, TV><<<bp, kBlock, 0, st_>>>(n_, ATp_, ATi_, ATv64_, ATv32_, S.ys, par_, dk_, it, g,
                S.x, S.x0, S.c, S.l, S.u, fDc_, S.xbs, S.xh, S.dxh);
    }
    template <int TPR, int FMT, typename TV>
    void fused_dual(HState<TV>& S, int it, bool major, double g) {
        const int bd = (int)(((long long)m_ * TPR + kBlock - 1) / kBlock);
        if (major)
            k_fused_dual<TPR, FMT, true, TV><<<bd, kBlock, 0, st_>>>(m_, Ap_, Ai_, Av64_, Av32_, S.xbs, par_, dk_, it, g,
                S.y, S.y0, S.rl, S.ru, fDr_, S.ys, S.yh, S.dyh);
        else
            k_fused_dual<TPR, FMT, false, TV><<<bd, kBlock, 0, st_>>>(m_, Ap_, Ai_, Av64_, Av32_, S.xbs, par_, dk_, it, g,
                S.y, S.y0, S.rl, S.ru, fDr_, S.ys, S.yh, S.dyh);
    }
    template <int FMT, typename TV>
    void fused_iter(HState<TV>& S, int it, bool major, double g) {
        switch (tpr_KT_) {
            case 1: fused_primal<1, FMT, TV>(S, it, major, g); break;
            case 4: fused_primal<4, FMT, TV>(S, it, major, g); break;
            case 8: fused_primal<8, FMT, TV>(S, it, major, g); break;
            case 16: fused_primal<16, FMT, TV>(S, it, major, g); break;
            default: fused_primal<32, FMT, TV>(S, it, major, g); break;
        }
        switch (tpr_K_) {
            case 1: fused_dual<1, FMT, TV>(S, it, major, g); break;
            case 4: fused_dual<4, FMT, TV>(S, it, major, g); break;
            case 8: fused_dual<8, FMT, TV>(S, it, major, g); break;
            case 16: fused_dual<16, FMT, TV>(S, it, major, g); break;
            default: fused_dual<32, FMT, TV>(S, it, major, g); break;
        }
    }
    void enqueue_any(int iters, double g) {
        if (fp32_) enqueue(f32_, iters, g);
        else enqueue(d64_, iters, g);
    }
    template <typename TV>
    void enqueue(HState<TV>& S, int iters, double g) {
        for (int it = 0; it < iters; ++it) {
            const bool major = it == iters - 1;
            if (fuse_) {
                switch (fmt_) {
                    case kPat: fused_iter<kPat, TV>(S, it, major, g); break;
                    case kF32: fused_iter<kF32, TV>(S, it, major, g); break;
                    default: fused_iter<kF64, TV>(S, it, major, g); break;
                }
                continue;
            }
            spmv_c<TV>(true, S.ys, fDc_, S.KTy);                          // K'y
            if (major)
                k_h_primal<true, TV><<<grid(n_), kBlock, 0, st_>>>(n_, par_, dk_, it, g, S.x, S.x0, S.c, S.KTy, S.l, S.u, fDc_, S.xbs, S.xh, S.dxh);
            else
                k_h_primal<false, TV><<<grid(n_), kBlock, 0, st_>>>(n_, par_, dk_, it, g, S.x, S.x0, S.c, S.KTy, S.l, S.u, fDc_, S.xbs, S.xh, S.dxh);
            spmv_c<TV>(false, S.xbs, fDr_, S.Kx);                         // K xbar
            if (major)
                k_h_dual<true, TV><<<grid(m_), kBlock, 0, st_>>>(m_, par_, dk_, it, g, S.y, S.y0, S.Kx, S.rl, S.ru, fDr_, S.ys, S.yh, S.dyh);
            else
                k_h_dual<false, TV><<<grid(m_), kBlock, 0, st_>>>(m_, par_, dk_, it, g, S.y, S.y0, S.Kx, S.rl, S.ru, fDr_, S.ys, S.yh, S.dyh);
        }
    }
    void kkt_on(const double* x, const double* y, const double* Kx, const double* KTy, cudaStream_t s, PdhgKkt& o) {
        const PdhgData& d = *d_;
        CUDA_OK(cudaMemsetAsync(sums_ + 3, 0, 7 * sizeof(double), s));
        CUDA_OK(cudaMemsetAsync(sums_ + 14, 0, 2 * sizeof(double), s));
        k_kkt_rows<<<grid(m_), kBlock, 0, s>>>(m_, Kx, y, rl_, ru_, wr_, sums_ + 3, (unsigned long long*)(sums_ + 14));
        k_kkt_cols<<<grid(n_), kBlock, 0, s>>>(n_, x, KTy, c_, l_, u_, wc_, sums_ + 6, (unsigned long long*)(sums_ + 15));
        double h[13];
        CUDA_OK(cudaMemcpyAsync(h, sums_ + 3, 13 * sizeof(double), cudaMemcpyDeviceToHost, s));
        CUDA_OK(cudaStreamSynchronize(s));
        o.rel_p_inf = h[11]; o.rel_d_inf = h[12];
        const double p2 = h[0], p2o = h[1], drow = h[2], d2 = h[3], d2o = h[4], pobj = h[5], dcol = h[6];
        const double dobj = drow + dcol;
        o.err_p2 = p2; o.err_d2 = d2; o.gap_s = std::fabs(pobj - dobj);
        o.pobj = pobj * d.obj_unscale; o.dobj = dobj * d.obj_unscale;
        o.rel_p = std::sqrt(p2o) / (1.0 + d.bnorm);
        o.rel_d = std::sqrt(d2o) / (1.0 + d.cnorm);
        o.rel_gap = std::fabs(o.pobj - o.dobj) / (1.0 + std::fabs(o.pobj) + std::fabs(o.dobj));
    }

    const PdhgData* d_ = nullptr;
    int m_ = 0, n_ = 0, sms_ = 1;
    std::string devname_;
    std::vector<void*> allocs_;
    int *Kp_, *Ki_, *KTp_, *KTi_;
    double *Kv_, *KTv_, *c_, *l_, *u_, *rl_, *ru_, *wr_, *wc_;
    double *x_, *xn_, *xs_, *xl_, *y_, *yn_, *ys_, *yl_, *Kx_, *Kxn_, *KTy_;
    double *ta_x_, *ta_y_, *ta_Kx_, *ta_KTy_, *sums_;
    int tpr_K_ = 1, tpr_KT_ = 1;
    double esum_ = 0;

    template <typename T>
    T* up(const std::vector<T>& v) { T* p = dupload(v); allocs_.push_back(p); return p; }
    double* mk(size_t n) { double* p = dalloc<double>(n); allocs_.push_back(p); return p; }
    int grid(int n) const { return std::max(1, std::min((n + kBlock - 1) / kBlock, 32 * sms_)); }

    static int pick_tpr(long long nnz, int rows) {
        double avg = rows > 0 ? (double)nnz / rows : 1;
        if (avg <= 3) return 1;
        if (avg <= 6) return 4;
        if (avg <= 16) return 8;
        if (avg <= 48) return 16;
        return 32;
    }
    void spmv(const int* rp, const int* ci, const double* v, int rows, int tpr, const double* x, double* y) {
        long long threads = (long long)rows * tpr;
        int blocks = (int)((threads + kBlock - 1) / kBlock);
        switch (tpr) {
            case 1: k_spmv<1><<<blocks, kBlock>>>(rows, rp, ci, v, x, y); break;
            case 4: k_spmv<4><<<blocks, kBlock>>>(rows, rp, ci, v, x, y); break;
            case 8: k_spmv<8><<<blocks, kBlock>>>(rows, rp, ci, v, x, y); break;
            case 16: k_spmv<16><<<blocks, kBlock>>>(rows, rp, ci, v, x, y); break;
            default: k_spmv<32><<<blocks, kBlock>>>(rows, rp, ci, v, x, y); break;
        }
    }
    void make_average() {
        k_scale<<<grid(n_), kBlock>>>(n_, 1.0 / esum_, xs_, ta_x_);
        k_scale<<<grid(m_), kBlock>>>(m_, 1.0 / esum_, ys_, ta_y_);
        spmv(Kp_, Ki_, Kv_, m_, tpr_K_, ta_x_, ta_Kx_);
        spmv(KTp_, KTi_, KTv_, n_, tpr_KT_, ta_y_, ta_KTy_);
    }
};

}  // namespace

bool cuda_available() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return false;
    return n > 0;
}

PdhgBackend* make_cuda_backend() {
    if (!cuda_available()) return nullptr;
    return new CudaBackend();
}

}  // namespace nirnay
