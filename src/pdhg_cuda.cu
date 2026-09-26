// NIRNAY — CUDA backend for the PDHG engine.
//
// All kernels are written from scratch (no cuSPARSE / cuBLAS):
//   * CSR sparse mat-vec with a row-group size chosen from the average row length
//   * fused projected primal / proximal dual steps with on-the-fly reductions
//   * fused KKT-residual kernels (original-space weights) for termination checks
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
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
                           const double* __restrict__ wr, double* sums) {
    double acc[3] = {0.0, 0.0, 0.0};
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += gridDim.x * blockDim.x) {
        double k = Kx[i];
        double pr = k - fmin(fmax(k, rl[i]), ru[i]);
        acc[0] += pr * pr;
        double po = pr * wr[i];
        acc[1] += po * po;
        double yi = y[i];
        if (yi > 0 && !isinf(rl[i])) acc[2] += yi * rl[i];
        else if (yi < 0 && !isinf(ru[i])) acc[2] += yi * ru[i];
    }
    block_reduce_add<3>(acc, sums);
}

__global__ void k_kkt_cols(int n, const double* __restrict__ x, const double* __restrict__ KTy,
                           const double* __restrict__ c, const double* __restrict__ l,
                           const double* __restrict__ u, const double* __restrict__ wc, double* sums) {
    double acc[4] = {0.0, 0.0, 0.0, 0.0};
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x) {
        double lam = c[j] - KTy[j];
        bool fl = !isinf(l[j]), fu = !isinf(u[j]);
        double lp = (fl && fu) ? lam : fl ? fmax(lam, 0.0) : fu ? fmin(lam, 0.0) : 0.0;
        double dr = lam - lp;
        acc[0] += dr * dr;
        double dor = dr * wc[j];
        acc[1] += dor * dor;
        acc[2] += c[j] * x[j];
        if (lp > 0) acc[3] += lp * l[j];
        else if (lp < 0) acc[3] += lp * u[j];
    }
    block_reduce_add<4>(acc, sums);
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
        k_kkt_rows<<<grid(m_), kBlock>>>(m_, Kx, y, rl_, ru_, wr_, sums_ + 3);
        k_kkt_cols<<<grid(n_), kBlock>>>(n_, x, KTy, c_, l_, u_, wc_, sums_ + 6);
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
