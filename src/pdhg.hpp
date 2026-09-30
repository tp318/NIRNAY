// NIRNAY — restarted primal-dual hybrid gradient (PDHG) for large LPs.
//
//   min c'x  s.t.  rl <= Kx <= ru,  l <= x <= u
//
// Algorithm (after Applegate et al., "Practical LP using PDHG", 2021; Lu & Yang, cuPDLP, 2023):
//   * Ruiz (10 passes) + Pock-Chambolle diagonal preconditioning, bound/objective rescaling
//   * adaptive step size, primal weight updates at restarts
//   * adaptive restarts on the KKT error of current vs averaged iterate
//   * 2 sparse mat-vecs per iteration; everything else is vector work
//
// The iteration kernels run on a pluggable backend: CPU (OpenMP) or CUDA GPU.
// Both backends execute the identical algorithm so GPU speedups are like-for-like.
// PDHG is a moderate-accuracy method (default relative KKT tolerance 1e-4).
#pragma once
#include <string>

#include "common.hpp"
#include "model.hpp"

namespace nirnay {

struct PdhgOptions {
    double tol = 1e-4;
    int max_iter = 1000000;
    double time_limit = 1e30;
    std::string device = "auto";  // auto | cpu | gpu
    std::string alg = "halpern";  // halpern (reflected Halpern PDHG, constant step) | pdlp (adaptive step + averaging)
    double reflection = 1.0;      // Halpern reflection coefficient gamma in [0, 1]
    int check_every = 0;          // iterations between KKT checks; 0 = adaptive
    bool compress = true;         // GPU: store the matrix as pattern / fp32 when lossless
    bool graphs = true;           // GPU: replay iteration batches as CUDA graphs
    bool strict = true;           // also require worst-case (inf-norm) residuals; off when crossover follows
    std::string precision = "auto";  // GPU iterate storage: fp64 | mixed (fp32 first, then fp64) | auto
};

struct PdhgInfo {
    std::string device;
    std::string matrix_format;    // GPU matrix storage chosen by the compressor
    int restarts = 0;
    long long checks = 0;
    long long fp32_iters = 0;     // iterations run with float iterates (mixed precision)
    double ms_per_iter = 0;
    double setup_seconds = 0;
    double norm_seconds = 0;      // power iteration for ||K||
};

LpSolution solve_pdhg(const Model& mdl, const PdhgOptions& opt, PdhgInfo& info);

// ---------------------------------------------------------------- backend API
struct PdhgKkt {
    double rel_p = 0, rel_d = 0, rel_gap = 0;   // original-space relative measures
    double pobj = 0, dobj = 0;                  // original-space objectives (without offset)
    double err_p2 = 0, err_d2 = 0, gap_s = 0;   // scaled-space pieces for the restart metric
    double rel_p_inf = 0, rel_d_inf = 0;        // worst row / column violation, each relative to 1 + |own bound or cost|
};

struct PdhgData {
    int m = 0, n = 0;
    // K in CSR (rows) and K' in CSR (= K in CSC)
    std::vector<int> Kp, Ki, KTp, KTi;
    std::vector<double> Kx, KTx;
    std::vector<double> c, l, u, rl, ru;       // scaled problem
    std::vector<double> wr, wc;                // original-space residual weights
    double obj_unscale = 1.0;                  // pobj_orig = pobj_s * obj_unscale
    double bnorm = 0, cnorm = 0;               // original-space norms for relative measures
    // Unscaled matrix values in the same CSR orders, with K = diag(Dr) * A * diag(Dc).
    // Lets the GPU keep A in a compressed exact form and apply the scaling on the fly.
    std::vector<double> Araw, ATraw, Dr, Dc;
};

class PdhgBackend {
public:
    virtual ~PdhgBackend() = default;
    virtual std::string name() const = 0;
    virtual void load(const PdhgData& d) = 0;
    // Trial step from (x, y): xn, yn; returns ||dx||^2, ||dy||^2, dy'K dx
    virtual void step(double tau, double sigma, double& dx2, double& dy2, double& inter) = 0;
    // Accept the trial point; add it to the running average with weight eta.
    virtual void accept(double eta) = 0;
    virtual void kkt(bool average, PdhgKkt& out) = 0;
    // Restart at current or average point; returns distance moved since last restart.
    virtual void restart(bool to_average, double& dist_x, double& dist_y) = 0;
    virtual void get(bool average, std::vector<double>& x, std::vector<double>& y) = 0;

    // ---- reflected Halpern PDHG (cuPDLPx-style). State stays on the backend; the host
    // only reads a few scalars once per batch.
    //   zhat = PDHG(z);  z <- (k+1)/(k+2) * ((1+g) zhat - g z) + 1/(k+2) * z0
    virtual double op_norm(int max_iter, double tol) = 0;           // ||K||_2 by power iteration
    virtual void h_reset() = 0;                                      // z = z0 = feasible start
    virtual void h_steps(double tau, double sigma) = 0;
    // `iters` iterations with inner counters k0 .. k0+iters-1; the last one keeps zhat and zhat - z.
    virtual void h_run(int iters, long long k0, double gamma) = 0;
    // eta * ||zhat - z||_P^2 = omega |dx|^2 + |dy|^2 / omega + 2 eta dy'K dx  (returns the sqrt)
    virtual double h_fixed_point(double omega, double eta) = 0;
    virtual void h_kkt(PdhgKkt& out) = 0;                            // KKT at zhat
    virtual void h_restart(double& dist_x, double& dist_y) = 0;      // |zhat - z0|; z0 = z = zhat
    virtual void h_get(std::vector<double>& x, std::vector<double>& y) = 0;
    virtual std::string matrix_format() const { return "csr-f64"; }
    virtual void configure(bool /*compress*/, bool /*graphs*/) {}
    // Store iterates in float (true) or double (false); state converts in place. false if unsupported.
    virtual bool set_fp32(bool /*on*/) { return false; }
};

PdhgBackend* make_cpu_backend();
PdhgBackend* make_cuda_backend();   // nullptr if built without CUDA or no device
bool cuda_available();

}  // namespace nirnay
