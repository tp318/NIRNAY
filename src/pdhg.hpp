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
};

struct PdhgInfo {
    std::string device;
    int restarts = 0;
    double ms_per_iter = 0;
    double setup_seconds = 0;
};

LpSolution solve_pdhg(const Model& mdl, const PdhgOptions& opt, PdhgInfo& info);

// ---------------------------------------------------------------- backend API
struct PdhgKkt {
    double rel_p = 0, rel_d = 0, rel_gap = 0;   // original-space relative measures
    double pobj = 0, dobj = 0;                  // original-space objectives (without offset)
    double err_p2 = 0, err_d2 = 0, gap_s = 0;   // scaled-space pieces for the restart metric
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
};

PdhgBackend* make_cpu_backend();
PdhgBackend* make_cuda_backend();   // nullptr if built without CUDA or no device
bool cuda_available();

}  // namespace nirnay
