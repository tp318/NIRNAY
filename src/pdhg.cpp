#include "pdhg.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace nirnay {

// =========================================================== CPU backend
namespace {

class CpuBackend : public PdhgBackend {
public:
    std::string name() const override {
#ifdef _OPENMP
        return "cpu(" + std::to_string(omp_get_max_threads()) + " threads)";
#else
        return "cpu";
#endif
    }
    void load(const PdhgData& d) override {
        d_ = &d;
        m_ = d.m; n_ = d.n;
        x_.assign(n_, 0.0);
        for (int j = 0; j < n_; ++j) x_[j] = std::min(std::max(0.0, d.l[j]), d.u[j]);
        y_.assign(m_, 0.0);
        xn_ = x_; yn_ = y_;
        Kx_.assign(m_, 0.0); Kxn_.assign(m_, 0.0); KTy_.assign(n_, 0.0);
        spmv_K(x_, Kx_);
        xs_.assign(n_, 0.0); ys_.assign(m_, 0.0); esum_ = 0;
        xl_ = x_; yl_ = y_;
        ta_x_.assign(n_, 0.0); ta_y_.assign(m_, 0.0); ta_Kx_.assign(m_, 0.0); ta_KTy_.assign(n_, 0.0);
    }
    void step(double tau, double sigma, double& dx2, double& dy2, double& inter) override {
        const PdhgData& d = *d_;
        double sdx = 0, sdy = 0, sin = 0;
#pragma omp parallel for reduction(+:sdx) schedule(static)
        for (int j = 0; j < n_; ++j) {
            double v = x_[j] - tau * (d.c[j] - KTy_[j]);
            v = std::min(std::max(v, d.l[j]), d.u[j]);
            xn_[j] = v;
            double t = v - x_[j];
            sdx += t * t;
        }
        spmv_K(xn_, Kxn_);
#pragma omp parallel for reduction(+:sdy,sin) schedule(static)
        for (int i = 0; i < m_; ++i) {
            double kbar = 2.0 * Kxn_[i] - Kx_[i];
            double v = y_[i] - sigma * kbar;
            double a = d.rl[i] > -kInf ? std::max(v + sigma * d.rl[i], 0.0) : 0.0;
            double b = d.ru[i] < kInf ? std::min(v + sigma * d.ru[i], 0.0) : 0.0;
            double yn = a + b;
            yn_[i] = yn;
            double t = yn - y_[i];
            sdy += t * t;
            sin += t * (Kxn_[i] - Kx_[i]);
        }
        dx2 = sdx; dy2 = sdy; inter = sin;
    }
    void accept(double eta) override {
        x_.swap(xn_); y_.swap(yn_); Kx_.swap(Kxn_);
        spmv_KT(y_, KTy_);
#pragma omp parallel for schedule(static)
        for (int j = 0; j < n_; ++j) xs_[j] += eta * x_[j];
#pragma omp parallel for schedule(static)
        for (int i = 0; i < m_; ++i) ys_[i] += eta * y_[i];
        esum_ += eta;
    }
    void kkt(bool average, PdhgKkt& o) override {
        if (average && esum_ > 0) {
            make_average();
            kkt_at(ta_x_, ta_y_, ta_Kx_, ta_KTy_, o);
        } else {
            kkt_at(x_, y_, Kx_, KTy_, o);
        }
    }

    // ---------------- reflected Halpern PDHG
    double op_norm(int max_iter, double tol) override {
        std::vector<double> v(n_), Kv(m_), w(n_);
        unsigned long long st = 0x9E3779B97F4A7C15ull;
        for (int j = 0; j < n_; ++j) {  // pseudo-random start: a uniform vector can miss the top singular vector
            st ^= st << 13; st ^= st >> 7; st ^= st << 17;
            v[j] = 0.5 + (double)(st >> 11) * (1.0 / 9007199254740992.0);
        }
        {
            double nv = 0;
            for (double t : v) nv += t * t;
            nv = std::sqrt(nv);
            if (nv > 0) for (double& t : v) t /= nv;
        }
        double s = 0;
        for (int it = 0; it < max_iter; ++it) {
            spmv_K(v, Kv);
            spmv_KT(Kv, w);
            double nw = 0;
            for (int j = 0; j < n_; ++j) nw += w[j] * w[j];
            nw = std::sqrt(nw);
            if (nw == 0) return 0;
            double s_new = std::sqrt(nw);
            for (int j = 0; j < n_; ++j) v[j] = w[j] / nw;
            if (std::fabs(s_new - s) <= tol * s_new) { s = s_new; break; }
            s = s_new;
        }
        return s;
    }
    void h_reset() override {
        const PdhgData& d = *d_;
        x0_.assign(n_, 0.0);
        for (int j = 0; j < n_; ++j) x0_[j] = std::min(std::max(0.0, d.l[j]), d.u[j]);
        y0_.assign(m_, 0.0);
        x_ = x0_; y_ = y0_;
        xh_ = x_; yh_ = y_;
        dxh_.assign(n_, 0.0); dyh_.assign(m_, 0.0);
        xbar_.assign(n_, 0.0); Kxbar_.assign(m_, 0.0);
        spmv_KT(y_, KTy_);
    }
    void h_steps(double tau, double sigma) override { tau_ = tau; sigma_ = sigma; }
    void h_run(int iters, long long k0, double g) override {
        const PdhgData& d = *d_;
        for (int it = 0; it < iters; ++it) {
            const double k = (double)(k0 + it);
            const double w = (k + 1.0) / (k + 2.0);
            const bool major = it == iters - 1;
            spmv_KT(y_, KTy_);
#pragma omp parallel for schedule(static)
            for (int j = 0; j < n_; ++j) {
                double xh = std::min(std::max(x_[j] - tau_ * (d.c[j] - KTy_[j]), d.l[j]), d.u[j]);
                double xr = 2.0 * xh - x_[j];
                xbar_[j] = xr;
                if (major) { xh_[j] = xh; dxh_[j] = xh - x_[j]; }
                x_[j] = w * (g * xr + (1.0 - g) * x_[j]) + (1.0 - w) * x0_[j];
            }
            spmv_K(xbar_, Kxbar_);
#pragma omp parallel for schedule(static)
            for (int i = 0; i < m_; ++i) {
                double v = y_[i] - sigma_ * Kxbar_[i];
                double a = d.rl[i] > -kInf ? std::max(v + sigma_ * d.rl[i], 0.0) : 0.0;
                double b = d.ru[i] < kInf ? std::min(v + sigma_ * d.ru[i], 0.0) : 0.0;
                double yh = a + b;
                double yr = 2.0 * yh - y_[i];
                if (major) { yh_[i] = yh; dyh_[i] = yh - y_[i]; }
                y_[i] = w * (g * yr + (1.0 - g) * y_[i]) + (1.0 - w) * y0_[i];
            }
        }
    }
    double h_fixed_point(double omega, double eta) override {
        std::vector<double> KTdy(n_);
        spmv_KT(dyh_, KTdy);
        double px = 0, py = 0, cr = 0;
        for (int j = 0; j < n_; ++j) { px += dxh_[j] * dxh_[j]; cr += KTdy[j] * dxh_[j]; }
        for (int i = 0; i < m_; ++i) py += dyh_[i] * dyh_[i];
        return std::sqrt(std::max(0.0, omega * px + py / omega + 2.0 * eta * cr));
    }
    void h_kkt(PdhgKkt& o) override {
        std::vector<double> Kx(m_), KTy(n_);
        spmv_K(xh_, Kx);
        spmv_KT(yh_, KTy);
        kkt_at(xh_, yh_, Kx, KTy, o);
    }
    void h_restart(double& dist_x, double& dist_y) override {
        double sx = 0, sy = 0;
        for (int j = 0; j < n_; ++j) { double t = xh_[j] - x0_[j]; sx += t * t; }
        for (int i = 0; i < m_; ++i) { double t = yh_[i] - y0_[i]; sy += t * t; }
        dist_x = std::sqrt(sx); dist_y = std::sqrt(sy);
        x0_ = xh_; y0_ = yh_; x_ = xh_; y_ = yh_;
    }
    void h_get(std::vector<double>& x, std::vector<double>& y) override { x = xh_; y = yh_; }

private:
    std::vector<double> x0_, y0_, xh_, yh_, dxh_, dyh_, xbar_, Kxbar_;
    double tau_ = 0, sigma_ = 0;

    void kkt_at(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& Kx,
                const std::vector<double>& KTy, PdhgKkt& o) const {
        const PdhgData& d = *d_;
        double p2 = 0, p2o = 0, drow = 0, d2 = 0, d2o = 0, pobj = 0, dcol = 0, pmax = 0, dmax = 0;
#pragma omp parallel for reduction(+:p2,p2o,drow) schedule(static)
        for (int i = 0; i < m_; ++i) {
            double k = Kx[i];
            double pr = k - std::min(std::max(k, d.rl[i]), d.ru[i]);
            p2 += pr * pr;
            double po = pr * d.wr[i];
            p2o += po * po;
            if (y[i] > 0 && d.rl[i] > -kInf) drow += y[i] * d.rl[i];
            else if (y[i] < 0 && d.ru[i] < kInf) drow += y[i] * d.ru[i];
        }
#pragma omp parallel for reduction(+:d2,d2o,pobj,dcol) schedule(static)
        for (int j = 0; j < n_; ++j) {
            double lam = d.c[j] - KTy[j];
            bool fl = d.l[j] > -kInf, fu = d.u[j] < kInf;
            double lp = (fl && fu) ? lam : fl ? std::max(lam, 0.0) : fu ? std::min(lam, 0.0) : 0.0;
            double dr = lam - lp;
            d2 += dr * dr;
            double dor = dr * d.wc[j];
            d2o += dor * dor;
            pobj += d.c[j] * x[j];
            if (lp > 0) dcol += lp * d.l[j];
            else if (lp < 0) dcol += lp * d.u[j];
        }
        double dobj = drow + dcol;
        o.err_p2 = p2; o.err_d2 = d2; o.gap_s = std::fabs(pobj - dobj);
        // worst single violations (serial: MSVC's OpenMP 2.0 has no max reduction; runs once per check)
        for (int i = 0; i < m_; ++i) {
            double proj = std::min(std::max(Kx[i], d.rl[i]), d.ru[i]);
            double pr = Kx[i] - proj;
            if (pr != 0.0) pmax = std::max(pmax, std::fabs(pr * d.wr[i]) / (1.0 + std::fabs(proj * d.wr[i])));
        }
        for (int j = 0; j < n_; ++j) {
            double lam = d.c[j] - KTy[j];
            bool fl = d.l[j] > -kInf, fu = d.u[j] < kInf;
            double lp = (fl && fu) ? lam : fl ? std::max(lam, 0.0) : fu ? std::min(lam, 0.0) : 0.0;
            double dr = lam - lp;
            if (dr != 0.0) dmax = std::max(dmax, std::fabs(dr * d.wc[j]) / (1.0 + std::fabs(d.c[j] * d.wc[j])));
        }
        o.rel_p_inf = pmax; o.rel_d_inf = dmax;
        o.pobj = pobj * d.obj_unscale; o.dobj = dobj * d.obj_unscale;
        o.rel_p = std::sqrt(p2o) / (1.0 + d.bnorm);
        o.rel_d = std::sqrt(d2o) / (1.0 + d.cnorm);
        o.rel_gap = std::fabs(o.pobj - o.dobj) / (1.0 + std::fabs(o.pobj) + std::fabs(o.dobj));
    }
public:
    void restart(bool to_average, double& dist_x, double& dist_y) override {
        if (to_average && esum_ > 0) {
            make_average();
            x_ = ta_x_; y_ = ta_y_; Kx_ = ta_Kx_; KTy_ = ta_KTy_;
        }
        double sx = 0, sy = 0;
        for (int j = 0; j < n_; ++j) { double t = x_[j] - xl_[j]; sx += t * t; }
        for (int i = 0; i < m_; ++i) { double t = y_[i] - yl_[i]; sy += t * t; }
        dist_x = std::sqrt(sx); dist_y = std::sqrt(sy);
        xl_ = x_; yl_ = y_;
        std::fill(xs_.begin(), xs_.end(), 0.0); std::fill(ys_.begin(), ys_.end(), 0.0); esum_ = 0;
    }
    void get(bool average, std::vector<double>& x, std::vector<double>& y) override {
        if (average && esum_ > 0) { make_average(); x = ta_x_; y = ta_y_; }
        else { x = x_; y = y_; }
    }

private:
    const PdhgData* d_ = nullptr;
    int m_ = 0, n_ = 0;
    std::vector<double> x_, y_, xn_, yn_, Kx_, Kxn_, KTy_, xs_, ys_, xl_, yl_;
    std::vector<double> ta_x_, ta_y_, ta_Kx_, ta_KTy_;
    double esum_ = 0;

    void spmv_K(const std::vector<double>& x, std::vector<double>& out) const {
        const PdhgData& d = *d_;
#pragma omp parallel for schedule(dynamic, 1024)
        for (int i = 0; i < m_; ++i) {
            double s = 0;
            for (int p = d.Kp[i]; p < d.Kp[i + 1]; ++p) s += d.Kx[p] * x[d.Ki[p]];
            out[i] = s;
        }
    }
    void spmv_KT(const std::vector<double>& y, std::vector<double>& out) const {
        const PdhgData& d = *d_;
#pragma omp parallel for schedule(dynamic, 1024)
        for (int j = 0; j < n_; ++j) {
            double s = 0;
            for (int p = d.KTp[j]; p < d.KTp[j + 1]; ++p) s += d.KTx[p] * y[d.KTi[p]];
            out[j] = s;
        }
    }
    void make_average() {
        const double inv = 1.0 / esum_;
        for (int j = 0; j < n_; ++j) ta_x_[j] = xs_[j] * inv;
        for (int i = 0; i < m_; ++i) ta_y_[i] = ys_[i] * inv;
        spmv_K(ta_x_, ta_Kx_);
        spmv_KT(ta_y_, ta_KTy_);
    }
};

}  // namespace

PdhgBackend* make_cpu_backend() { return new CpuBackend(); }

#ifndef NIRNAY_CUDA
PdhgBackend* make_cuda_backend() { return nullptr; }
bool cuda_available() { return false; }
#endif

// =========================================================== driver
LpSolution solve_pdhg(const Model& mdl, const PdhgOptions& opt, PdhgInfo& info) {
    Timer setup;
    LpSolution sol;
    const int m = mdl.m, n = mdl.n;
    PdhgData d;
    d.m = m; d.n = n;
    CscMatrix KT = mdl.A.transpose();       // CSC of A' == CSR of A
    d.Kp = KT.colptr; d.Ki = KT.rowidx; d.Kx = KT.val;
    d.KTp = mdl.A.colptr; d.KTi = mdl.A.rowidx; d.KTx = mdl.A.val;
    d.Araw = d.Kx; d.ATraw = d.KTx;

    // ---- Ruiz equilibration + Pock-Chambolle
    std::vector<double> Dr(m, 1.0), Dc(n, 1.0), rmax(m), cmax(n);
    for (int it = 0; it < 10; ++it) {
        std::fill(rmax.begin(), rmax.end(), 0.0);
        std::fill(cmax.begin(), cmax.end(), 0.0);
        for (int j = 0; j < n; ++j)
            for (int p = d.KTp[j]; p < d.KTp[j + 1]; ++p) {
                int i = d.KTi[p];
                double a = std::fabs(d.KTx[p]) * Dr[i] * Dc[j];
                rmax[i] = std::max(rmax[i], a); cmax[j] = std::max(cmax[j], a);
            }
        for (int i = 0; i < m; ++i) if (rmax[i] > 0) Dr[i] /= std::sqrt(rmax[i]);
        for (int j = 0; j < n; ++j) if (cmax[j] > 0) Dc[j] /= std::sqrt(cmax[j]);
    }
    std::fill(rmax.begin(), rmax.end(), 0.0);
    std::fill(cmax.begin(), cmax.end(), 0.0);
    for (int j = 0; j < n; ++j)
        for (int p = d.KTp[j]; p < d.KTp[j + 1]; ++p) {
            int i = d.KTi[p];
            double a = std::fabs(d.KTx[p]) * Dr[i] * Dc[j];
            rmax[i] += a; cmax[j] += a;
        }
    for (int i = 0; i < m; ++i) if (rmax[i] > 0) Dr[i] /= std::sqrt(rmax[i]);
    for (int j = 0; j < n; ++j) if (cmax[j] > 0) Dc[j] /= std::sqrt(cmax[j]);
    // Round the scaling to float so the GPU can store it in 4 bytes and stay exactly consistent.
    for (double& v : Dr) v = (double)(float)v;
    for (double& v : Dc) v = (double)(float)v;
    double kmax = 0;
    for (int j = 0; j < n; ++j)
        for (int p = d.KTp[j]; p < d.KTp[j + 1]; ++p) {
            d.KTx[p] *= Dr[d.KTi[p]] * Dc[j];
            kmax = std::max(kmax, std::fabs(d.KTx[p]));
        }
    for (int i = 0; i < m; ++i)
        for (int p = d.Kp[i]; p < d.Kp[i + 1]; ++p) d.Kx[p] *= Dr[i] * Dc[d.Ki[p]];

    // ---- bound / objective rescaling
    double cn = 0, bn = 0, cno = 0, bno = 0;
    for (int j = 0; j < n; ++j) { double v = mdl.c[j] * Dc[j]; cn += v * v; cno += mdl.c[j] * mdl.c[j]; }
    for (int i = 0; i < m; ++i) {
        double lo = mdl.row_lo[i], up = mdl.row_up[i];
        if (is_finite(lo)) { bn += lo * Dr[i] * lo * Dr[i]; bno += lo * lo; }
        if (is_finite(up) && up != lo) { bn += up * Dr[i] * up * Dr[i]; bno += up * up; }
    }
    cn = std::sqrt(cn); bn = std::sqrt(bn);
    const double cs = 1.0 / (cn + 1.0), bs = 1.0 / (bn + 1.0);
    d.c.resize(n); d.l.resize(n); d.u.resize(n); d.wc.resize(n);
    d.rl.resize(m); d.ru.resize(m); d.wr.resize(m);
    for (int j = 0; j < n; ++j) {
        d.c[j] = mdl.c[j] * Dc[j] * cs;
        d.l[j] = mdl.col_lo[j] * bs / Dc[j];
        d.u[j] = mdl.col_up[j] * bs / Dc[j];
        d.wc[j] = 1.0 / (Dc[j] * cs);
    }
    for (int i = 0; i < m; ++i) {
        d.rl[i] = mdl.row_lo[i] * Dr[i] * bs;
        d.ru[i] = mdl.row_up[i] * Dr[i] * bs;
        d.wr[i] = 1.0 / (Dr[i] * bs);
    }
    d.obj_unscale = 1.0 / (cs * bs);
    d.bnorm = std::sqrt(bno);
    d.cnorm = std::sqrt(cno);
    d.Dr = Dr; d.Dc = Dc;

    // ---- backend
    std::unique_ptr<PdhgBackend> be;
    bool want_gpu = opt.device == "gpu" || (opt.device == "auto" && mdl.A.nnz() >= 20000);
    if (want_gpu) be.reset(make_cuda_backend());
    if (!be) {
        if (opt.device == "gpu") NLOG("PDHG: CUDA backend unavailable, falling back to CPU\n");
        be.reset(make_cpu_backend());
    }
    be->configure(opt.compress, opt.graphs);
    be->load(d);
    info.device = be->name();
    info.matrix_format = be->matrix_format();
    info.setup_seconds = setup.seconds();

    if (opt.alg == "halpern") {
        bool fp32 = false;
        long long fp32_iters = 0;
        auto finish = [&](bool converged, long long k, double el, const PdhgKkt& best) {
            info.ms_per_iter = k > 0 ? 1000.0 * el / k : 0;
            info.matrix_format = be->matrix_format();
            if (fp32) fp32_iters = k;
            info.fp32_iters = fp32_iters;
            NLOG("PDHG: %lld iterations, %d restarts, %lld checks, %.3fs (%.4f ms/iter), %s\n", k, info.restarts,
                 info.checks, el, info.ms_per_iter, converged ? "converged" : "not converged");
            std::vector<double> xs, ys;
            be->h_get(xs, ys);
            sol.x.resize(n); sol.y.resize(m);
            for (int j = 0; j < n; ++j) sol.x[j] = xs[j] * Dc[j] / bs;
            for (int i = 0; i < m; ++i) sol.y[i] = ys[i] * Dr[i] / cs;
            sol.status = converged ? Status::Optimal : (el > opt.time_limit ? Status::TimeLimit : Status::IterationLimit);
            sol.iterations = (int)k;
            sol.pinf = best.rel_p; sol.dinf = best.rel_d; sol.gap = best.rel_gap;
            sol.obj = best.pobj + mdl.obj_offset;
            sol.dual_obj = best.dobj + mdl.obj_offset;
            return sol;
        };
        Timer tn;
        // Ruiz + Pock-Chambolle (alpha = 1) scaling guarantees ||K||_2 <= sqrt(||K||_1 ||K||_inf) <= 1.
        // Power iteration only gives a lower estimate, and an underestimate breaks the step condition
        // (seen as exponential divergence on presolved bore3d), so pad it and cap it at the bound.
        const double kest = be->op_norm(500, 1e-6);
        info.norm_seconds = tn.seconds();
        const double knorm = std::min(1.0, 1.02 * kest);
        const double eta = knorm > 0 ? 0.998 / knorm : 1.0;
        double omega = 1.0, best_omega = 1.0, best_gap_ratio = kInf;
        double pid_sum = 0, pid_last = 0;
        NLOG("PDHG[%s, %s, halpern g=%.2f]: %d rows, %d cols, %d nnz; setup %.2fs (||K|| %.4g in %.2fs); tol %.0e\n",
             info.device.c_str(), info.matrix_format.c_str(), opt.reflection, m, n, mdl.A.nnz(), info.setup_seconds,
             knorm, info.norm_seconds, opt.tol);
        NLOG("%9s %15s %15s %9s %9s %9s %8s\n", "iter", "primal obj", "dual obj", "rel.p", "rel.d", "rel.gap", "time");
        be->h_reset();
        // Mixed precision (NIRNAY): the first phase stores AND computes the iterates in float, while every
        // check is still evaluated in double. On consumer GPUs this halves vector traffic and escapes the
        // 1/64-rate FP64 units (RTX 3050: 1.65-1.7x faster iterations on large LPs). The state converts to
        // double in place once float stops making progress or nears its accuracy floor.
        const bool want_mixed = opt.precision == "mixed" || (opt.precision == "auto" && mdl.A.nnz() >= 250000);
        fp32 = want_mixed && be->set_fp32(true);
        double fp32_best = kInf;
        int fp32_since_best = 0;
        be->h_steps(eta / omega, eta * omega);
        Timer t;
        long long total = 0, inner = 0;
        double r_start = kInf, r_last = kInf;
        bool need_r_start = true;
        PdhgKkt kk{};
        double last_print = -1;
        // Adaptive check interval (NIRNAY): checks cost a host sync + 3 SpMVs, so space them by the
        // predicted distance to convergence, in powers of two so each size is captured as one CUDA graph.
        int batch = opt.check_every > 0 ? opt.check_every : 64;
        double kkt_prev = -1; long long it_prev = 0;
        auto metric = [&](const PdhgKkt& q) { return std::max(q.rel_p, std::max(q.rel_d, q.rel_gap)); };
        while (true) {
            if (need_r_start) {
                // fixed-point error of the first iteration of a restart epoch = restart reference
                be->h_run(1, inner, opt.reflection);
                ++inner; ++total;
                r_start = be->h_fixed_point(omega, eta);
                r_last = kInf;
                need_r_start = false;
            }
            be->h_run(batch, inner, opt.reflection);
            inner += batch; total += batch;
            const double r = be->h_fixed_point(omega, eta);
            be->h_kkt(kk);
            ++info.checks;
            const double el = t.seconds();
            if (g_verbose > 1 || (g_verbose > 0 && (el - last_print >= 1.0 || info.checks == 1))) {
                last_print = el;
                NLOG("%9lld %15.8e %15.8e %9.2e %9.2e %9.2e %7.2fs\n", total, kk.pobj + mdl.obj_offset,
                     kk.dobj + mdl.obj_offset, kk.rel_p, kk.rel_d, kk.rel_gap, el);
            }
            // Optimal only if the standard l2 KKT criterion AND the worst single row / column violation
            // meet the tolerance: the l2 norm alone passed ganges with one row violated by 0.47.
            if (kk.rel_p <= opt.tol && kk.rel_d <= opt.tol && kk.rel_gap <= opt.tol &&
                (!opt.strict || (kk.rel_p_inf <= opt.tol && kk.rel_d_inf <= opt.tol)))
                return finish(true, total, el, kk);
            if (total >= opt.max_iter || el > opt.time_limit || !std::isfinite(r)) return finish(false, total, el, kk);
            if (fp32) {
                double mt = std::max(kk.rel_p, std::max(kk.rel_d, kk.rel_gap));
                if (opt.strict) mt = std::max(mt, std::max(kk.rel_p_inf, kk.rel_d_inf));
                // Leave float when it nears its accuracy floor, or when the residual has not set a new best
                // (by 5%) for 6 checks. A window test ("no 20% gain over 4 checks") fired on the ordinary
                // early oscillation between restarts, at KKT 0.7, before float had done any useful work.
                if (mt < 0.95 * fp32_best) { fp32_best = mt; fp32_since_best = 0; }
                else ++fp32_since_best;
                const bool floor = mt < 1e-5;
                const bool stalled = fp32_since_best >= 6;
                if (floor || stalled || !std::isfinite(mt)) {
                    be->set_fp32(false);
                    fp32 = false;
                    fp32_iters = total;
                    NLOG("PDHG: float phase done after %lld iterations (%s, KKT %.1e); continuing in double\n", total,
                         floor ? "accuracy floor" : "stalled", mt);
                }
            }
            if (opt.check_every <= 0) {
                const double mcur = metric(kk);
                int want = 64;
                if (kkt_prev > 0 && mcur < kkt_prev && total > it_prev) {
                    const double rate = std::log(mcur / kkt_prev) / (double)(total - it_prev);  // < 0
                    const double left = std::log(opt.tol / mcur) / rate;                         // iterations to go
                    want = (int)std::min(1024.0, std::max(32.0, 0.5 * left));
                }
                int p2 = 32;
                while (p2 * 2 <= want) p2 *= 2;
                batch = p2;
                kkt_prev = mcur; it_prev = total;
            }
            // cuPDLPx adaptive restart on the fixed-point residual
            bool restart = false;
            if (r <= 0.2 * r_start) restart = true;
            else if (r <= 0.5 * r_start && r > r_last) restart = true;
            else if (inner >= 0.36 * total) restart = true;
            r_last = r;
            if (restart) {
                double dx, dy;
                be->h_restart(dx, dy);
                const double ratio = kk.rel_p > 0 ? kk.rel_d / kk.rel_p : 0;
                if (dx > 1e-16 && dy > 1e-16 && dx < 1e12 && dy < 1e12 && ratio > 1e-8 && ratio < 1e8) {
                    const double e = std::log(dy) - std::log(dx) - std::log(omega);  // PID on log primal weight
                    pid_sum = 0.3 * pid_sum + e;
                    omega *= std::exp(0.99 * e + 0.01 * pid_sum + 0.0 * (e - pid_last));
                    pid_last = e;
                } else {
                    omega = best_omega; pid_sum = 0; pid_last = 0;
                }
                if (ratio > 0) {
                    const double g = std::fabs(std::log10(ratio));
                    if (g < best_gap_ratio) { best_gap_ratio = g; best_omega = omega; }
                }
                be->h_steps(eta / omega, eta * omega);
                inner = 0;
                need_r_start = true;
                ++info.restarts;
            }
        }
    }

    double omega = (cn * cs > 1e-10 && bn * bs > 1e-10) ? (cn * cs) / (bn * bs) : 1.0;
    double eta = 1.0 / std::max(kmax, 1e-12);
    NLOG("PDHG[%s]: %d rows, %d cols, %d nnz; setup %.2fs; tol %.0e\n", info.device.c_str(), m, n,
         mdl.A.nnz(), info.setup_seconds, opt.tol);
    NLOG("%9s %15s %15s %9s %9s %9s %8s\n", "iter", "primal obj", "dual obj", "rel.p", "rel.d", "rel.gap", "time");

    Timer t;
    long long k = 0;        // accepted iterations
    long long total_trials = 0;
    long long since_restart = 0;
    double kkt_last_restart = kInf, kkt_prev_cand = kInf;
    bool converged = false;
    bool use_avg_final = false;
    PdhgKkt best{};
    int restarts = 0;
    double last_print = -1;
    while (true) {
        double dx2, dy2, inter;
        be->step(eta / omega, eta * omega, dx2, dy2, inter);
        ++total_trials;
        double num = omega * dx2 + dy2 / omega;
        double eta_max = std::fabs(inter) > 0 ? num / (2.0 * std::fabs(inter)) : kInf;
        double kk = (double)(total_trials + 1);
        double eta_new = std::min((1.0 - std::pow(kk, -0.3)) * eta_max, (1.0 + std::pow(kk, -0.6)) * eta);
        if (eta <= eta_max) {
            be->accept(eta);
            ++k; ++since_restart;
            eta = eta_new;
        } else {
            eta = eta_new;
            if (total_trials > 50 * (k + 10)) break;  // pathological step-size failure
            continue;
        }
        if (k % 64 != 0) continue;
        PdhgKkt kc, ka;
        be->kkt(false, kc);
        be->kkt(true, ka);
        auto metric = [&](const PdhgKkt& q) { return std::sqrt(omega * q.err_p2 + q.err_d2 / omega + q.gap_s * q.gap_s); };
        double mc = metric(kc), ma = metric(ka);
        bool avg_better = ma < mc;
        const PdhgKkt& cand = avg_better ? ka : kc;
        double mcand = avg_better ? ma : mc;
        auto done = [&](const PdhgKkt& q) { return q.rel_p <= opt.tol && q.rel_d <= opt.tol && q.rel_gap <= opt.tol; };
        double el = t.seconds();
        if (g_verbose > 1 || (g_verbose > 0 && (el - last_print >= 1.0 || k == 64))) {
            last_print = el;
            NLOG("%9lld %15.8e %15.8e %9.2e %9.2e %9.2e %7.1fs\n", k, cand.pobj + mdl.obj_offset,
                 cand.dobj + mdl.obj_offset, cand.rel_p, cand.rel_d, cand.rel_gap, el);
        }
        if (done(kc) || done(ka)) {
            converged = true;
            use_avg_final = !done(kc);
            best = use_avg_final ? ka : kc;
            break;
        }
        best = cand;
        use_avg_final = avg_better;
        if (k >= opt.max_iter || el > opt.time_limit) break;
        // adaptive restart
        if (kkt_last_restart == kInf) { kkt_last_restart = mcand; kkt_prev_cand = mcand; continue; }
        bool do_restart = false;
        if (mcand <= 0.2 * kkt_last_restart) do_restart = true;
        else if (mcand <= 0.8 * kkt_last_restart && mcand > kkt_prev_cand) do_restart = true;
        else if (since_restart >= 0.36 * k) do_restart = true;
        kkt_prev_cand = mcand;
        if (do_restart) {
            double dxr, dyr;
            be->restart(avg_better, dxr, dyr);
            if (dxr > 1e-10 && dyr > 1e-10) omega = std::exp(0.5 * std::log(dyr / dxr) + 0.5 * std::log(omega));
            kkt_last_restart = mcand;
            kkt_prev_cand = kInf;
            since_restart = 0;
            ++restarts;
        }
    }
    double el = t.seconds();
    info.restarts = restarts;
    info.ms_per_iter = k > 0 ? 1000.0 * el / k : 0;
    NLOG("PDHG: %lld iterations, %d restarts, %.2fs (%.3f ms/iter), %s\n", k, restarts, el, info.ms_per_iter,
         converged ? "converged" : "not converged");

    std::vector<double> xs, ys;
    be->get(use_avg_final, xs, ys);
    sol.x.resize(n); sol.y.resize(m);
    for (int j = 0; j < n; ++j) sol.x[j] = xs[j] * Dc[j] / bs;
    for (int i = 0; i < m; ++i) sol.y[i] = ys[i] * Dr[i] / cs;
    sol.status = converged ? Status::Optimal : (el > opt.time_limit ? Status::TimeLimit : Status::IterationLimit);
    sol.iterations = (int)k;
    sol.pinf = best.rel_p; sol.dinf = best.rel_d; sol.gap = best.rel_gap;
    sol.obj = best.pobj + mdl.obj_offset;
    sol.dual_obj = best.dobj + mdl.obj_offset;
    return sol;
}

}  // namespace nirnay
