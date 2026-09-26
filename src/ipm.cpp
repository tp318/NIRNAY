#include "ipm.hpp"

#include <algorithm>
#include <cmath>

#include "ldlt.hpp"

namespace nirnay {

namespace {

using Vec = std::vector<double>;

double inf_norm(const Vec& v) {
    double s = 0;
    for (double a : v) s = std::max(s, std::fabs(a));
    return s;
}

struct Direction {
    Vec dx, dy, dsl, dsu, dzl, dzu;
};

class Ipm {
public:
    Ipm(const Model& mdl, const IpmOptions& opt) : mdl_(mdl), opt_(opt) { setup(); }
    LpSolution run();

private:
    const Model& mdl_;
    IpmOptions opt_;
    int n_ = 0, m_ = 0, N_ = 0;
    std::vector<int> slack_row_;
    CscMatrix Ah_, AhT_;
    Vec b_, c_, l_, u_, qdiag_;
    std::vector<char> hasL_, hasU_;
    int ncomp_ = 0;
    // KKT
    std::vector<int> kcp_, kri_;
    Vec kval_;
    LdlFactor F_;
    Vec D_;
    double rho_ = 1e-10, delta_ = 1e-10;
    // iterate
    Vec x_, y_, sl_, su_, zl_, zu_;
    // residuals
    Vec rp_, rd_, rbl_, rbu_;

    void setup();
    void fill_kkt(bool useQ);
    void kkt_mul(const Vec& v, Vec& out, bool useQ) const;
    void kkt_solve(Vec& rhs, bool useQ);
    void qmul(const Vec& x, Vec& out) const;  // out = Q x (N-sized, zero on slack part)
    void residuals();
    void newton(const Vec& rcl, const Vec& rcu, Direction& d);
    void step_lengths(const Direction& d, double& ap, double& ad) const;
    void starting_point();
};

void Ipm::setup() {
    n_ = mdl_.n;
    m_ = mdl_.m;
    for (int i = 0; i < m_; ++i)
        if (mdl_.row_lo[i] != mdl_.row_up[i]) slack_row_.push_back(i);
    N_ = n_ + (int)slack_row_.size();

    // Â = [A, -S]
    Ah_.m = m_; Ah_.n = N_;
    Ah_.colptr = mdl_.A.colptr;
    Ah_.rowidx = mdl_.A.rowidx;
    Ah_.val = mdl_.A.val;
    for (int r : slack_row_) {
        Ah_.rowidx.push_back(r);
        Ah_.val.push_back(-1.0);
        Ah_.colptr.push_back((int)Ah_.rowidx.size());
    }
    AhT_ = Ah_.transpose();

    b_.assign(m_, 0.0);
    for (int i = 0; i < m_; ++i)
        if (mdl_.row_lo[i] == mdl_.row_up[i]) b_[i] = mdl_.row_lo[i];
    c_.assign(N_, 0.0);
    l_.assign(N_, -kInf);
    u_.assign(N_, kInf);
    for (int j = 0; j < n_; ++j) { c_[j] = mdl_.c[j]; l_[j] = mdl_.col_lo[j]; u_[j] = mdl_.col_up[j]; }
    for (size_t k = 0; k < slack_row_.size(); ++k) {
        l_[n_ + k] = mdl_.row_lo[slack_row_[k]];
        u_[n_ + k] = mdl_.row_up[slack_row_[k]];
    }
    hasL_.resize(N_); hasU_.resize(N_);
    ncomp_ = 0;
    for (int j = 0; j < N_; ++j) {
        hasL_[j] = is_finite(l_[j]);
        hasU_[j] = is_finite(u_[j]);
        ncomp_ += hasL_[j] + hasU_[j];
    }
    qdiag_.assign(N_, 0.0);
    if (mdl_.has_q())
        for (int j = 0; j < n_; ++j)
            for (int p = mdl_.Q.colptr[j]; p < mdl_.Q.colptr[j + 1]; ++p)
                if (mdl_.Q.rowidx[p] == j) qdiag_[j] += mdl_.Q.val[p];

    // KKT pattern (full symmetric CSC)
    const int K = N_ + m_;
    kcp_.assign(K + 1, 0);
    kri_.clear();
    for (int j = 0; j < N_; ++j) {
        if (j < n_ && mdl_.has_q())
            for (int p = mdl_.Q.colptr[j]; p < mdl_.Q.colptr[j + 1]; ++p)
                if (mdl_.Q.rowidx[p] != j) kri_.push_back(mdl_.Q.rowidx[p]);
        kri_.push_back(j);
        for (int p = Ah_.colptr[j]; p < Ah_.colptr[j + 1]; ++p) kri_.push_back(N_ + Ah_.rowidx[p]);
        kcp_[j + 1] = (int)kri_.size();
    }
    for (int i = 0; i < m_; ++i) {
        for (int p = AhT_.colptr[i]; p < AhT_.colptr[i + 1]; ++p) kri_.push_back(AhT_.rowidx[p]);
        kri_.push_back(N_ + i);
        kcp_[N_ + i + 1] = (int)kri_.size();
    }
    kval_.assign(kri_.size(), 0.0);
    std::vector<signed char> sign(K);
    for (int j = 0; j < K; ++j) sign[j] = (j < N_) ? -1 : 1;
    F_.analyse(K, kcp_, kri_, sign);
    D_.assign(N_, 0.0);
}

void Ipm::fill_kkt(bool useQ) {
    int p = 0;
    for (int j = 0; j < N_; ++j) {
        if (j < n_ && mdl_.has_q())
            for (int q = mdl_.Q.colptr[j]; q < mdl_.Q.colptr[j + 1]; ++q)
                if (mdl_.Q.rowidx[q] != j) kval_[p++] = useQ ? -mdl_.Q.val[q] : 0.0;
        kval_[p++] = -((useQ ? qdiag_[j] : 0.0) + D_[j] + rho_);
        for (int q = Ah_.colptr[j]; q < Ah_.colptr[j + 1]; ++q) kval_[p++] = Ah_.val[q];
    }
    for (int i = 0; i < m_; ++i) {
        for (int q = AhT_.colptr[i]; q < AhT_.colptr[i + 1]; ++q) kval_[p++] = AhT_.val[q];
        kval_[p++] = delta_;
    }
}

void Ipm::qmul(const Vec& x, Vec& out) const {
    std::fill(out.begin(), out.end(), 0.0);
    if (mdl_.has_q()) mdl_.Q.mul_add(x.data(), out.data());
}

// Unregularised KKT product: [-(Q+D) Â'; Â 0] v
void Ipm::kkt_mul(const Vec& v, Vec& out, bool useQ) const {
    out.assign(N_ + m_, 0.0);
    if (useQ && mdl_.has_q()) {
        for (int j = 0; j < n_; ++j) {
            double s = 0;
            for (int p = mdl_.Q.colptr[j]; p < mdl_.Q.colptr[j + 1]; ++p) s += mdl_.Q.val[p] * v[mdl_.Q.rowidx[p]];
            out[j] -= s;
        }
    }
    for (int j = 0; j < N_; ++j) {
        out[j] -= D_[j] * v[j];
        double s = 0;
        for (int p = Ah_.colptr[j]; p < Ah_.colptr[j + 1]; ++p) {
            s += Ah_.val[p] * v[N_ + Ah_.rowidx[p]];
            out[N_ + Ah_.rowidx[p]] += Ah_.val[p] * v[j];
        }
        out[j] += s;
    }
}

void Ipm::kkt_solve(Vec& rhs, bool useQ) {
    Vec sol = rhs;
    F_.solve(sol);
    const double rn = inf_norm(rhs);
    Vec r, corr, best = sol;
    double best_res = kInf;
    for (int it = 0; it < 30; ++it) {
        kkt_mul(sol, r, useQ);
        for (size_t k = 0; k < r.size(); ++k) r[k] = rhs[k] - r[k];
        double res = inf_norm(r);
        if (res < best_res) { best_res = res; best = sol; }
        else if (res > 2.0 * best_res) break;  // refinement diverging: keep best iterate
        if (res <= 1e-15 * (1.0 + rn)) break;
        corr = r;
        F_.solve(corr);
        for (size_t k = 0; k < sol.size(); ++k) sol[k] += corr[k];
    }
    rhs.swap(best);
}

void Ipm::residuals() {
    // rp = b - Â x
    rp_ = b_;
    Ah_.mul_add(x_.data(), rp_.data(), -1.0);
    // rd = c + Qx - Â'y - zl + zu
    rd_.assign(N_, 0.0);
    qmul(x_, rd_);
    for (int j = 0; j < N_; ++j) rd_[j] += c_[j] - zl_[j] + zu_[j];
    Ah_.mul_t_add(y_.data(), rd_.data(), -1.0);
    rbl_.assign(N_, 0.0);
    rbu_.assign(N_, 0.0);
    for (int j = 0; j < N_; ++j) {
        if (hasL_[j]) rbl_[j] = l_[j] - x_[j] + sl_[j];
        if (hasU_[j]) rbu_[j] = u_[j] - x_[j] - su_[j];
    }
}

void Ipm::newton(const Vec& rcl, const Vec& rcu, Direction& d) {
    Vec rhs(N_ + m_, 0.0);
    for (int j = 0; j < N_; ++j) {
        double rx = -rd_[j];
        if (hasL_[j]) rx += (rcl[j] + zl_[j] * rbl_[j]) / sl_[j];
        if (hasU_[j]) rx -= (rcu[j] - zu_[j] * rbu_[j]) / su_[j];
        rhs[j] = -rx;
    }
    for (int i = 0; i < m_; ++i) rhs[N_ + i] = rp_[i];
    kkt_solve(rhs, true);
    d.dx.assign(rhs.begin(), rhs.begin() + N_);
    d.dy.assign(rhs.begin() + N_, rhs.end());
    d.dsl.assign(N_, 0.0); d.dsu.assign(N_, 0.0);
    d.dzl.assign(N_, 0.0); d.dzu.assign(N_, 0.0);
    for (int j = 0; j < N_; ++j) {
        if (hasL_[j]) {
            d.dsl[j] = d.dx[j] - rbl_[j];
            d.dzl[j] = (rcl[j] - zl_[j] * d.dsl[j]) / sl_[j];
        }
        if (hasU_[j]) {
            d.dsu[j] = rbu_[j] - d.dx[j];
            d.dzu[j] = (rcu[j] - zu_[j] * d.dsu[j]) / su_[j];
        }
    }
}

void Ipm::step_lengths(const Direction& d, double& ap, double& ad) const {
    ap = 1.0; ad = 1.0;
    for (int j = 0; j < N_; ++j) {
        if (hasL_[j]) {
            if (d.dsl[j] < 0) ap = std::min(ap, -sl_[j] / d.dsl[j]);
            if (d.dzl[j] < 0) ad = std::min(ad, -zl_[j] / d.dzl[j]);
        }
        if (hasU_[j]) {
            if (d.dsu[j] < 0) ap = std::min(ap, -su_[j] / d.dsu[j]);
            if (d.dzu[j] < 0) ad = std::min(ad, -zu_[j] / d.dzu[j]);
        }
    }
}

void Ipm::starting_point() {
    // Factor [-(I+ρ) Â'; Â δ] (Q omitted) and compute least-norm primal / least-squares dual.
    std::fill(D_.begin(), D_.end(), 1.0);
    fill_kkt(false);
    F_.factor(kval_, 1e-13, 2e-7);
    Vec rhs(N_ + m_, 0.0);
    for (int i = 0; i < m_; ++i) rhs[N_ + i] = b_[i];
    // bias the primal toward the box centre for boxed variables
    for (int j = 0; j < N_; ++j) {
        double t = 0;
        if (hasL_[j] && hasU_[j]) t = 0.5 * (l_[j] + u_[j]);
        else if (hasL_[j] && l_[j] > 0) t = l_[j];
        else if (hasU_[j] && u_[j] < 0) t = u_[j];
        rhs[j] = -t;  // min 0.5||x - t||^2  s.t. Âx = b
    }
    kkt_solve(rhs, false);
    x_.assign(rhs.begin(), rhs.begin() + N_);
    Vec qx(N_, 0.0);
    qmul(x_, qx);
    Vec rhs2(N_ + m_, 0.0);
    for (int j = 0; j < N_; ++j) rhs2[j] = c_[j] + qx[j];
    kkt_solve(rhs2, false);
    y_.assign(rhs2.begin() + N_, rhs2.end());
    Vec zhat(N_);
    for (int j = 0; j < N_; ++j) zhat[j] = c_[j] + qx[j];
    Ah_.mul_t_add(y_.data(), zhat.data(), -1.0);

    sl_.assign(N_, 1.0); su_.assign(N_, 1.0);
    zl_.assign(N_, 0.0); zu_.assign(N_, 0.0);
    double smin = kInf, zmin = kInf;
    for (int j = 0; j < N_; ++j) {
        if (hasL_[j]) { sl_[j] = x_[j] - l_[j]; smin = std::min(smin, sl_[j]); }
        if (hasU_[j]) { su_[j] = u_[j] - x_[j]; smin = std::min(smin, su_[j]); }
        if (hasL_[j] && hasU_[j]) {
            zl_[j] = std::max(zhat[j], 0.0); zu_[j] = std::max(-zhat[j], 0.0);
        } else if (hasL_[j]) {
            zl_[j] = zhat[j];
        } else if (hasU_[j]) {
            zu_[j] = -zhat[j];
        }
        if (hasL_[j]) zmin = std::min(zmin, zl_[j]);
        if (hasU_[j]) zmin = std::min(zmin, zu_[j]);
    }
    if (ncomp_ == 0) return;
    double dp = std::max(-1.5 * smin, 0.0), dd = std::max(-1.5 * zmin, 0.0);
    double sz = 0, ssum = 0, zsum = 0;
    for (int j = 0; j < N_; ++j) {
        if (hasL_[j]) { sl_[j] += dp; zl_[j] += dd; sz += sl_[j] * zl_[j]; ssum += sl_[j]; zsum += zl_[j]; }
        if (hasU_[j]) { su_[j] += dp; zu_[j] += dd; sz += su_[j] * zu_[j]; ssum += su_[j]; zsum += zu_[j]; }
    }
    double dp2 = zsum > 0 ? 0.5 * sz / zsum : 1.0;
    double dd2 = ssum > 0 ? 0.5 * sz / ssum : 1.0;
    if (!(dp2 > 1e-8)) dp2 = 1.0;
    if (!(dd2 > 1e-8)) dd2 = 1.0;
    for (int j = 0; j < N_; ++j) {
        if (hasL_[j]) { sl_[j] += dp2; zl_[j] += dd2; }
        if (hasU_[j]) { su_[j] += dp2; zu_[j] += dd2; }
    }
}

LpSolution Ipm::run() {
    Timer timer;
    LpSolution sol;
    starting_point();

    double bnorm = inf_norm(b_);
    for (int j = 0; j < N_; ++j) {
        if (hasL_[j]) bnorm = std::max(bnorm, std::fabs(l_[j]));
        if (hasU_[j]) bnorm = std::max(bnorm, std::fabs(u_[j]));
    }
    const double cnorm = inf_norm(c_);
    const bool isqp = mdl_.has_q();

    NLOG("IPM: rows %d  cols %d  (augmented %d)  nnz(L) %lld\n", m_, n_, N_ + m_, F_.nnz_L());
    NLOG("%5s %15s %15s %9s %9s %9s %6s\n", "iter", "primal obj", "dual obj", "p.inf", "d.inf", "mu", "step");

    Vec rcl(N_), rcu(N_), qx(N_);
    Direction d, daff;
    double last_step = 0;
    int it = 0;
    for (;; ++it) {
        residuals();
        double comp = 0;
        for (int j = 0; j < N_; ++j) {
            if (hasL_[j]) comp += sl_[j] * zl_[j];
            if (hasU_[j]) comp += su_[j] * zu_[j];
        }
        const double mu = ncomp_ > 0 ? comp / ncomp_ : 0.0;
        qmul(x_, qx);
        double xqx = 0, pobj = 0, dobj = 0;
        for (int j = 0; j < N_; ++j) { xqx += x_[j] * qx[j]; pobj += c_[j] * x_[j]; }
        pobj += 0.5 * xqx;
        for (int i = 0; i < m_; ++i) dobj += b_[i] * y_[i];
        for (int j = 0; j < N_; ++j) {
            if (hasL_[j]) dobj += l_[j] * zl_[j];
            if (hasU_[j]) dobj -= u_[j] * zu_[j];
        }
        dobj -= 0.5 * xqx;
        const double pinf = std::max(inf_norm(rp_), std::max(inf_norm(rbl_), inf_norm(rbu_))) / (1.0 + bnorm);
        const double dinf = inf_norm(rd_) / (1.0 + cnorm);
        const double gap = std::fabs(pobj - dobj) / (1.0 + std::fabs(pobj));
        NLOG("%5d %15.8e %15.8e %9.2e %9.2e %9.2e %6.3f\n", it, pobj, dobj, pinf, dinf, mu, last_step);

        sol.pinf = pinf; sol.dinf = dinf; sol.gap = gap;
        sol.obj = pobj + mdl_.obj_offset; sol.dual_obj = dobj + mdl_.obj_offset;
        sol.iterations = it;
        if (!std::isfinite(pobj) || !std::isfinite(dobj) || !std::isfinite(mu)) {
            sol.status = Status::NumericalError;
            break;
        }
        if (pinf <= opt_.tol && dinf <= opt_.tol && gap <= opt_.tol) { sol.status = Status::Optimal; break; }
        const double xn = inf_norm(x_), yn = std::max(inf_norm(y_), std::max(inf_norm(zl_), inf_norm(zu_)));
        if (it > 5 && (xn > 1e12 * (1 + bnorm) || yn > 1e12 * (1 + cnorm))) {
            sol.status = Status::InfeasibleOrUnbounded;
            NLOG("IPM: iterates diverging (|x|=%.1e, |y,z|=%.1e) -> infeasible or unbounded suspected\n", xn, yn);
            break;
        }
        if (it >= opt_.max_iter) { sol.status = Status::IterationLimit; break; }
        if (timer.seconds() > opt_.time_limit) { sol.status = Status::TimeLimit; break; }

        // factorise
        for (int j = 0; j < N_; ++j) {
            double dj = 0;
            if (hasL_[j]) dj += zl_[j] / sl_[j];
            if (hasU_[j]) dj += zu_[j] / su_[j];
            D_[j] = dj;
        }
        fill_kkt(true);
        F_.factor(kval_, 1e-13, 2e-7);

        // predictor
        for (int j = 0; j < N_; ++j) {
            rcl[j] = hasL_[j] ? -sl_[j] * zl_[j] : 0.0;
            rcu[j] = hasU_[j] ? -su_[j] * zu_[j] : 0.0;
        }
        newton(rcl, rcu, daff);
        double ap, ad;
        step_lengths(daff, ap, ad);
        if (isqp) ap = ad = std::min(ap, ad);
        double sigma = 0;
        if (ncomp_ > 0) {
            double caff = 0;
            for (int j = 0; j < N_; ++j) {
                if (hasL_[j]) caff += (sl_[j] + ap * daff.dsl[j]) * (zl_[j] + ad * daff.dzl[j]);
                if (hasU_[j]) caff += (su_[j] + ap * daff.dsu[j]) * (zu_[j] + ad * daff.dzu[j]);
            }
            double muaff = caff / ncomp_;
            sigma = std::pow(std::max(0.0, muaff / mu), 3.0);
            sigma = std::min(sigma, 1.0);
        }
        // Mehrotra corrector
        const double target = sigma * mu;
        for (int j = 0; j < N_; ++j) {
            rcl[j] = hasL_[j] ? target - sl_[j] * zl_[j] - daff.dsl[j] * daff.dzl[j] : 0.0;
            rcu[j] = hasU_[j] ? target - su_[j] * zu_[j] - daff.dsu[j] * daff.dzu[j] : 0.0;
        }
        newton(rcl, rcu, d);
        step_lengths(d, ap, ad);
        if (isqp) ap = ad = std::min(ap, ad);

        // Gondzio multiple centrality correctors
        for (int k = 0; k < opt_.max_correctors && ncomp_ > 0; ++k) {
            double a_cur = std::min(ap, ad);
            if (a_cur >= 0.999) break;
            double a_try = std::min(1.0, 1.5 * a_cur + 0.1);
            const double bmin = 0.1, bmax = 10.0;
            Vec rcl2 = rcl, rcu2 = rcu;
            auto adj = [&](double v) {
                if (v < bmin * target) return bmin * target - v;
                if (v > bmax * target) return std::max(bmax * target - v, -bmax * target);
                return 0.0;
            };
            for (int j = 0; j < N_; ++j) {
                if (hasL_[j]) rcl2[j] += adj((sl_[j] + a_try * d.dsl[j]) * (zl_[j] + a_try * d.dzl[j]));
                if (hasU_[j]) rcu2[j] += adj((su_[j] + a_try * d.dsu[j]) * (zu_[j] + a_try * d.dzu[j]));
            }
            Direction d2;
            newton(rcl2, rcu2, d2);
            double ap2, ad2;
            step_lengths(d2, ap2, ad2);
            if (isqp) ap2 = ad2 = std::min(ap2, ad2);
            if (std::min(ap2, ad2) >= 1.01 * a_cur) {
                d = std::move(d2); ap = ap2; ad = ad2; rcl = rcl2; rcu = rcu2;
            } else {
                break;
            }
        }

        const double eta = std::max(0.9, std::min(0.9999, 1.0 - 10.0 * mu / (1.0 + std::fabs(pobj))));
        ap = std::min(1.0, eta * ap);
        ad = std::min(1.0, eta * ad);
        last_step = std::min(ap, ad);
        for (int j = 0; j < N_; ++j) {
            x_[j] += ap * d.dx[j];
            if (hasL_[j]) { sl_[j] += ap * d.dsl[j]; zl_[j] += ad * d.dzl[j]; }
            if (hasU_[j]) { su_[j] += ap * d.dsu[j]; zu_[j] += ad * d.dzu[j]; }
        }
        for (int i = 0; i < m_; ++i) y_[i] += ad * d.dy[i];
    }

    sol.x.assign(x_.begin(), x_.begin() + n_);
    sol.y = y_;
    sol.z.assign(n_, 0.0);
    for (int j = 0; j < n_; ++j) sol.z[j] = zl_[j] - zu_[j];
    return sol;
}

}  // namespace

LpSolution solve_ipm(const Model& mdl, const IpmOptions& opt) {
    Ipm ipm(mdl, opt);
    return ipm.run();
}

}  // namespace nirnay
