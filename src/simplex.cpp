#include "simplex.hpp"

#include <algorithm>
#include <cmath>

namespace nirnay {

double DualSimplex::rand01() {
    rng_ ^= rng_ << 13; rng_ ^= rng_ >> 7; rng_ ^= rng_ << 17;
    return (rng_ >> 11) * (1.0 / 9007199254740992.0);
}

void DualSimplex::load(const Model& mdl) {
    n_ = mdl.n; m_ = mdl.m; N_ = n_ + m_;
    A_ = mdl.A;
    AT_ = A_.transpose();
    c_.assign(N_, 0.0);
    lo_.assign(N_, 0.0); up_.assign(N_, 0.0);
    for (int j = 0; j < n_; ++j) { c_[j] = mdl.c[j]; lo_[j] = mdl.col_lo[j]; up_[j] = mdl.col_up[j]; }
    for (int i = 0; i < m_; ++i) { lo_[n_ + i] = mdl.row_lo[i]; up_[n_ + i] = mdl.row_up[i]; }
    offset_ = mdl.obj_offset;
    cw_ = c_;
    x_.assign(N_, 0.0); d_.assign(N_, 0.0); y_.assign(m_, 0.0);
    head_.resize(m_); pos_.assign(N_, -1); st_.assign(N_, AT_LO);
    for (int i = 0; i < m_; ++i) { head_[i] = n_ + i; pos_[n_ + i] = i; st_[n_ + i] = BASIC; }
    for (int j = 0; j < n_; ++j) { st_[j] = AT_LO; normalize_status(j); set_nonbasic_value(j); }
    w_.assign(m_, 1.0);
    arow_.assign(N_, 0.0);
    is_touched_.assign(N_, 0);
    factored_ = false;
    iters_ = 0;
}

void DualSimplex::normalize_status(int j) {
    if (st_[j] == BASIC) return;
    const bool fl = is_finite(lo_[j]), fu = is_finite(up_[j]);
    if (fl && fu && lo_[j] == up_[j]) { st_[j] = FIXED; return; }
    switch (st_[j]) {
        case AT_LO: if (!fl) st_[j] = fu ? AT_UP : AT_ZERO; break;
        case AT_UP: if (!fu) st_[j] = fl ? AT_LO : AT_ZERO; break;
        case AT_ZERO: if (fl) st_[j] = AT_LO; else if (fu) st_[j] = AT_UP; break;
        case FIXED: st_[j] = fl ? AT_LO : (fu ? AT_UP : AT_ZERO); break;
        default: break;
    }
}

void DualSimplex::set_nonbasic_value(int j) {
    switch (st_[j]) {
        case AT_LO: case FIXED: x_[j] = lo_[j]; break;
        case AT_UP: x_[j] = up_[j]; break;
        case AT_ZERO: x_[j] = 0.0; break;
        default: break;
    }
}

void DualSimplex::set_col_bounds(int j, double lo, double up) {
    lo_[j] = lo; up_[j] = up;
    if (st_[j] != BASIC) { normalize_status(j); set_nonbasic_value(j); }
}

void DualSimplex::set_basis(const Basis& b) {
    head_ = b.head; st_ = b.status;
    std::fill(pos_.begin(), pos_.end(), -1);
    for (int k = 0; k < m_; ++k) pos_[head_[k]] = k;
    for (int j = 0; j < N_; ++j) if (st_[j] != BASIC) { normalize_status(j); set_nonbasic_value(j); }
    w_.assign(m_, 1.0);
    factored_ = false;
}

void DualSimplex::add_rows(const std::vector<std::vector<std::pair<int, double>>>& rows,
                           const std::vector<double>& rlo, const std::vector<double>& rup) {
    const int k = (int)rows.size();
    if (k == 0) return;
    std::vector<int> ri, ci; std::vector<double> v;
    for (int j = 0; j < n_; ++j)
        for (int p = A_.colptr[j]; p < A_.colptr[j + 1]; ++p) { ri.push_back(A_.rowidx[p]); ci.push_back(j); v.push_back(A_.val[p]); }
    for (int t = 0; t < k; ++t)
        for (auto& e : rows[t]) { ri.push_back(m_ + t); ci.push_back(e.first); v.push_back(e.second); }
    A_ = csc_from_triplets(m_ + k, n_, ri, ci, v);
    AT_ = A_.transpose();
    for (int t = 0; t < k; ++t) {
        c_.push_back(0.0); cw_.push_back(0.0);
        lo_.push_back(rlo[t]); up_.push_back(rup[t]);
        double act = 0;
        for (auto& e : rows[t]) act += e.second * x_[e.first];
        x_.push_back(act); d_.push_back(0.0);
        st_.push_back(BASIC); pos_.push_back(m_ + t);
        head_.push_back(N_ + t);
        y_.push_back(0.0); w_.push_back(1.0);
    }
    m_ += k; N_ += k;
    arow_.assign(N_, 0.0);
    is_touched_.assign(N_, 0);
    factored_ = false;
}

void DualSimplex::remove_rows(const std::vector<int>& rows) {
    if (rows.empty()) return;
    std::vector<char> drop(m_, 0);
    for (int i : rows) if (st_[n_ + i] == BASIC) drop[i] = 1;
    std::vector<int> newrow(m_, -1);
    int m2 = 0;
    for (int i = 0; i < m_; ++i) if (!drop[i]) newrow[i] = m2++;
    if (m2 == m_) return;
    std::vector<int> ri, ci; std::vector<double> v;
    for (int j = 0; j < n_; ++j)
        for (int p = A_.colptr[j]; p < A_.colptr[j + 1]; ++p) {
            int i = newrow[A_.rowidx[p]];
            if (i >= 0) { ri.push_back(i); ci.push_back(j); v.push_back(A_.val[p]); }
        }
    A_ = csc_from_triplets(m2, n_, ri, ci, v);
    AT_ = A_.transpose();
    auto var_map = [&](int j) { return j < n_ ? j : (newrow[j - n_] >= 0 ? n_ + newrow[j - n_] : -1); };
    const int N2 = n_ + m2;
    std::vector<double> c2(N2), cw2(N2), lo2(N2), up2(N2), x2(N2), d2(N2);
    std::vector<signed char> st2(N2);
    for (int j = 0; j < N_; ++j) {
        int k = var_map(j);
        if (k < 0) continue;
        c2[k] = c_[j]; cw2[k] = cw_[j]; lo2[k] = lo_[j]; up2[k] = up_[j]; x2[k] = x_[j]; d2[k] = d_[j]; st2[k] = st_[j];
    }
    std::vector<int> head2; std::vector<double> w2;
    for (int k = 0; k < m_; ++k) {
        int j = var_map(head_[k]);
        if (j < 0) continue;
        head2.push_back(j); w2.push_back(w_[k]);
    }
    std::vector<double> y2(m2);
    for (int i = 0; i < m_; ++i) if (newrow[i] >= 0) y2[newrow[i]] = y_[i];
    c_.swap(c2); cw_.swap(cw2); lo_.swap(lo2); up_.swap(up2); x_.swap(x2); d_.swap(d2); st_.swap(st2);
    head_.swap(head2); w_.swap(w2); y_.swap(y2);
    m_ = m2; N_ = N2;
    pos_.assign(N_, -1);
    for (int k = 0; k < m_; ++k) pos_[head_[k]] = k;
    arow_.assign(N_, 0.0);
    is_touched_.assign(N_, 0);
    touched_.clear();
    factored_ = false;
}

void DualSimplex::col_scatter_add(int j, double s, std::vector<double>& v) const {
    if (j < n_) {
        for (int p = A_.colptr[j]; p < A_.colptr[j + 1]; ++p) v[A_.rowidx[p]] += s * A_.val[p];
    } else {
        v[j - n_] -= s;
    }
}

double DualSimplex::col_dot(int j, const std::vector<double>& v) const {
    if (j < n_) {
        double s = 0;
        for (int p = A_.colptr[j]; p < A_.colptr[j + 1]; ++p) s += A_.val[p] * v[A_.rowidx[p]];
        return s;
    }
    return -v[j - n_];
}

bool DualSimplex::refactor() {
    for (int attempt = 0; attempt < 4; ++attempt) {
        std::vector<int> Bp(m_ + 1, 0), Bi; std::vector<double> Bx;
        for (int k = 0; k < m_; ++k) {
            int j = head_[k];
            if (j < n_) {
                for (int p = A_.colptr[j]; p < A_.colptr[j + 1]; ++p) { Bi.push_back(A_.rowidx[p]); Bx.push_back(A_.val[p]); }
            } else {
                Bi.push_back(j - n_); Bx.push_back(-1.0);
            }
            Bp[k + 1] = (int)Bi.size();
        }
        int ns = lu_.factor(m_, Bp, Bi, Bx);
        if (ns == 0) { factored_ = true; return true; }
        // basis repair: replace singular columns with logicals of unpivoted rows
        const auto& sp = lu_.singular_positions();
        const auto& ur = lu_.unpivoted_rows();
        NLOG2("simplex: singular basis (%d columns), repairing\n", ns);
        for (size_t t = 0; t < sp.size() && t < ur.size(); ++t) {
            int k = sp[t], jout = head_[k], jin = n_ + ur[t];
            st_[jout] = AT_LO; pos_[jout] = -1;
            normalize_status(jout); set_nonbasic_value(jout);
            head_[k] = jin; pos_[jin] = k; st_[jin] = BASIC;
            w_[k] = 1.0;
        }
    }
    factored_ = false;
    return false;
}

void DualSimplex::compute_primal() {
    std::vector<double> rhs(m_, 0.0);
    for (int j = 0; j < N_; ++j)
        if (st_[j] != BASIC && x_[j] != 0.0) col_scatter_add(j, -x_[j], rhs);
    lu_.ftran(rhs);
    for (int k = 0; k < m_; ++k) x_[head_[k]] = rhs[k];
}

void DualSimplex::compute_dual() {
    std::vector<double> cb(m_);
    for (int k = 0; k < m_; ++k) cb[k] = cw_[head_[k]];
    lu_.btran(cb);
    y_ = cb;
    for (int j = 0; j < N_; ++j) d_[j] = (st_[j] == BASIC) ? 0.0 : cw_[j] - col_dot(j, y_);
}

// Make boxed nonbasics dual feasible by flipping; count remaining dual infeasibilities.
int DualSimplex::flip_and_count_dual_infeasible(double tol, bool& flipped) {
    int cnt = 0;
    flipped = false;
    for (int j = 0; j < N_; ++j) {
        const signed char s = st_[j];
        if (s == BASIC || s == FIXED) continue;
        const bool boxed = is_finite(lo_[j]) && is_finite(up_[j]);
        if (s == AT_LO && d_[j] < -tol) {
            if (boxed) { st_[j] = AT_UP; set_nonbasic_value(j); flipped = true; } else ++cnt;
        } else if (s == AT_UP && d_[j] > tol) {
            if (boxed) { st_[j] = AT_LO; set_nonbasic_value(j); flipped = true; } else ++cnt;
        } else if (s == AT_ZERO && std::fabs(d_[j]) > tol) {
            ++cnt;
        }
    }
    return cnt;
}

bool DualSimplex::phase1(const SimplexOptions& opt) {
    // Auxiliary problem: same costs, bounds replaced by boxes in [-1, 1] by type.
    std::vector<double> slo = lo_, sup = up_;
    for (int j = 0; j < N_; ++j) {
        const bool fl = is_finite(slo[j]), fu = is_finite(sup[j]);
        if (fl && fu) { lo_[j] = 0; up_[j] = 0; }
        else if (fl) { lo_[j] = 0; up_[j] = 1; }
        else if (fu) { lo_[j] = -1; up_[j] = 0; }
        else { lo_[j] = -1; up_[j] = 1; }
        if (st_[j] != BASIC) {
            if (lo_[j] == up_[j]) st_[j] = FIXED;
            else st_[j] = d_[j] >= 0 ? AT_LO : AT_UP;
            set_nonbasic_value(j);
        }
    }
    compute_primal();
    SimplexOptions o = opt;
    o.cutoff = kInf;
    SimplexResult r = iterate(o, false);
    lo_ = slo; up_ = sup;
    for (int j = 0; j < N_; ++j) {
        if (st_[j] == BASIC) continue;
        const bool fl = is_finite(lo_[j]), fu = is_finite(up_[j]);
        if (fl && fu) st_[j] = (lo_[j] == up_[j]) ? FIXED : (d_[j] >= 0 ? AT_LO : AT_UP);
        else if (fl) st_[j] = AT_LO;
        else if (fu) st_[j] = AT_UP;
        else st_[j] = AT_ZERO;
        set_nonbasic_value(j);
    }
    compute_primal();
    return r == SimplexResult::Optimal;
}

SimplexResult DualSimplex::iterate(const SimplexOptions& opt, bool allow_cutoff) {
    const double tolP = opt.tol_primal, tolD = opt.tol_dual, pivtol = 1e-9;
    std::vector<double> rho(m_), acol(m_), tau(m_), flipcol(m_);
    std::vector<int> cand;
    std::vector<int> flips;
    int trouble = 0;
    for (;;) {
        if (iters_ >= opt.iter_limit) return SimplexResult::IterLimit;
        if ((iters_ & 63) == 0 && timer_.seconds() > opt.time_limit) return SimplexResult::TimeLimit;
        if (lu_.num_updates() >= opt.refactor_freq) {
            if (!refactor()) return SimplexResult::Numerical;
            compute_primal();
            compute_dual();
        }
        if (allow_cutoff && is_finite(opt.cutoff) && (iters_ % 8) == 0) {
            double obj = offset_;
            for (int j = 0; j < N_; ++j) obj += cw_[j] * x_[j];
            if (obj > opt.cutoff) return SimplexResult::Cutoff;
        }
        // ---- CHUZR: dual steepest edge
        int r = -1;
        double best = 0;
        for (int k = 0; k < m_; ++k) {
            const int j = head_[k];
            const double xv = x_[j];
            double inf;
            if (xv < lo_[j] - tolP) inf = lo_[j] - xv;
            else if (xv > up_[j] + tolP) inf = xv - up_[j];
            else continue;
            double sc = inf * inf / w_[k];
            if (sc > best) { best = sc; r = k; }
        }
        if (r < 0) return SimplexResult::Optimal;
        const int jl = head_[r];
        const bool to_upper = x_[jl] > up_[jl];
        const double delta = to_upper ? x_[jl] - up_[jl] : lo_[jl] - x_[jl];  // > 0
        const double s = to_upper ? 1.0 : -1.0;

        // ---- BTRAN and pivot row
        std::fill(rho.begin(), rho.end(), 0.0);
        rho[r] = 1.0;
        lu_.btran(rho);
        for (int j : touched_) { arow_[j] = 0.0; is_touched_[j] = 0; }
        touched_.clear();
        for (int i = 0; i < m_; ++i) {
            const double ri = rho[i];
            if (ri == 0.0) continue;
            for (int p = AT_.colptr[i]; p < AT_.colptr[i + 1]; ++p) {
                const int j = AT_.rowidx[p];
                if (!is_touched_[j]) { is_touched_[j] = 1; touched_.push_back(j); }
                arow_[j] += ri * AT_.val[p];
            }
            const int jlog = n_ + i;
            if (!is_touched_[jlog]) { is_touched_[jlog] = 1; touched_.push_back(jlog); }
            arow_[jlog] = -ri;
        }
        // ---- candidates
        cand.clear();
        for (int j : touched_) {
            const signed char stj = st_[j];
            if (stj == BASIC || stj == FIXED) continue;
            const double at = s * arow_[j];
            if (std::fabs(at) < pivtol) continue;
            if ((stj == AT_LO && at > 0) || (stj == AT_UP && at < 0) || stj == AT_ZERO) cand.push_back(j);
        }
        // ---- Harris ratio test with bound flipping
        flips.clear();
        double slope = delta;
        int q = -1;
        auto ratio = [&](int j) {
            const double at = s * arow_[j];
            if (st_[j] == AT_ZERO) return 0.0;
            return std::max(d_[j] / at, 0.0);
        };
        auto harris = [&](int j) {
            const double at = s * arow_[j];
            if (st_[j] == AT_LO) return (d_[j] + tolD) / at;
            if (st_[j] == AT_UP) return (d_[j] - tolD) / at;
            return tolD / std::fabs(at);
        };
        while (!cand.empty()) {
            double thmax = kInf;
            for (int j : cand) thmax = std::min(thmax, harris(j));
            int qbest = -1; double amax = -1;
            double group_slope = 0;
            bool all_boxed = true;
            for (int j : cand) {
                if (ratio(j) <= thmax) {
                    double a = std::fabs(arow_[j]);
                    if (a > amax) { amax = a; qbest = j; }
                    if (is_finite(lo_[j]) && is_finite(up_[j])) group_slope += a * (up_[j] - lo_[j]);
                    else all_boxed = false;
                }
            }
            if (qbest < 0) break;
            if (all_boxed && slope - group_slope > 0 && cand.size() > 1) {
                // pass the whole group: those variables flip to their opposite bound
                slope -= group_slope;
                size_t wpos = 0;
                for (size_t t = 0; t < cand.size(); ++t) {
                    int j = cand[t];
                    if (ratio(j) <= thmax) flips.push_back(j);
                    else cand[wpos++] = j;
                }
                cand.resize(wpos);
                if (cand.empty()) {  // passed every breakpoint: take the last group's best as entering
                    q = qbest;
                    flips.erase(std::remove(flips.begin(), flips.end(), q), flips.end());
                    break;
                }
                continue;
            }
            q = qbest;
            break;
        }
        if (q < 0) {
            if (!flips.empty()) {
                // should not happen (slope stayed positive with no candidates left)
                q = flips.back(); flips.pop_back();
            } else {
                return SimplexResult::Infeasible;  // dual unbounded -> primal infeasible
            }
        }
        const double atq = s * arow_[q];
        double t = d_[q] / atq;
        if (st_[q] == AT_ZERO) t = d_[q] / atq;
        if (t < 0) {  // slightly dual infeasible within tolerance: shift cost so d_q = 0
            cw_[q] -= d_[q];
            d_[q] = 0.0;
            t = 0.0;
        }
        const double theta_d = s * t;

        // ---- FTRAN entering column
        std::fill(acol.begin(), acol.end(), 0.0);
        col_scatter_add(q, 1.0, acol);
        lu_.ftran(acol);
        const double arq = acol[r];
        if (std::fabs(arq) < 1e-11 || std::fabs(arq - arow_[q]) > 1e-8 * (1.0 + std::fabs(arq))) {
            if (++trouble > 5) return SimplexResult::Numerical;
            if (!refactor()) return SimplexResult::Numerical;
            compute_primal();
            compute_dual();
            continue;
        }
        trouble = 0;

        // ---- dual update
        for (int j : touched_) {
            if (st_[j] == BASIC) continue;
            d_[j] -= theta_d * arow_[j];
        }
        d_[q] = 0.0;
        d_[jl] = -theta_d;
        for (int i = 0; i < m_; ++i) y_[i] += theta_d * rho[i];

        // ---- bound flips
        if (!flips.empty()) {
            std::fill(flipcol.begin(), flipcol.end(), 0.0);
            for (int j : flips) {
                double old = x_[j];
                st_[j] = (st_[j] == AT_LO) ? AT_UP : AT_LO;
                set_nonbasic_value(j);
                col_scatter_add(j, -(x_[j] - old), flipcol);
            }
            lu_.ftran(flipcol);
            for (int k = 0; k < m_; ++k) x_[head_[k]] += flipcol[k];
        }

        // ---- primal update
        const double target = to_upper ? up_[jl] : lo_[jl];
        const double theta_p = (x_[jl] - target) / arq;
        for (int k = 0; k < m_; ++k) x_[head_[k]] -= theta_p * acol[k];
        x_[q] += theta_p;
        x_[jl] = target;

        // ---- dual steepest-edge weights
        tau = rho;
        double wr = 0;
        for (double v : rho) wr += v * v;
        lu_.ftran(tau);
        for (int k = 0; k < m_; ++k) {
            if (k == r || acol[k] == 0.0) continue;
            const double ratio_k = acol[k] / arq;
            w_[k] = std::max(w_[k] - 2.0 * ratio_k * tau[k] + ratio_k * ratio_k * wr, 1e-4);
        }
        w_[r] = std::max(wr / (arq * arq), 1e-4);

        // ---- basis change
        head_[r] = q; pos_[q] = r; st_[q] = BASIC;
        pos_[jl] = -1;
        st_[jl] = (lo_[jl] == up_[jl]) ? FIXED : (to_upper ? AT_UP : AT_LO);
        lu_.update(r, acol);
        ++iters_;
        if (g_verbose > 1 && (iters_ % 1000) == 0) {
            double obj = offset_;
            for (int j = 0; j < N_; ++j) obj += cw_[j] * x_[j];
            NLOG2("  dual simplex it %lld  obj %.10e\n", iters_, obj);
        }
    }
}

SimplexResult DualSimplex::solve(const SimplexOptions& opt) {
    timer_.reset();
    for (int j = 0; j < N_; ++j) if (st_[j] != BASIC) { normalize_status(j); set_nonbasic_value(j); }
    if (!factored_ && !refactor()) return SimplexResult::Numerical;
    cw_ = c_;
    compute_primal();
    compute_dual();
    SimplexResult res = SimplexResult::Numerical;
    for (int attempt = 0; attempt < 4; ++attempt) {
        bool flipped;
        int ninf = flip_and_count_dual_infeasible(opt.tol_dual, flipped);
        if (flipped) compute_primal();
        if (ninf > 0) {
            if (!phase1(opt)) return SimplexResult::Numerical;
            compute_dual();
            ninf = flip_and_count_dual_infeasible(opt.tol_dual, flipped);
            if (flipped) compute_primal();
            if (ninf > 0) return SimplexResult::DualInfeasible;
        }
        const bool perturb = opt.perturb && attempt == 0 && !is_finite(opt.cutoff);
        if (perturb) {
            double cmax = 0;
            for (int j = 0; j < n_; ++j) cmax = std::max(cmax, std::fabs(c_[j]));
            for (int j = 0; j < n_; ++j) {
                if (st_[j] == BASIC || st_[j] == FIXED) continue;
                double xi = (1e-7 * std::fabs(c_[j]) + 1e-8 * (1.0 + cmax)) * (0.5 + rand01());
                if (st_[j] == AT_LO) { cw_[j] += xi; d_[j] += xi; }
                else if (st_[j] == AT_UP) { cw_[j] -= xi; d_[j] -= xi; }
            }
        }
        res = iterate(opt, is_finite(opt.cutoff));
        if (res != SimplexResult::Optimal) return res;
        // remove perturbation / shifts and re-verify dual feasibility
        bool shifted = false;
        for (int j = 0; j < N_; ++j) if (cw_[j] != c_[j]) { shifted = true; break; }
        if (!shifted) return res;
        cw_ = c_;
        compute_dual();
        ninf = flip_and_count_dual_infeasible(opt.tol_dual, flipped);
        if (flipped) compute_primal();
        if (ninf == 0 && !flipped) return SimplexResult::Optimal;
        bool primal_ok = true;
        for (int k = 0; k < m_; ++k) {
            int j = head_[k];
            if (x_[j] < lo_[j] - opt.tol_primal || x_[j] > up_[j] + opt.tol_primal) { primal_ok = false; break; }
        }
        if (ninf == 0 && primal_ok) return SimplexResult::Optimal;
        NLOG2("simplex: cleanup after perturbation (%d dual infeasibilities)\n", ninf);
    }
    return res;
}

double DualSimplex::objective() const {
    double obj = offset_;
    for (int j = 0; j < n_; ++j) obj += c_[j] * x_[j];
    return obj;
}

void DualSimplex::tableau_row(int r, std::vector<double>& alpha) {
    std::vector<double> rho(m_, 0.0);
    rho[r] = 1.0;
    lu_.btran(rho);
    alpha.assign(N_, 0.0);
    for (int i = 0; i < m_; ++i) {
        const double ri = rho[i];
        if (ri == 0.0) continue;
        for (int p = AT_.colptr[i]; p < AT_.colptr[i + 1]; ++p) alpha[AT_.rowidx[p]] += ri * AT_.val[p];
        alpha[n_ + i] = -ri;
    }
    for (int k = 0; k < m_; ++k) alpha[head_[k]] = 0.0;
}

}  // namespace nirnay
