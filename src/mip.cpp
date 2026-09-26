#include "mip.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <queue>

#include "scaling.hpp"
#include "simplex.hpp"

namespace nirnay {

namespace {

constexpr double kIntTol = 1e-6;

struct BoundChange { int j; double lo, up; };

struct Node {
    std::vector<BoundChange> ch;
    double bound = -kInf;
    double estimate = 0;
    int depth = 0;
    std::shared_ptr<DualSimplex::Basis> basis;
    int br_var = -1;
    int br_dir = 0;          // -1 down, +1 up
    double br_frac = 0;      // distance moved by the branching
    double parent_obj = 0;
};

struct NodeCmp {
    bool operator()(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) const {
        return a->bound > b->bound;
    }
};

class BranchAndCut {
public:
    BranchAndCut(const Model& mdl, const MipOptions& opt) : red_(mdl), opt_(opt) {}
    MipResult run();

private:
    const Model& red_;
    MipOptions opt_;
    Scaling sc_;
    Model sm_;
    DualSimplex lp_;
    int n_ = 0;
    std::vector<char> isint_;
    std::vector<int> ints_;
    std::vector<double> root_lo_, root_up_;
    std::vector<int> modified_;
    std::vector<char> is_modified_;
    // incumbent (scaled space, structural only)
    bool has_inc_ = false;
    double inc_obj_ = kInf;
    std::vector<double> inc_x_;
    // pseudocosts
    std::vector<double> ps_dn_, ps_up_;
    std::vector<int> pn_dn_, pn_up_;
    std::vector<int> lock_dn_, lock_up_;
    bool int_obj_ = false;
    double obj_step_ = 0;
    Timer timer_;
    long long nodes_ = 0;
    int ncuts_ = 0;
    bool trouble_ = false;
    MipResult res_;

    double cutoff() const {
        if (!has_inc_) return kInf;
        double c = inc_obj_ - std::max(opt_.abs_gap * sc_.obj, 1e-9 * std::fabs(inc_obj_));
        if (int_obj_) c = std::min(c, inc_obj_ - obj_step_ + 1e-6 * obj_step_);
        return c;
    }
    bool prunable(double bound) const { return bound >= cutoff(); }
    double frac(double v) const { return v - std::floor(v); }
    bool is_frac(double v) const { double f = frac(v); return f > kIntTol && f < 1 - kIntTol; }

    void set_bound(int j, double lo, double up) {
        if (!is_modified_[j]) { is_modified_[j] = 1; modified_.push_back(j); }
        lp_.set_col_bounds(j, lo, up);
    }
    void reset_bounds() {
        for (int j : modified_) { lp_.set_col_bounds(j, root_lo_[j], root_up_[j]); is_modified_[j] = 0; }
        modified_.clear();
    }
    void apply(const Node& nd) {
        reset_bounds();
        for (const auto& c : nd.ch) set_bound(c.j, c.lo, c.up);
    }
    SimplexResult solve_lp(double cut, long long iter_cap = -1) {
        SimplexOptions so;
        so.perturb = false;
        so.cutoff = cut;
        so.time_limit = std::max(0.0, opt_.time_limit - timer_.seconds());
        if (iter_cap > 0) so.iter_limit = lp_.iterations() + iter_cap;
        return lp_.solve(so);
    }
    bool try_incumbent(std::vector<double> xs, const char* who);
    void simple_rounding(const std::vector<double>& x);
    void diving(int max_depth);
    void fix_and_lp(const std::vector<double>& x);
    int root_cuts();
    int select_branch(const std::vector<double>& x, double node_obj, int depth, Node& nd, bool& resolve, bool& infeasible);
    void update_pseudocost(int j, int dir, double dist, double delta);
    double pseudo(int j, int dir) const;
    void log_progress(size_t open, double bound, bool force);
    double last_log_ = -1;
};

double BranchAndCut::pseudo(int j, int dir) const {
    if (dir < 0) {
        if (pn_dn_[j] > 0) return ps_dn_[j] / pn_dn_[j];
    } else {
        if (pn_up_[j] > 0) return ps_up_[j] / pn_up_[j];
    }
    // average over initialised variables
    double s = 0; int c = 0;
    for (int k : ints_) {
        if (dir < 0 && pn_dn_[k] > 0) { s += ps_dn_[k] / pn_dn_[k]; ++c; }
        if (dir > 0 && pn_up_[k] > 0) { s += ps_up_[k] / pn_up_[k]; ++c; }
    }
    return c ? s / c : 1.0;
}

void BranchAndCut::update_pseudocost(int j, int dir, double dist, double delta) {
    if (dist < 1e-9 || !std::isfinite(delta)) return;
    delta = std::max(delta, 0.0);
    if (dir < 0) { ps_dn_[j] += delta / dist; pn_dn_[j]++; }
    else { ps_up_[j] += delta / dist; pn_up_[j]++; }
}

bool BranchAndCut::try_incumbent(std::vector<double> xs, const char* who) {
    // round integers, check feasibility in the unscaled space
    for (int j : ints_) xs[j] = std::round(xs[j]);
    std::vector<double> xu = xs;
    xs.resize(n_);
    xu.resize(n_);
    unscale_primal(sc_, xu);
    for (int j = 0; j < n_; ++j) {
        if (xu[j] < red_.col_lo[j] - 1e-6 * (1 + std::fabs(red_.col_lo[j]))) return false;
        if (xu[j] > red_.col_up[j] + 1e-6 * (1 + std::fabs(red_.col_up[j]))) return false;
    }
    auto ax = red_.activities(xu);
    for (int i = 0; i < red_.m; ++i) {
        if (ax[i] < red_.row_lo[i] - 1e-6 * (1 + std::fabs(red_.row_lo[i]))) return false;
        if (ax[i] > red_.row_up[i] + 1e-6 * (1 + std::fabs(red_.row_up[i]))) return false;
    }
    double obj = red_.objective(xu) * sc_.obj;
    if (obj < inc_obj_ - 1e-12 * (1 + std::fabs(obj))) {
        inc_obj_ = obj;
        inc_x_ = xs;
        has_inc_ = true;
        NLOG("  * new incumbent %.10e  (%s, node %lld, %.2fs)\n", obj / sc_.obj, who, nodes_, timer_.seconds());
        return true;
    }
    return false;
}

void BranchAndCut::simple_rounding(const std::vector<double>& x) {
    std::vector<double> xr(x.begin(), x.begin() + n_);
    for (int j : ints_) {
        if (!is_frac(xr[j])) continue;
        if (lock_dn_[j] == 0) xr[j] = std::floor(xr[j]);
        else if (lock_up_[j] == 0) xr[j] = std::ceil(xr[j]);
        else return;
    }
    try_incumbent(xr, "rounding");
}

void BranchAndCut::fix_and_lp(const std::vector<double>& x) {
    // Fix every integer at its rounded value and optimise the continuous part.
    auto saved = lp_.basis();
    std::vector<std::pair<int, std::pair<double, double>>> old;
    for (int j : ints_) {
        double v = std::round(x[j]);
        v = std::max(lp_.lo(j), std::min(lp_.up(j), v));
        old.push_back({j, {lp_.lo(j), lp_.up(j)}});
        set_bound(j, v, v);
    }
    SimplexResult r = solve_lp(cutoff(), 5000);
    if (r == SimplexResult::Optimal) try_incumbent(lp_.primal(), "fix-and-LP");
    for (auto& o : old) set_bound(o.first, o.second.first, o.second.second);
    lp_.set_basis(saved);
}

void BranchAndCut::diving(int max_depth) {
    auto saved = lp_.basis();
    std::vector<std::pair<int, std::pair<double, double>>> old;
    auto fix = [&](int j, double lo, double up) {
        old.push_back({j, {lp_.lo(j), lp_.up(j)}});
        set_bound(j, lo, up);
    };
    for (int depth = 0; depth < max_depth; ++depth) {
        auto x = lp_.primal();
        int best = -1; double bf = 2;
        for (int j : ints_) {
            if (!is_frac(x[j])) continue;
            double f = frac(x[j]);
            double d = std::min(f, 1 - f);
            if (d < bf) { bf = d; best = j; }
        }
        if (best < 0) { try_incumbent(x, "diving"); break; }
        simple_rounding(x);
        const double v = x[best];
        const bool up = frac(v) >= 0.5;
        const double lo = lp_.lo(best), upb = lp_.up(best);
        if (up) fix(best, std::ceil(v), upb); else fix(best, lo, std::floor(v));
        SimplexResult r = solve_lp(cutoff(), 2000);
        if (r != SimplexResult::Optimal) {
            // backtrack once: other direction
            set_bound(best, lo, upb);
            if (up) fix(best, lo, std::floor(v)); else fix(best, std::ceil(v), upb);
            r = solve_lp(cutoff(), 2000);
            if (r != SimplexResult::Optimal) break;
        }
    }
    for (auto it = old.rbegin(); it != old.rend(); ++it) set_bound(it->first, it->second.first, it->second.second);
    lp_.set_basis(saved);
}

int BranchAndCut::root_cuts() {
    const int max_rounds = opt_.cut_rounds;
    const size_t max_per_round = 60;
    int total = 0;
    double prev_obj = lp_.objective();
    int stall = 0;
    const int m0 = lp_.rows();
    std::vector<double> alpha;
    for (int round = 0; round < max_rounds; ++round) {
        if (timer_.seconds() > 0.2 * opt_.time_limit) break;
        const auto& xv = lp_.values();
        const auto& head = lp_.head();
        const int m = lp_.rows();
        const int N = lp_.vars();
        std::vector<std::pair<double, int>> cand;
        for (int k = 0; k < m; ++k) {
            int j = head[k];
            if (j >= n_ || !isint_[j]) continue;
            double f = frac(xv[j]);
            if (f < 0.01 || f > 0.99) continue;
            cand.push_back({std::fabs(f - 0.5), k});
        }
        std::sort(cand.begin(), cand.end());
        if (cand.size() > max_per_round) cand.resize(max_per_round);
        const CscMatrix& A = lp_.matrix();
        CscMatrix AT = A.transpose();
        std::vector<std::vector<std::pair<int, double>>> rows;
        std::vector<double> rlo, rup;
        for (auto& cd : cand) {
            const int k = cd.second;
            const double beta = xv[head[k]];
            const double f0 = frac(beta);
            lp_.tableau_row(k, alpha);
            std::vector<double> pi(n_, 0.0);
            double rhs = 1.0;
            bool ok = true;
            for (int j = 0; j < N && ok; ++j) {
                double a = alpha[j];
                if (std::fabs(a) < 1e-11) continue;
                int st = lp_.status_of(j);
                if (st == DualSimplex::BASIC || st == DualSimplex::FIXED) continue;
                if (st == DualSimplex::AT_ZERO) { ok = false; break; }
                const bool at_up = (st == DualSimplex::AT_UP);
                const double ap = at_up ? -a : a;
                const bool jint = (j < n_) && isint_[j];
                double g;
                if (jint) {
                    double fj = frac(ap);
                    g = (fj <= f0) ? fj / f0 : (1 - fj) / (1 - f0);
                } else {
                    g = (ap > 0) ? ap / f0 : -ap / (1 - f0);
                }
                if (g == 0.0) continue;
                if (g > 1e7) { ok = false; break; }
                const double bnd = at_up ? lp_.up(j) : lp_.lo(j);
                const double sgn = at_up ? -1.0 : 1.0;  // t = sgn * (x_j - bnd)
                if (j < n_) {
                    pi[j] += sgn * g;
                } else {
                    const int i = j - n_;
                    for (int p = AT.colptr[i]; p < AT.colptr[i + 1]; ++p) pi[AT.rowidx[p]] += sgn * g * AT.val[p];
                }
                rhs += sgn * g * bnd;
            }
            if (!ok) continue;
            // clean tiny coefficients conservatively, check dynamism and efficacy
            double pmax = 0;
            for (double v : pi) pmax = std::max(pmax, std::fabs(v));
            if (pmax < 1e-9) continue;
            std::vector<std::pair<int, double>> row;
            double pmin = kInf, act = 0, nrm = 0;
            for (int j = 0; j < n_ && ok; ++j) {
                double v = pi[j];
                if (v == 0.0) continue;
                if (std::fabs(v) < 1e-9 * pmax) {
                    double b = (v > 0) ? lp_.up(j) : lp_.lo(j);
                    if (!is_finite(b)) { ok = false; break; }
                    rhs -= v * b;
                    continue;
                }
                row.push_back({j, v});
                pmin = std::min(pmin, std::fabs(v));
                act += v * xv[j];
                nrm += v * v;
            }
            if (!ok || row.empty() || pmax / pmin > 1e8 || !std::isfinite(rhs)) continue;
            nrm = std::sqrt(nrm);
            const double viol = rhs - act;
            if (viol / nrm < 1e-5 || viol < 1e-6 * (1 + std::fabs(rhs))) continue;
            for (auto& e : row) e.second /= pmax;
            rows.push_back(std::move(row));
            rlo.push_back(rhs / pmax);
            rup.push_back(kInf);
        }
        if (rows.empty()) break;
        lp_.add_rows(rows, rlo, rup);
        total += (int)rows.size();
        SimplexOptions so;
        so.perturb = false;
        so.time_limit = std::max(0.0, opt_.time_limit - timer_.seconds());
        SimplexResult r = lp_.solve(so);
        if (r != SimplexResult::Optimal) {
            if (r == SimplexResult::Infeasible) return -1;
            trouble_ = true;
            break;
        }
        double obj = lp_.objective();
        // cut management: drop cuts that are no longer binding (basic logical with slack)
        std::vector<int> purge;
        {
            const auto& xv2 = lp_.values();
            for (int i = m0; i < lp_.rows(); ++i) {
                const int j = n_ + i;
                if (lp_.status_of(j) != DualSimplex::BASIC) continue;
                if (xv2[j] > lp_.lo(j) + 1e-6 * (1 + std::fabs(lp_.lo(j)))) purge.push_back(i);
            }
        }
        if (!purge.empty()) {
            lp_.remove_rows(purge);
            lp_.solve(so);
        }
        NLOG("  cut round %d: +%zu GMI cuts (%zu slack purged, %d active), LP bound %.10e\n", round + 1,
             rows.size(), purge.size(), lp_.rows() - m0, obj / sc_.obj);
        if (obj - prev_obj < 1e-4 * (1 + std::fabs(obj))) { if (++stall >= 2) break; } else stall = 0;
        prev_obj = obj;
        if (lp_.rows() - m0 > 2 * std::max(50, red_.m)) break;
    }
    return total;
}

int BranchAndCut::select_branch(const std::vector<double>& x, double node_obj, int depth, Node& nd,
                                bool& resolve, bool& infeasible) {
    resolve = false; infeasible = false;
    std::vector<int> fr;
    for (int j : ints_) if (is_frac(x[j])) fr.push_back(j);
    if (fr.empty()) return -1;
    const double eps = 1e-6;
    auto score_of = [&](double dn, double up) { return std::max(dn, eps) * std::max(up, eps); };
    std::vector<std::pair<double, int>> ranked;
    for (int j : fr) {
        double f = frac(x[j]);
        ranked.push_back({-score_of(f * pseudo(j, -1), (1 - f) * pseudo(j, +1)), j});
    }
    std::sort(ranked.begin(), ranked.end());
    const int rel = opt_.sb_reliability;
    int sb_budget = depth < 25 ? opt_.sb_candidates : 0;
    int best = ranked[0].second;
    double best_score = -ranked[0].first;
    int sb_done = 0;
    for (auto& rk : ranked) {
        if (sb_done >= sb_budget) break;
        const int j = rk.second;
        if (std::min(pn_dn_[j], pn_up_[j]) >= rel) continue;
        ++sb_done;
        const double v = x[j], lo = lp_.lo(j), up = lp_.up(j), f = frac(v);
        double obj_dn, obj_up;
        set_bound(j, lo, std::floor(v));
        SimplexResult r = solve_lp(cutoff(), opt_.sb_iters);
        bool inf_dn = (r == SimplexResult::Infeasible || r == SimplexResult::Cutoff);
        obj_dn = inf_dn ? kInf : lp_.objective();
        set_bound(j, std::ceil(v), up);
        r = solve_lp(cutoff(), opt_.sb_iters);
        bool inf_up = (r == SimplexResult::Infeasible || r == SimplexResult::Cutoff);
        obj_up = inf_up ? kInf : lp_.objective();
        set_bound(j, lo, up);
        if (!inf_dn) update_pseudocost(j, -1, f, obj_dn - node_obj);
        if (!inf_up) update_pseudocost(j, +1, 1 - f, obj_up - node_obj);
        if (inf_dn && inf_up) { infeasible = true; return -1; }
        if (inf_dn || inf_up) {
            // one side is infeasible/pruned: tighten the node and re-solve
            if (inf_dn) nd.ch.push_back({j, std::ceil(v), up}); else nd.ch.push_back({j, lo, std::floor(v)});
            if (inf_dn) set_bound(j, std::ceil(v), up); else set_bound(j, lo, std::floor(v));
            resolve = true;
            return -1;
        }
        double sc = score_of(obj_dn - node_obj, obj_up - node_obj);
        if (sc > best_score) { best_score = sc; best = j; }
    }
    return best;
}

void BranchAndCut::log_progress(size_t open, double bound, bool force) {
    double t = timer_.seconds();
    if (!force && t - last_log_ < 1.0) return;
    last_log_ = t;
    double inc = has_inc_ ? inc_obj_ / sc_.obj : kInf;
    double b = bound / sc_.obj;
    double gap = has_inc_ ? std::fabs(inc - b) / std::max(1e-9, std::fabs(inc)) : kInf;
    NLOG("  %9lld nodes %8zu open  incumbent %16.9e  bound %16.9e  gap %8.4f%%  %7.1fs\n",
         nodes_, open, inc, b, 100 * gap, t);
}

MipResult BranchAndCut::run() {
    timer_.reset();
    n_ = red_.n;
    sc_ = compute_scaling(red_, /*keep_int_cols=*/true);
    sm_ = apply_scaling(red_, sc_);
    isint_.assign(n_, 0);
    for (int j = 0; j < n_; ++j) if (red_.is_int[j]) { isint_[j] = 1; ints_.push_back(j); }
    ps_dn_.assign(n_, 0); ps_up_.assign(n_, 0); pn_dn_.assign(n_, 0); pn_up_.assign(n_, 0);
    is_modified_.assign(n_, 0);
    // locks for rounding
    lock_dn_.assign(n_, 0); lock_up_.assign(n_, 0);
    for (int j = 0; j < n_; ++j)
        for (int p = sm_.A.colptr[j]; p < sm_.A.colptr[j + 1]; ++p) {
            int i = sm_.A.rowidx[p];
            double a = sm_.A.val[p];
            bool hl = is_finite(sm_.row_lo[i]), hu = is_finite(sm_.row_up[i]);
            if ((a > 0 && hl) || (a < 0 && hu)) lock_dn_[j]++;
            if ((a > 0 && hu) || (a < 0 && hl)) lock_up_[j]++;
        }
    // integral objective?
    int_obj_ = true;
    for (int j = 0; j < n_ && int_obj_; ++j) {
        double c = red_.c[j];
        if (c == 0.0) continue;
        if (!isint_[j] || std::fabs(c - std::round(c)) > 1e-12) int_obj_ = false;
    }
    obj_step_ = sc_.obj;  // one unit of the unscaled objective

    lp_.load(sm_);
    root_lo_ = sm_.col_lo; root_up_ = sm_.col_up;
    SimplexOptions so;
    so.time_limit = opt_.time_limit;
    SimplexResult r = lp_.solve(so);
    if (r == SimplexResult::Infeasible) { res_.status = Status::Infeasible; return res_; }
    if (r == SimplexResult::DualInfeasible) { res_.status = Status::InfeasibleOrUnbounded; return res_; }
    if (r != SimplexResult::Optimal) { res_.status = (r == SimplexResult::TimeLimit) ? Status::TimeLimit : Status::NumericalError; return res_; }
    double root_obj = lp_.objective();
    res_.root_bound = root_obj / sc_.obj;
    NLOG("MIP: %d rows, %d cols (%zu integer)  root LP %.10e  (%lld simplex iters, %.2fs)%s\n",
         red_.m, n_, ints_.size(), root_obj / sc_.obj, lp_.iterations(), timer_.seconds(),
         int_obj_ ? "  [integral objective]" : "");

    if (opt_.heuristics) {
        simple_rounding(lp_.primal());
        diving(60);
        if (!has_inc_) fix_and_lp(lp_.primal());
        SimplexResult rr = solve_lp(kInf);
        (void)rr;
    }
    if (opt_.cuts) {
        int nc = root_cuts();
        if (nc < 0) { res_.status = Status::Infeasible; return res_; }
        ncuts_ = nc;
        root_obj = lp_.objective();
        if (opt_.heuristics && nc > 0) { simple_rounding(lp_.primal()); diving(60); solve_lp(kInf); }
    }
    res_.root_bound_after_cuts = root_obj / sc_.obj;
    root_lo_.resize(n_); root_up_.resize(n_);

    std::priority_queue<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>, NodeCmp> heap;
    auto root = std::make_shared<Node>();
    root->bound = root_obj;
    std::shared_ptr<Node> cur = root;
    bool cur_is_child_of_lp = true;  // LP currently holds this node's parent basis (no reload needed)
    double global_bound = root_obj;

    while (true) {
        if (!cur) {
            while (!heap.empty() && prunable(heap.top()->bound)) heap.pop();
            if (heap.empty()) break;
            cur = heap.top(); heap.pop();
            cur_is_child_of_lp = false;
        }
        global_bound = cur->bound;
        if (!heap.empty()) global_bound = std::min(global_bound, heap.top()->bound);
        if (has_inc_) {
            double gap = (inc_obj_ - global_bound) / std::max(1e-9, std::fabs(inc_obj_));
            if (gap <= opt_.rel_gap || inc_obj_ - global_bound <= opt_.abs_gap * sc_.obj) break;
        }
        if (nodes_ >= opt_.node_limit || timer_.seconds() > opt_.time_limit) break;
        log_progress(heap.size(), global_bound, false);

        Node& nd = *cur;
        if (prunable(nd.bound)) { cur.reset(); continue; }
        apply(nd);
        if (!cur_is_child_of_lp && nd.basis) lp_.set_basis(*nd.basis);
        ++nodes_;
        bool resolve = true;
        double obj = 0;
        std::vector<double> x;
        int bvar = -1;
        bool node_done = false;
        while (resolve) {
            SimplexResult rr = solve_lp(cutoff());
            if (rr == SimplexResult::Infeasible || rr == SimplexResult::Cutoff) {
                if (nd.br_var >= 0 && rr == SimplexResult::Cutoff)
                    update_pseudocost(nd.br_var, nd.br_dir, nd.br_frac, cutoff() - nd.parent_obj);
                node_done = true; break;
            }
            if (rr != SimplexResult::Optimal) { trouble_ = true; node_done = true; break; }
            obj = lp_.objective();
            if (nd.br_var >= 0) { update_pseudocost(nd.br_var, nd.br_dir, nd.br_frac, obj - nd.parent_obj); nd.br_var = -1; }
            if (prunable(obj)) { node_done = true; break; }
            x = lp_.primal();
            bool anyfrac = false;
            for (int j : ints_) if (is_frac(x[j])) { anyfrac = true; break; }
            if (!anyfrac) { try_incumbent(x, "LP integral"); node_done = true; break; }
            if (opt_.heuristics) {
                simple_rounding(x);
                if (nodes_ % 200 == 0) { diving(40); solve_lp(cutoff()); x = lp_.primal(); obj = lp_.objective(); }
                if (!has_inc_ && nodes_ % 50 == 0) { fix_and_lp(x); solve_lp(cutoff()); x = lp_.primal(); obj = lp_.objective(); }
                if (prunable(obj)) { node_done = true; break; }
            }
            bool infeasible;
            bvar = select_branch(x, obj, nd.depth, nd, resolve, infeasible);
            if (infeasible) { node_done = true; break; }
            if (resolve) continue;
            if (bvar < 0) { node_done = true; break; }
            // strong branching moved the LP: restore this node's LP optimum before branching
            SimplexResult r2 = solve_lp(kInf);
            if (r2 != SimplexResult::Optimal) { trouble_ = true; node_done = true; break; }
            x = lp_.primal(); obj = lp_.objective();
            if (!is_frac(x[bvar])) {
                bvar = -1;
                for (int j : ints_) if (is_frac(x[j])) { bvar = j; break; }
                if (bvar < 0) { try_incumbent(x, "LP integral"); node_done = true; break; }
            }
        }
        if (node_done) { cur.reset(); continue; }

        // ---- branch
        const double v = x[bvar], f = frac(v);
        auto basis = std::make_shared<DualSimplex::Basis>(lp_.basis());
        auto dn = std::make_shared<Node>();
        auto up = std::make_shared<Node>();
        dn->ch = nd.ch; up->ch = nd.ch;
        dn->ch.push_back({bvar, lp_.lo(bvar), std::floor(v)});
        up->ch.push_back({bvar, std::ceil(v), lp_.up(bvar)});
        dn->bound = up->bound = obj;
        dn->depth = up->depth = nd.depth + 1;
        dn->basis = up->basis = basis;
        dn->br_var = up->br_var = bvar;
        dn->br_dir = -1; up->br_dir = +1;
        dn->br_frac = f; up->br_frac = 1 - f;
        dn->parent_obj = up->parent_obj = obj;
        dn->estimate = obj + f * pseudo(bvar, -1);
        up->estimate = obj + (1 - f) * pseudo(bvar, +1);
        auto first = dn->estimate <= up->estimate ? dn : up;
        auto second = (first == dn) ? up : dn;
        heap.push(second);
        cur = first;
        cur_is_child_of_lp = true;
    }

    const bool exhausted = !cur && heap.empty();
    double bound = exhausted ? (has_inc_ ? inc_obj_ : kInf) : global_bound;
    if (has_inc_ && bound > inc_obj_) bound = inc_obj_;
    log_progress(heap.size(), has_inc_ ? std::min(bound, inc_obj_) : bound, true);
    res_.nodes = nodes_;
    res_.lp_iterations = lp_.iterations();
    res_.cuts = ncuts_;
    res_.trouble = trouble_;
    if (has_inc_) {
        std::vector<double> xu = inc_x_;
        unscale_primal(sc_, xu);
        for (int j : ints_) xu[j] = std::round(xu[j]);
        res_.x = xu;
        res_.obj = inc_obj_ / sc_.obj;
        res_.bound = bound / sc_.obj;
        double gap = (res_.obj - res_.bound) / std::max(1e-9, std::fabs(res_.obj));
        bool closed = exhausted || gap <= opt_.rel_gap || res_.obj - res_.bound <= opt_.abs_gap;
        res_.status = (closed && !trouble_) ? Status::Optimal : Status::Feasible;
        if (!closed && nodes_ >= opt_.node_limit) res_.status = Status::Feasible;
    } else {
        res_.bound = bound / sc_.obj;
        if (exhausted && !trouble_) res_.status = Status::Infeasible;
        else res_.status = (timer_.seconds() > opt_.time_limit) ? Status::TimeLimit
                         : (nodes_ >= opt_.node_limit ? Status::NodeLimit : Status::NumericalError);
    }
    return res_;
}

}  // namespace

MipResult solve_mip(const Model& mdl, const MipOptions& opt) {
    BranchAndCut bc(mdl, opt);
    return bc.run();
}

}  // namespace nirnay
