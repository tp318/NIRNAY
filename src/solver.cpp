#include "solver.hpp"

#include <algorithm>
#include <cmath>

#include "ipm.hpp"
#include "mip.hpp"
#include "pdhg.hpp"
#include "scaling.hpp"
#include "simplex.hpp"

namespace nirnay {

int g_verbose = 1;

namespace {

// Largest violation measured in the UNSCALED space, relative to bound magnitude.
void unscaled_violation(const DualSimplex& ds, const Model& sm, const Scaling& sc, double& pviol, double& dviol) {
    pviol = 0; dviol = 0;
    const auto& x = ds.values();
    const int n = sm.n, m = sm.m;
    for (int j = 0; j < n + m; ++j) {
        double lo = ds.lo(j), up = ds.up(j), v = x[j];
        double viol = std::max(lo - v, v - up);
        if (viol <= 0) continue;
        double unscale = j < n ? sc.col[j] : 1.0 / sc.row[j - n];
        double mag = 1.0 + std::fabs(unscale) * std::max(is_finite(lo) ? std::fabs(lo) : 0.0, is_finite(up) ? std::fabs(up) : 0.0);
        pviol = std::max(pviol, viol * unscale / mag);
    }
    auto d = ds.reduced_costs();
    double cmax = 0;
    for (int j = 0; j < n; ++j) cmax = std::max(cmax, std::fabs(sm.c[j] / (sc.col[j] * sc.obj)));
    for (int j = 0; j < n; ++j) {
        int st = ds.status_of(j);
        double dj = d[j] / (sc.col[j] * sc.obj);
        double viol = 0;
        if (st == DualSimplex::AT_LO) viol = std::max(0.0, -dj);
        else if (st == DualSimplex::AT_UP) viol = std::max(0.0, dj);
        else if (st == DualSimplex::AT_ZERO) viol = std::fabs(dj);
        dviol = std::max(dviol, viol / (1.0 + cmax));
    }
}

// Crossover basis from an approximate primal-dual point (x, y) of the UNSCALED model `red`,
// expressed for the simplex on the scaled model `sm`. Each of the n + m simplex variables
// (columns, then row activities) gets a basic-ness score: relative distance to its nearest bound
// minus the relative size of its reduced cost (dual). The m highest scores form the basis; the
// rest sit at their nearest bound. The simplex repairs a singular guess with logical columns.
DualSimplex::Basis crossover_basis(const Model& sm, const Scaling& sc, const std::vector<double>& x,
                                   const std::vector<double>& y) {
    const int m = sm.m, n = sm.n, N = n + m;
    std::vector<double> xs(n), ys(m), act(m, 0.0), rc(n);
    for (int j = 0; j < n; ++j) xs[j] = x[j] / sc.col[j];
    for (int i = 0; i < m; ++i) ys[i] = y[i] * sc.obj / sc.row[i];
    for (int j = 0; j < n; ++j) {
        double d = sm.c[j];
        for (int p = sm.A.colptr[j]; p < sm.A.colptr[j + 1]; ++p) {
            act[sm.A.rowidx[p]] += sm.A.val[p] * xs[j];
            d -= sm.A.val[p] * ys[sm.A.rowidx[p]];
        }
        rc[j] = d;
    }
    double cmax = 1.0;
    for (int j = 0; j < n; ++j) cmax = std::max(cmax, std::fabs(sm.c[j]));
    std::vector<double> score(N), val(N), lo(N), up(N);
    for (int k = 0; k < N; ++k) {
        const bool col = k < n;
        lo[k] = col ? sm.col_lo[k] : sm.row_lo[k - n];
        up[k] = col ? sm.col_up[k] : sm.row_up[k - n];
        val[k] = col ? xs[k] : act[k - n];
        const double dual = col ? rc[k] : ys[k - n];
        double dist = std::min(is_finite(lo[k]) ? val[k] - lo[k] : kInf, is_finite(up[k]) ? up[k] - val[k] : kInf);
        dist = std::max(dist, 0.0);
        if (lo[k] == up[k]) score[k] = -kInf;                       // fixed: never basic by choice
        else score[k] = std::min(dist, 1e6) / (1.0 + std::fabs(val[k])) - std::fabs(dual) / cmax;
    }
    std::vector<int> ord(N);
    for (int k = 0; k < N; ++k) ord[k] = k;
    std::nth_element(ord.begin(), ord.begin() + m, ord.end(), [&](int a, int b) { return score[a] > score[b]; });
    DualSimplex::Basis b;
    b.head.assign(ord.begin(), ord.begin() + m);
    b.status.assign(N, DualSimplex::AT_LO);
    for (int k : b.head) b.status[k] = DualSimplex::BASIC;
    for (int k = 0; k < N; ++k) {
        if (b.status[k] == DualSimplex::BASIC) continue;
        const bool fl = is_finite(lo[k]), fu = is_finite(up[k]);
        if (fl && fu && lo[k] == up[k]) b.status[k] = DualSimplex::FIXED;
        else if (fl && fu) b.status[k] = (val[k] - lo[k] <= up[k] - val[k]) ? DualSimplex::AT_LO : DualSimplex::AT_UP;
        else if (fl) b.status[k] = DualSimplex::AT_LO;
        else if (fu) b.status[k] = DualSimplex::AT_UP;
        else b.status[k] = DualSimplex::AT_ZERO;
    }
    return b;
}

LpSolution solve_simplex_lp(const Model& sm, const Scaling& sc, const SolverOptions& opt,
                            const DualSimplex::Basis* warm = nullptr) {
    LpSolution s;
    DualSimplex ds;
    ds.load(sm);
    if (warm) ds.set_basis(*warm);
    SimplexOptions so;
    so.time_limit = opt.time_limit;
    SimplexResult r = ds.solve(so);
    // Unscaled cleanup: tighten tolerances and warm-start until the solution is
    // accurate in the original (unscaled) space.
    for (int round = 0; round < 4 && r == SimplexResult::Optimal; ++round) {
        double pv, dv;
        unscaled_violation(ds, sm, sc, pv, dv);
        if (pv <= 1e-9 && dv <= 1e-9) break;
        if (pv > 1e-9) so.tol_primal = std::max(1e-13, so.tol_primal * std::min(0.1, 1e-9 / pv));
        if (dv > 1e-9) so.tol_dual = std::max(1e-13, so.tol_dual * std::min(0.1, 1e-9 / dv));
        so.perturb = false;
        NLOG("Dual simplex: unscaled infeasibility p %.1e d %.1e -> tolerances %.0e / %.0e, warm restart\n",
             pv, dv, so.tol_primal, so.tol_dual);
        r = ds.solve(so);
    }
    s.iterations = (int)ds.iterations();
    switch (r) {
        case SimplexResult::Optimal:
            s.status = Status::Optimal;
            s.x = ds.primal(); s.y = ds.row_duals(); s.z = ds.reduced_costs();
            s.obj = ds.objective();
            break;
        case SimplexResult::Infeasible: s.status = Status::Infeasible; break;
        case SimplexResult::DualInfeasible: {
            // Distinguish unbounded from infeasible: solve the feasibility problem.
            Model f = sm;
            std::fill(f.c.begin(), f.c.end(), 0.0);
            DualSimplex d2;
            d2.load(f);
            SimplexResult r2 = d2.solve(so);
            s.status = (r2 == SimplexResult::Optimal) ? Status::Unbounded
                     : (r2 == SimplexResult::Infeasible ? Status::Infeasible : Status::InfeasibleOrUnbounded);
            break;
        }
        case SimplexResult::TimeLimit: s.status = Status::TimeLimit; break;
        case SimplexResult::IterLimit: s.status = Status::IterationLimit; break;
        default: s.status = Status::NumericalError; break;
    }
    NLOG("Dual simplex: %s after %lld iterations\n", status_name(s.status), ds.iterations());
    return s;
}

KktReport evaluate_kkt(const Model& mdl, const std::vector<double>& x, const std::vector<double>& y,
                       std::vector<double>& z) {
    KktReport k;
    k.available = true;
    const int n = mdl.n, m = mdl.m;
    std::vector<double> qx(n, 0.0);
    if (mdl.has_q()) mdl.Q.mul_add(x.data(), qx.data());
    z.assign(n, 0.0);
    for (int j = 0; j < n; ++j) z[j] = mdl.c[j] + qx[j];
    mdl.A.mul_t_add(y.data(), z.data(), -1.0);
    // relative primal infeasibility: violation / (1 + |violated bound|)
    auto ax = mdl.activities(x);
    double pinf = 0;
    for (int i = 0; i < m; ++i) {
        if (ax[i] < mdl.row_lo[i]) pinf = std::max(pinf, (mdl.row_lo[i] - ax[i]) / (1 + std::fabs(mdl.row_lo[i])));
        if (ax[i] > mdl.row_up[i]) pinf = std::max(pinf, (ax[i] - mdl.row_up[i]) / (1 + std::fabs(mdl.row_up[i])));
    }
    for (int j = 0; j < n; ++j) {
        if (x[j] < mdl.col_lo[j]) pinf = std::max(pinf, (mdl.col_lo[j] - x[j]) / (1 + std::fabs(mdl.col_lo[j])));
        if (x[j] > mdl.col_up[j]) pinf = std::max(pinf, (x[j] - mdl.col_up[j]) / (1 + std::fabs(mdl.col_up[j])));
    }
    k.primal_inf = pinf;
    double cmax = 0;
    for (int j = 0; j < n; ++j) cmax = std::max(cmax, std::fabs(mdl.c[j]));
    // relative dual infeasibility: wrong-signed multipliers of absent bounds
    double dinf = 0, dobj = mdl.obj_offset, xqx = 0;
    for (int j = 0; j < n; ++j) xqx += x[j] * qx[j];
    dobj -= 0.5 * xqx;
    for (int i = 0; i < m; ++i) {
        const double yi = y[i];
        if (yi > 0) {
            if (is_finite(mdl.row_lo[i])) dobj += yi * mdl.row_lo[i]; else dinf = std::max(dinf, yi / (1 + cmax));
        } else if (yi < 0) {
            if (is_finite(mdl.row_up[i])) dobj += yi * mdl.row_up[i]; else dinf = std::max(dinf, -yi / (1 + cmax));
        }
    }
    for (int j = 0; j < n; ++j) {
        const double zj = z[j];
        const double sc = 1 + std::fabs(mdl.c[j]);
        if (zj > 0) {
            if (is_finite(mdl.col_lo[j])) dobj += zj * mdl.col_lo[j]; else dinf = std::max(dinf, zj / sc);
        } else if (zj < 0) {
            if (is_finite(mdl.col_up[j])) dobj += zj * mdl.col_up[j]; else dinf = std::max(dinf, -zj / sc);
        }
    }
    const double pobj = mdl.objective(x);
    k.dual_inf = dinf;
    k.dual_obj = dobj;
    k.rel_gap = std::fabs(pobj - dobj) / (1.0 + std::fabs(pobj));
    return k;
}

void finalize(const Model& mdl, SolveResult& r) {
    if (r.x.size() != (size_t)mdl.n) return;
    r.viol = check_primal(mdl, r.x);
    double obj = mdl.objective(r.x);
    r.objective = mdl.maximize ? -obj : obj;
    if (r.duals_valid && mdl.num_int() == 0) {
        r.kkt = evaluate_kkt(mdl, r.x, r.y, r.z);
        if (mdl.maximize) {
            r.kkt.dual_obj = -r.kkt.dual_obj;
            for (double& v : r.y) v = -v;
            for (double& v : r.z) v = -v;
        }
    }
}

}  // namespace

SolveResult solve(const Model& model, const SolverOptions& opt) {
    Timer total;
    SolveResult res;
    res.orig_rows = model.m;
    res.orig_cols = model.n;
    res.orig_nnz = model.A.nnz();
    res.int_cols = model.num_int();
    const bool is_mip = res.int_cols > 0;

    // ---------------- presolve
    Timer tp;
    Presolver pre;
    Status ps = pre.run(model, opt.presolve);
    res.time_presolve = tp.seconds();
    res.pstats = pre.stats();
    if (ps != Status::NotSolved) {
        res.status = ps;
        res.method = "presolve";
        res.time_total = total.seconds();
        NLOG("Presolve: problem detected %s\n", status_name(ps));
        return res;
    }
    const Model& red = pre.reduced();
    res.red_rows = red.m;
    res.red_cols = red.n;
    NLOG("Presolve: %d rows, %d cols -> %d rows, %d cols  (singleton %d, forcing %d, redundant %d, fixed %d, empty %d) %.3fs\n",
         model.m, model.n, red.m, red.n, res.pstats.singleton_rows, res.pstats.forcing_rows,
         res.pstats.redundant_rows, res.pstats.fixed_cols, res.pstats.empty_cols, res.time_presolve);

    std::vector<double> xr, yr;
    Timer ts;
    bool pdhg_needs_check = false;
    bool routed_to_pdhg = false;
    double pdhg_tol_used = 1e-4;
    if (red.n == 0) {
        res.status = Status::Optimal;
        res.method = "presolve";
        yr.assign(red.m, 0.0);
    } else if (is_mip) {
        if (red.has_q()) {
            res.status = Status::NotSolved;
            res.note = "MIQP is on the roadmap; current release supports LP, convex QP and MILP";
        } else {
            MipOptions mo;
            mo.rel_gap = opt.mip_gap;
            mo.node_limit = opt.node_limit;
            mo.time_limit = std::max(0.0, opt.time_limit - total.seconds());
            mo.cuts = opt.cuts;
            mo.heuristics = opt.heuristics;
            mo.sb_candidates = opt.sb_candidates;
            mo.sb_reliability = opt.sb_reliability;
            mo.sb_iters = opt.sb_iters;
            mo.cut_rounds = opt.cut_rounds;
            MipResult mr = solve_mip(red, mo);
            res.method = "branch-and-cut";
            res.status = mr.status;
            res.nodes = mr.nodes;
            res.iterations = mr.lp_iterations;
            res.cuts_added = mr.cuts;
            res.bound = model.maximize ? -mr.bound : mr.bound;
            if (!mr.x.empty()) {
                xr = mr.x;
                res.mip_gap = std::fabs(mr.obj - mr.bound) / std::max(1e-9, std::fabs(mr.obj));
            }
            char buf[160];
            std::snprintf(buf, sizeof buf, "root LP bound %.10g, after cuts %.10g%s",
                          model.maximize ? -mr.root_bound : mr.root_bound,
                          model.maximize ? -mr.root_bound_after_cuts : mr.root_bound_after_cuts,
                          mr.trouble ? "; numerical trouble in some node LPs" : "");
            res.note = buf;
        }
    } else {
        std::string method = opt.method;
        if (method == "auto") method = "ipm";
        // Heterogeneous routing (auto, LP only): the IPM declines when its symbolic factor would exceed
        // kRouteFactorNnz, and the model goes to PDHG (GPU if present) + simplex crossover instead. The
        // threshold comes from Netlib: every LP where PDHG + crossover beat the IPM route had nnz(L) > 0.5M
        // (maros-r7 1.2M, pilot87 0.56M, mcf_25x25 3.6M); every LP the IPM handled well had nnz(L) < 0.2M.
        const long long kRouteFactorNnz = 500000;
        bool ipm_uncapped = false;
        for (int pass = 0; pass < 3; ++pass) {
        if (method == "pdhg") {
            if (red.has_q()) {
                res.status = Status::NotSolved;
                res.note = "PDHG engine currently supports LP only";
            } else {
                PdhgOptions po;
                po.max_iter = opt.pdhg_max_iter;
                // routed: PDHG gets half the budget, the uncapped IPM keeps the rest as a fallback
                po.time_limit = routed_to_pdhg ? 0.5 * opt.time_limit : opt.time_limit;
                po.device = opt.device;
                po.alg = opt.pdhg_alg;
                po.reflection = opt.pdhg_reflection;
                po.check_every = opt.pdhg_check;
                po.compress = opt.pdhg_compress;
                po.graphs = opt.pdhg_graphs;
                po.precision = opt.pdhg_precision;
                const bool xo = opt.crossover == "on" || (opt.crossover == "auto" && (long long)red.m + red.n <= 500000);
                po.strict = !xo;   // crossover only needs the active set; it certifies the final answer itself
                // Crossover only needs the active set, so 1e-4 suffices; a standalone PDHG answer defaults to 1e-6.
                po.tol = opt.pdhg_tol > 0 ? opt.pdhg_tol : (xo ? 1e-4 : 1e-6);
                pdhg_tol_used = po.tol;
                PdhgInfo info;
                LpSolution s = solve_pdhg(red, po, info);
                res.method = "pdhg-" + info.device;
                res.status = s.status;
                res.iterations = s.iterations;
                xr = s.x; yr = s.y;
                char buf[200];
                std::snprintf(buf, sizeof buf, "PDHG %s device %s [%s]; restarts %d; rel. primal %.2e dual %.2e gap %.2e; %.4f ms/iter",
                              po.alg.c_str(), info.device.c_str(), info.matrix_format.c_str(), info.restarts, s.pinf, s.dinf, s.gap, info.ms_per_iter);
                res.note = buf;
                // Crossover: finish exactly with the dual simplex warm-started from a basis guessed from the
                // PDHG point. Runs whenever PDHG produced a point, including after a time/iteration limit.
                // routed: crossover stops at 75% of the budget so the IPM fallback keeps a share
                const double left = (routed_to_pdhg ? 0.75 : 1.0) * opt.time_limit - ts.seconds();
                if (xo && !s.x.empty() && left > 0) {
                    Timer tx;
                    Scaling sc = compute_scaling(red, false);
                    Model sm = apply_scaling(red, sc);
                    DualSimplex::Basis wb = crossover_basis(sm, sc, s.x, s.y);
                    SolverOptions o2 = opt;
                    o2.time_limit = left;
                    LpSolution s2 = solve_simplex_lp(sm, sc, o2, &wb);
                    NLOG("Crossover: %s after %d simplex iterations, %.3fs\n", status_name(s2.status), s2.iterations, tx.seconds());
                    if (s2.status == Status::Optimal) {
                        xr = s2.x; yr = s2.y;
                        std::vector<double> zr = s2.z;
                        unscale_primal(sc, xr);
                        unscale_dual(sc, yr, zr);
                        res.status = Status::Optimal;
                        res.method += "+crossover";
                        res.iterations += s2.iterations;
                        char b2[96];
                        std::snprintf(b2, sizeof b2, "; crossover %d simplex iters %.3fs", s2.iterations, tx.seconds());
                        res.note += b2;
                    } else {
                        res.note += std::string("; crossover ") + status_name(s2.status);
                        pdhg_needs_check = res.status == Status::Optimal;
                    }
                } else if (xo) {
                    pdhg_needs_check = res.status == Status::Optimal;
                }
            }
        } else if (method == "ipm") {
            Scaling sc = compute_scaling(red, false);
            Model sm = apply_scaling(red, sc);
            IpmOptions io;
            io.tol = opt.tol;
            io.time_limit = opt.time_limit;
            io.log_scale = (red.maximize ? -1.0 : 1.0) / sc.obj;
            if (opt.method == "auto" && !red.has_q() && !ipm_uncapped) io.max_factor_nnz = kRouteFactorNnz;
            io.time_limit = std::max(0.0, opt.time_limit - ts.seconds());
            LpSolution s = solve_ipm(sm, io);
            if (io.max_factor_nnz > 0 && s.status == Status::NotSolved) {
                method = "pdhg";
                routed_to_pdhg = true;
                continue;
            }
            res.method = "ipm";
            res.status = s.status;
            res.iterations = s.iterations;
            if (s.status == Status::Optimal || red.has_q() || opt.method != "auto") {
                xr = s.x; yr = s.y;
                std::vector<double> zr = s.z;
                unscale_primal(sc, xr);
                unscale_dual(sc, yr, zr);
            } else {
                // IPM could not certify: fall back to the dual simplex, which returns
                // exact infeasibility / unboundedness certificates.
                NLOG("IPM ended with %s; switching to dual simplex\n", status_name(s.status));
                SolverOptions o2 = opt;
                o2.time_limit = std::max(0.0, opt.time_limit - ts.seconds());   // what is left, not the full limit
                LpSolution s2 = solve_simplex_lp(sm, sc, o2);
                res.method = "ipm->dual-simplex";
                res.status = s2.status;
                res.iterations += s2.iterations;
                if (s2.status == Status::Optimal) {
                    xr = s2.x; yr = s2.y;
                    std::vector<double> zr = s2.z;
                    unscale_primal(sc, xr);
                    unscale_dual(sc, yr, zr);
                }
            }
        } else if (method == "simplex") {
            if (red.has_q()) {
                res.status = Status::NotSolved;
                res.note = "simplex does not support QP; use --method ipm";
            } else {
                Scaling sc = compute_scaling(red, false);
                Model sm = apply_scaling(red, sc);
                LpSolution s = solve_simplex_lp(sm, sc, opt);
                res.method = "dual-simplex";
                res.status = s.status;
                res.iterations = s.iterations;
                if (s.status == Status::Optimal) {
                    xr = s.x; yr = s.y;
                    std::vector<double> zr = s.z;
                    unscale_primal(sc, xr);
                    unscale_dual(sc, yr, zr);
                }
            }
        } else {
            res.status = Status::NotSolved;
            res.note = "unknown method '" + method + "'";
        }
        if (method == "pdhg" && routed_to_pdhg && (res.status != Status::Optimal || pdhg_needs_check) &&
            opt.time_limit - ts.seconds() > 0.5) {
            NLOG("Routing: PDHG + crossover not certified (%s); falling back to the IPM\n", status_name(res.status));
            method = "ipm";
            ipm_uncapped = true;
            pdhg_needs_check = false;
            xr.clear(); yr.clear();
            continue;
        }
        break;
        }
        if (routed_to_pdhg) res.note = "routed to PDHG (large IPM factor); " + res.note;
    }
    res.time_solve = ts.seconds();

    if (!xr.empty() || red.n == 0) {
        if (xr.empty()) xr.assign(red.n, 0.0);
        pre.postsolve(xr, yr, res.x, res.y, res.duals_valid);
        finalize(model, res);
    }
    // PDHG stopped on the l2 criterion for crossover, and crossover did not finish: keep "optimal" only if
    // the independent original-model check agrees at the requested tolerance.
    // (An unavailable check counts as a failure: greenbeb reached this point with no KKT report and a row
    // violated by 0.63 when crossover ran out of time.)
    if (pdhg_needs_check && (!res.kkt.available ||
        std::max(res.kkt.primal_inf, std::max(res.kkt.dual_inf, res.kkt.rel_gap)) > 10 * pdhg_tol_used)) {
        res.status = Status::Feasible;
        res.note += "; approximate (original-model KKT above tolerance)";
    }
    res.time_total = total.seconds();
    return res;
}

}  // namespace nirnay
