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

LpSolution solve_simplex_lp(const Model& sm, const Scaling& sc, const SolverOptions& opt) {
    LpSolution s;
    DualSimplex ds;
    ds.load(sm);
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
        if (method == "pdhg") {
            if (red.has_q()) {
                res.status = Status::NotSolved;
                res.note = "PDHG engine currently supports LP only";
            } else {
                PdhgOptions po;
                po.tol = opt.pdhg_tol;
                po.max_iter = opt.pdhg_max_iter;
                po.time_limit = opt.time_limit;
                po.device = opt.device;
                PdhgInfo info;
                LpSolution s = solve_pdhg(red, po, info);
                res.method = "pdhg-" + info.device;
                res.status = s.status;
                res.iterations = s.iterations;
                xr = s.x; yr = s.y;
                char buf[200];
                std::snprintf(buf, sizeof buf, "PDHG device %s; restarts %d; rel. primal %.2e dual %.2e gap %.2e; %.3f ms/iter",
                              info.device.c_str(), info.restarts, s.pinf, s.dinf, s.gap, info.ms_per_iter);
                res.note = buf;
            }
        } else if (method == "ipm") {
            Scaling sc = compute_scaling(red, false);
            Model sm = apply_scaling(red, sc);
            IpmOptions io;
            io.tol = opt.tol;
            io.time_limit = opt.time_limit;
            io.log_scale = (red.maximize ? -1.0 : 1.0) / sc.obj;
            LpSolution s = solve_ipm(sm, io);
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
                LpSolution s2 = solve_simplex_lp(sm, sc, opt);
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
    }
    res.time_solve = ts.seconds();

    if (!xr.empty() || red.n == 0) {
        if (xr.empty()) xr.assign(red.n, 0.0);
        pre.postsolve(xr, yr, res.x, res.y, res.duals_valid);
        finalize(model, res);
    }
    res.time_total = total.seconds();
    return res;
}

}  // namespace nirnay
