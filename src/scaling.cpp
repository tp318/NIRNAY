#include "scaling.hpp"

#include <algorithm>
#include <cmath>

namespace nirnay {

namespace {
double pow2_round(double s) {
    if (!(s > 0) || !std::isfinite(s)) return 1.0;
    int e = (int)std::lround(std::log2(s));
    e = std::max(-60, std::min(60, e));
    return std::ldexp(1.0, e);
}
}  // namespace

Scaling compute_scaling(const Model& mdl, bool keep_int_cols, int passes) {
    Scaling s;
    const int m = mdl.m, n = mdl.n;
    s.row.assign(m, 1.0);
    s.col.assign(n, 1.0);
    const auto& A = mdl.A;
    if (A.nnz() == 0) return s;

    double amin = kInf, amax = 0;
    for (double v : A.val) { double a = std::fabs(v); if (a > 0) { amin = std::min(amin, a); amax = std::max(amax, a); } }
    if (amax / amin < 16.0 && amax <= 16.0 && amin >= 1.0 / 16.0) {
        // already well scaled; only objective scaling below
    } else {
        std::vector<double> rmin(m), rmax(m);
        auto col_fixed = [&](int j) { return keep_int_cols && !mdl.is_int.empty() && mdl.is_int[j]; };
        for (int pass = 0; pass < passes; ++pass) {
            // rows
            std::fill(rmin.begin(), rmin.end(), kInf);
            std::fill(rmax.begin(), rmax.end(), 0.0);
            for (int j = 0; j < n; ++j)
                for (int p = A.colptr[j]; p < A.colptr[j + 1]; ++p) {
                    double a = std::fabs(A.val[p]) * s.col[j];
                    if (a == 0) continue;
                    int i = A.rowidx[p];
                    rmin[i] = std::min(rmin[i], a); rmax[i] = std::max(rmax[i], a);
                }
            for (int i = 0; i < m; ++i)
                if (rmax[i] > 0) s.row[i] = 1.0 / std::sqrt(rmin[i] * rmax[i]);
            // columns
            for (int j = 0; j < n; ++j) {
                if (col_fixed(j)) continue;
                double cmin = kInf, cmax = 0;
                for (int p = A.colptr[j]; p < A.colptr[j + 1]; ++p) {
                    double a = std::fabs(A.val[p]) * s.row[A.rowidx[p]];
                    if (a == 0) continue;
                    cmin = std::min(cmin, a); cmax = std::max(cmax, a);
                }
                if (cmax > 0) s.col[j] = 1.0 / std::sqrt(cmin * cmax);
            }
        }
        // equilibrate rows then columns to max |a| = 1
        std::fill(rmax.begin(), rmax.end(), 0.0);
        for (int j = 0; j < n; ++j)
            for (int p = A.colptr[j]; p < A.colptr[j + 1]; ++p) {
                int i = A.rowidx[p];
                rmax[i] = std::max(rmax[i], std::fabs(A.val[p]) * s.col[j] * s.row[i]);
            }
        for (int i = 0; i < m; ++i) {
            if (rmax[i] > 0) s.row[i] /= rmax[i];
            s.row[i] = pow2_round(s.row[i]);
        }
        for (int j = 0; j < n; ++j) {
            if (!col_fixed(j)) {
                double cmax = 0;
                for (int p = A.colptr[j]; p < A.colptr[j + 1]; ++p)
                    cmax = std::max(cmax, std::fabs(A.val[p]) * s.row[A.rowidx[p]] * s.col[j]);
                if (cmax > 0) s.col[j] /= cmax;
            }
            s.col[j] = pow2_round(s.col[j]);
        }
    }
    // objective scaling: bring max |c'|, |Q'| to O(1)
    double cmax = 0;
    for (int j = 0; j < n; ++j) cmax = std::max(cmax, std::fabs(mdl.c[j] * s.col[j]));
    for (int j = 0; j < n; ++j)
        for (int p = mdl.Q.colptr.empty() ? 0 : mdl.Q.colptr[j]; !mdl.Q.colptr.empty() && p < mdl.Q.colptr[j + 1]; ++p)
            cmax = std::max(cmax, std::fabs(mdl.Q.val[p] * s.col[j] * s.col[mdl.Q.rowidx[p]]));
    if (cmax > 0) s.obj = pow2_round(1.0 / cmax);
    s.identity = (s.obj == 1.0);
    for (double v : s.row) if (v != 1.0) s.identity = false;
    for (double v : s.col) if (v != 1.0) s.identity = false;
    return s;
}

Model apply_scaling(const Model& mdl, const Scaling& s) {
    Model out = mdl;
    const int n = mdl.n, m = mdl.m;
    for (int j = 0; j < n; ++j) {
        for (int p = out.A.colptr[j]; p < out.A.colptr[j + 1]; ++p)
            out.A.val[p] *= s.row[out.A.rowidx[p]] * s.col[j];
        out.c[j] *= s.col[j] * s.obj;
        out.col_lo[j] /= s.col[j];
        out.col_up[j] /= s.col[j];
    }
    if (!out.Q.colptr.empty())
        for (int j = 0; j < n; ++j)
            for (int p = out.Q.colptr[j]; p < out.Q.colptr[j + 1]; ++p)
                out.Q.val[p] *= s.col[j] * s.col[out.Q.rowidx[p]] * s.obj;
    for (int i = 0; i < m; ++i) {
        out.row_lo[i] *= s.row[i];
        out.row_up[i] *= s.row[i];
    }
    out.obj_offset *= s.obj;
    return out;
}

void unscale_primal(const Scaling& s, std::vector<double>& x) {
    for (size_t j = 0; j < x.size(); ++j) x[j] *= s.col[j];
}

void unscale_dual(const Scaling& s, std::vector<double>& y, std::vector<double>& z) {
    for (size_t i = 0; i < y.size(); ++i) y[i] *= s.row[i] / s.obj;
    for (size_t j = 0; j < z.size(); ++j) z[j] /= (s.col[j] * s.obj);
}

}  // namespace nirnay
