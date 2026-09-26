#include "presolve.hpp"

#include <algorithm>
#include <cmath>

namespace nirnay {

Status Presolver::run(const Model& om, int level) {
    orig_ = &om;
    st_ = PresolveStats{};
    stack_.clear();
    forcing_used_ = false;
    const int n = om.n, m = om.m;
    const double ftol = 1e-9;
    const CscMatrix& A = om.A;
    const CscMatrix AT = A.transpose();
    const bool hasq = om.has_q();

    std::vector<double> lo = om.col_lo, up = om.col_up, rlo = om.row_lo, rup = om.row_up, c = om.c;
    double offset = om.obj_offset;
    std::vector<char> calive(n, 1), ralive(m, 1);
    std::vector<int> rcnt(m, 0), ccnt(n, 0);
    for (int j = 0; j < n; ++j) ccnt[j] = A.colptr[j + 1] - A.colptr[j];
    for (int i = 0; i < m; ++i) rcnt[i] = AT.colptr[i + 1] - AT.colptr[i];
    fix_val_.assign(n, 0.0);
    auto is_int = [&](int j) { return !om.is_int.empty() && om.is_int[j]; };

    auto fix = [&](int j, double v) {
        for (int p = A.colptr[j]; p < A.colptr[j + 1]; ++p) {
            int i = A.rowidx[p];
            if (!ralive[i]) continue;
            double d = A.val[p] * v;
            rlo[i] -= d; rup[i] -= d;
            rcnt[i]--;
        }
        offset += c[j] * v;
        if (hasq)
            for (int q = om.Q.colptr[j]; q < om.Q.colptr[j + 1]; ++q) {
                int i = om.Q.rowidx[q];
                if (i == j) offset += 0.5 * om.Q.val[q] * v * v;
                else if (calive[i]) c[i] += om.Q.val[q] * v;
            }
        calive[j] = 0;
        fix_val_[j] = v;
        stack_.push_back({Op::FixCol, -1, j, 0, lo[j], up[j], v, v});
        st_.cols_removed++;
    };
    auto drop_row = [&](int i) {
        ralive[i] = 0;
        for (int p = AT.colptr[i]; p < AT.colptr[i + 1]; ++p) {
            int j = AT.rowidx[p];
            if (calive[j]) ccnt[j]--;
        }
        stack_.push_back({Op::DropRow, i, -1, 0, 0, 0, 0, 0});
        st_.rows_removed++;
    };
    auto q_alive_offdiag = [&](int j, double& qjj) {
        qjj = 0;
        bool off = false;
        if (!hasq) return false;
        for (int q = om.Q.colptr[j]; q < om.Q.colptr[j + 1]; ++q) {
            int i = om.Q.rowidx[q];
            if (i == j) qjj += om.Q.val[q];
            else if (calive[i] && om.Q.val[q] != 0.0) off = true;
        }
        return off;
    };

    bool changed = true;
    for (int pass = 0; changed && pass < 50; ++pass) {
        changed = false;
        // ---- columns
        for (int j = 0; j < n; ++j) {
            if (!calive[j]) continue;
            if (is_int(j)) {
                double nl = std::ceil(lo[j] - 1e-9), nu = std::floor(up[j] + 1e-9);
                if (nl != lo[j] || nu != up[j]) { lo[j] = nl; up[j] = nu; }
            }
            if (lo[j] > up[j] + ftol * (1 + std::fabs(lo[j]))) return Status::Infeasible;
            if (is_finite(lo[j]) && is_finite(up[j]) && up[j] - lo[j] <= 1e-12 * std::max(1.0, std::fabs(lo[j]))) {
                fix(j, lo[j]); st_.fixed_cols++; changed = true; continue;
            }
            if (level >= 1 && ccnt[j] == 0) {
                double qjj;
                if (q_alive_offdiag(j, qjj)) continue;
                double v;
                if (qjj > 0) {
                    v = -c[j] / qjj;
                    v = std::max(lo[j], std::min(up[j], v));
                } else if (c[j] > 0) v = lo[j];
                else if (c[j] < 0) v = up[j];
                else v = is_finite(lo[j]) ? lo[j] : (is_finite(up[j]) ? up[j] : 0.0);
                if (!is_finite(v)) return Status::InfeasibleOrUnbounded;
                fix(j, v); st_.empty_cols++; changed = true;
            }
        }
        // ---- rows
        for (int i = 0; i < m; ++i) {
            if (!ralive[i]) continue;
            if (!is_finite(rlo[i]) && !is_finite(rup[i])) { drop_row(i); st_.redundant_rows++; changed = true; continue; }
            if (level < 1) continue;
            if (rcnt[i] == 0) {
                if (rlo[i] > ftol * (1 + std::fabs(rlo[i])) || rup[i] < -ftol * (1 + std::fabs(rup[i])))
                    return Status::Infeasible;
                drop_row(i); st_.redundant_rows++; changed = true; continue;
            }
            if (rcnt[i] == 1) {
                int j = -1; double a = 0;
                for (int p = AT.colptr[i]; p < AT.colptr[i + 1]; ++p)
                    if (calive[AT.rowidx[p]]) { j = AT.rowidx[p]; a = AT.val[p]; break; }
                double bl, bu;
                if (a > 0) { bl = rlo[i] / a; bu = rup[i] / a; }
                else { bl = rup[i] / a; bu = rlo[i] / a; }
                double olo = lo[j], oup = up[j];
                double nl = std::max(lo[j], bl), nu = std::min(up[j], bu);
                if (is_int(j)) { nl = std::ceil(nl - 1e-9); nu = std::floor(nu + 1e-9); }
                if (nl > nu + ftol * (1 + std::fabs(nl))) return Status::Infeasible;
                if (nl > nu) nu = nl;
                lo[j] = nl; up[j] = nu;
                if (nl != olo || nu != oup) st_.bounds_tightened++;
                ralive[i] = 0;
                ccnt[j]--;
                stack_.push_back({Op::SingletonRow, i, j, a, olo, oup, nl, nu});
                st_.rows_removed++; st_.singleton_rows++;
                changed = true;
                continue;
            }
            // activity bounds
            double minact = 0, maxact = 0;
            int mininf = 0, maxinf = 0;
            for (int p = AT.colptr[i]; p < AT.colptr[i + 1]; ++p) {
                int j = AT.rowidx[p];
                if (!calive[j]) continue;
                double a = AT.val[p];
                double l = lo[j], u = up[j];
                if (a > 0) {
                    if (is_finite(l)) minact += a * l; else mininf++;
                    if (is_finite(u)) maxact += a * u; else maxinf++;
                } else {
                    if (is_finite(u)) minact += a * u; else mininf++;
                    if (is_finite(l)) maxact += a * l; else maxinf++;
                }
            }
            const double scale = 1.0 + std::max(std::fabs(minact), std::fabs(maxact));
            if (mininf == 0 && is_finite(rup[i]) && minact > rup[i] + ftol * scale) return Status::Infeasible;
            if (maxinf == 0 && is_finite(rlo[i]) && maxact < rlo[i] - ftol * scale) return Status::Infeasible;
            bool lo_red = !is_finite(rlo[i]) || (mininf == 0 && minact >= rlo[i] - 1e-12 * scale);
            bool up_red = !is_finite(rup[i]) || (maxinf == 0 && maxact <= rup[i] + 1e-12 * scale);
            if (lo_red && up_red) { drop_row(i); st_.redundant_rows++; changed = true; continue; }
            bool force_min = mininf == 0 && is_finite(rup[i]) && minact >= rup[i] - 1e-12 * scale;
            bool force_max = maxinf == 0 && is_finite(rlo[i]) && maxact <= rlo[i] + 1e-12 * scale;
            if (force_min || force_max) {
                forcing_used_ = true;
                stack_.push_back({Op::ForcingRow, i, -1, 0, 0, 0, 0, 0});
                ralive[i] = 0;
                st_.rows_removed++; st_.forcing_rows++;
                for (int p = AT.colptr[i]; p < AT.colptr[i + 1]; ++p) {
                    int j = AT.rowidx[p];
                    if (!calive[j]) continue;
                    double a = AT.val[p];
                    double v = ((a > 0) == force_min) ? lo[j] : up[j];
                    fix(j, v);
                }
                changed = true;
                continue;
            }
        }
    }

    // ---- build reduced model
    col_map_.clear(); row_map_.clear();
    std::vector<int> cnew(n, -1), rnew(m, -1);
    for (int j = 0; j < n; ++j) if (calive[j]) { cnew[j] = (int)col_map_.size(); col_map_.push_back(j); }
    for (int i = 0; i < m; ++i) if (ralive[i]) { rnew[i] = (int)row_map_.size(); row_map_.push_back(i); }
    red_ = Model{};
    red_.name = om.name;
    red_.maximize = om.maximize;
    red_.n = (int)col_map_.size();
    red_.m = (int)row_map_.size();
    red_.obj_offset = offset;
    red_.A.m = red_.m; red_.A.n = red_.n;
    red_.A.colptr.assign(red_.n + 1, 0);
    for (int k = 0; k < red_.n; ++k) {
        int j = col_map_[k];
        for (int p = A.colptr[j]; p < A.colptr[j + 1]; ++p) {
            int i = rnew[A.rowidx[p]];
            if (i < 0) continue;
            red_.A.rowidx.push_back(i); red_.A.val.push_back(A.val[p]);
        }
        red_.A.colptr[k + 1] = (int)red_.A.rowidx.size();
        red_.c.push_back(c[j]);
        red_.col_lo.push_back(lo[j]);
        red_.col_up.push_back(up[j]);
        red_.is_int.push_back(is_int(j) ? 1 : 0);
        if (!om.col_names.empty()) red_.col_names.push_back(om.col_names[j]);
    }
    for (int k = 0; k < red_.m; ++k) {
        int i = row_map_[k];
        red_.row_lo.push_back(rlo[i]);
        red_.row_up.push_back(rup[i]);
        if (!om.row_names.empty()) red_.row_names.push_back(om.row_names[i]);
    }
    red_.Q.resize_empty(red_.n, red_.n);
    if (hasq) {
        for (int k = 0; k < red_.n; ++k) {
            int j = col_map_[k];
            for (int q = om.Q.colptr[j]; q < om.Q.colptr[j + 1]; ++q) {
                int i = cnew[om.Q.rowidx[q]];
                if (i < 0) continue;
                red_.Q.rowidx.push_back(i); red_.Q.val.push_back(om.Q.val[q]);
            }
            red_.Q.colptr[k + 1] = (int)red_.Q.rowidx.size();
        }
    }
    return Status::NotSolved;
}

void Presolver::postsolve(const std::vector<double>& xr, const std::vector<double>& yr,
                          std::vector<double>& x, std::vector<double>& y, bool& duals_valid) const {
    const Model& om = *orig_;
    x = fix_val_;
    x.resize(om.n);
    for (size_t k = 0; k < col_map_.size(); ++k) x[col_map_[k]] = xr[k];
    y.assign(om.m, 0.0);
    duals_valid = !forcing_used_ && yr.size() == row_map_.size();
    if (!duals_valid) return;
    for (size_t k = 0; k < row_map_.size(); ++k) y[row_map_[k]] = yr[k];

    std::vector<double> qx(om.n, 0.0);
    if (om.has_q()) om.Q.mul_add(x.data(), qx.data());
    for (auto it = stack_.rbegin(); it != stack_.rend(); ++it) {
        if (it->op != Op::SingletonRow) continue;
        const int j = it->col;
        double zj = om.c[j] + qx[j];
        for (int p = om.A.colptr[j]; p < om.A.colptr[j + 1]; ++p) zj -= om.A.val[p] * y[om.A.rowidx[p]];
        const double tol = 1e-7;
        const bool lo_from_row = it->new_lo > it->old_lo && std::fabs(x[j] - it->new_lo) <= tol * (1 + std::fabs(it->new_lo));
        const bool up_from_row = it->new_up < it->old_up && std::fabs(x[j] - it->new_up) <= tol * (1 + std::fabs(it->new_up));
        if ((lo_from_row && zj > 0) || (up_from_row && zj < 0)) y[it->row] = zj / it->a;
    }
}

}  // namespace nirnay
