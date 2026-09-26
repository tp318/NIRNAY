#include "model.hpp"

#include <algorithm>
#include <cmath>

namespace nirnay {

void CscMatrix::mul_add(const double* x, double* y, double alpha) const {
    for (int j = 0; j < n; ++j) {
        double xj = x[j];
        if (xj == 0.0) continue;
        xj *= alpha;
        for (int p = colptr[j]; p < colptr[j + 1]; ++p) y[rowidx[p]] += val[p] * xj;
    }
}

void CscMatrix::mul_t_add(const double* x, double* y, double alpha) const {
    for (int j = 0; j < n; ++j) {
        double s = 0.0;
        for (int p = colptr[j]; p < colptr[j + 1]; ++p) s += val[p] * x[rowidx[p]];
        y[j] += alpha * s;
    }
}

CscMatrix CscMatrix::transpose() const {
    CscMatrix T;
    T.m = n; T.n = m;
    T.colptr.assign(m + 1, 0);
    int nz = nnz();
    for (int p = 0; p < nz; ++p) T.colptr[rowidx[p] + 1]++;
    for (int i = 0; i < m; ++i) T.colptr[i + 1] += T.colptr[i];
    T.rowidx.resize(nz); T.val.resize(nz);
    std::vector<int> next(T.colptr.begin(), T.colptr.end() - 1);
    for (int j = 0; j < n; ++j)
        for (int p = colptr[j]; p < colptr[j + 1]; ++p) {
            int q = next[rowidx[p]]++;
            T.rowidx[q] = j; T.val[q] = val[p];
        }
    return T;
}

double CscMatrix::max_abs() const {
    double mx = 0;
    for (double v : val) mx = std::max(mx, std::fabs(v));
    return mx;
}

CscMatrix csc_from_triplets(int m, int n, const std::vector<int>& ri,
                            const std::vector<int>& ci, const std::vector<double>& v) {
    CscMatrix M;
    M.m = m; M.n = n;
    M.colptr.assign(n + 1, 0);
    size_t nz = v.size();
    for (size_t k = 0; k < nz; ++k) M.colptr[ci[k] + 1]++;
    for (int j = 0; j < n; ++j) M.colptr[j + 1] += M.colptr[j];
    std::vector<int> r(nz); std::vector<double> x(nz);
    std::vector<int> next(M.colptr.begin(), M.colptr.end() - 1);
    for (size_t k = 0; k < nz; ++k) { int q = next[ci[k]]++; r[q] = ri[k]; x[q] = v[k]; }
    // sort each column by row, merge duplicates, drop zeros
    std::vector<int> outptr(n + 1, 0);
    std::vector<std::pair<int, double>> buf;
    M.rowidx.clear(); M.val.clear();
    M.rowidx.reserve(nz); M.val.reserve(nz);
    for (int j = 0; j < n; ++j) {
        buf.clear();
        for (int p = M.colptr[j]; p < M.colptr[j + 1]; ++p) buf.emplace_back(r[p], x[p]);
        std::sort(buf.begin(), buf.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        for (size_t k = 0; k < buf.size();) {
            int row = buf[k].first; double s = 0;
            while (k < buf.size() && buf[k].first == row) s += buf[k++].second;
            if (s != 0.0) { M.rowidx.push_back(row); M.val.push_back(s); }
        }
        outptr[j + 1] = (int)M.rowidx.size();
    }
    M.colptr = outptr;
    return M;
}

int Model::num_int() const {
    int k = 0;
    for (char b : is_int) k += b ? 1 : 0;
    return k;
}

double Model::objective(const std::vector<double>& x) const {
    double f = obj_offset;
    for (int j = 0; j < n; ++j) f += c[j] * x[j];
    if (has_q()) {
        std::vector<double> qx(n, 0.0);
        Q.mul_add(x.data(), qx.data());
        for (int j = 0; j < n; ++j) f += 0.5 * x[j] * qx[j];
    }
    return f;
}

std::vector<double> Model::activities(const std::vector<double>& x) const {
    std::vector<double> ax(m, 0.0);
    A.mul_add(x.data(), ax.data());
    return ax;
}

Violation check_primal(const Model& mdl, const std::vector<double>& x) {
    Violation v;
    auto ax = mdl.activities(x);
    for (int i = 0; i < mdl.m; ++i) {
        double viol = std::max(mdl.row_lo[i] - ax[i], ax[i] - mdl.row_up[i]);
        v.max_row = std::max(v.max_row, viol);
    }
    for (int j = 0; j < mdl.n; ++j) {
        double viol = std::max(mdl.col_lo[j] - x[j], x[j] - mdl.col_up[j]);
        v.max_bound = std::max(v.max_bound, viol);
        if (!mdl.is_int.empty() && mdl.is_int[j])
            v.max_int = std::max(v.max_int, std::fabs(x[j] - std::round(x[j])));
    }
    return v;
}

}  // namespace nirnay
