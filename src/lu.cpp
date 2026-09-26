#include "lu.hpp"

#include <algorithm>
#include <climits>
#include <cmath>

namespace nirnay {

namespace {

// Doubly-linked count buckets.
struct Buckets {
    std::vector<int> head, next, prev, where;
    void init(int nitems, int maxcount) {
        head.assign(maxcount + 2, -1);
        next.assign(nitems, -1);
        prev.assign(nitems, -1);
        where.assign(nitems, -1);
    }
    void insert(int x, int c) {
        where[x] = c;
        next[x] = head[c]; prev[x] = -1;
        if (head[c] >= 0) prev[head[c]] = x;
        head[c] = x;
    }
    void remove(int x) {
        int c = where[x];
        if (c < 0) return;
        if (prev[x] >= 0) next[prev[x]] = next[x]; else head[c] = next[x];
        if (next[x] >= 0) prev[next[x]] = prev[x];
        where[x] = -1;
    }
    void move(int x, int c) { remove(x); insert(x, c); }
};

}  // namespace

int BasisLU::factor(int m, const std::vector<int>& Bp, const std::vector<int>& Bi, const std::vector<double>& Bx) {
    m_ = m;
    const double u = 0.1, abs_tol = 1e-11;
    std::vector<std::vector<int>> colR(m), rowC(m);
    std::vector<std::vector<double>> colV(m);
    for (int j = 0; j < m; ++j)
        for (int p = Bp[j]; p < Bp[j + 1]; ++p) {
            if (Bx[p] == 0.0) continue;
            colR[j].push_back(Bi[p]); colV[j].push_back(Bx[p]);
            rowC[Bi[p]].push_back(j);
        }
    std::vector<char> colAct(m, 1), rowAct(m, 1);
    Buckets cb, rb;
    cb.init(m, m); rb.init(m, m);
    for (int j = 0; j < m; ++j) cb.insert(j, (int)colR[j].size());
    for (int i = 0; i < m; ++i) rb.insert(i, (int)rowC[i].size());

    prow_.clear(); pcol_.clear(); pval_.clear();
    Lstart_.assign(1, 0); Lidx_.clear(); Lval_.clear();
    Ustart_.assign(1, 0); Uidx_.clear(); Uval_.clear();
    eta_r_.clear(); eta_start_.assign(1, 0); eta_idx_.clear(); eta_piv_.clear(); eta_val_.clear();
    sing_pos_.clear(); unpiv_rows_.clear();

    std::vector<int> pos(m, -1);
    std::vector<int> Lrows; std::vector<double> Lvals;
    std::vector<int> Ucols; std::vector<double> Uvals;

    auto col_max = [&](int j) {
        double mx = 0;
        for (double v : colV[j]) mx = std::max(mx, std::fabs(v));
        return mx;
    };
    auto find_in_col = [&](int j, int i) {
        const auto& r = colR[j];
        for (size_t k = 0; k < r.size(); ++k) if (r[k] == i) return (int)k;
        return -1;
    };

    for (int step = 0; step < m; ++step) {
        int bp = -1, bq = -1;
        long long bestmk = LLONG_MAX;
        double bestabs = 0;
        int examined = 0;
        for (int c = 1; c <= m; ++c) {
            for (int j = cb.head[c]; j >= 0; j = cb.next[j]) {
                double cmax = col_max(j);
                if (cmax < abs_tol) continue;
                const auto& r = colR[j];
                for (size_t k = 0; k < r.size(); ++k) {
                    double a = std::fabs(colV[j][k]);
                    if (a < u * cmax) continue;
                    long long mk = (long long)(rowC[r[k]].size() - 1) * (c - 1);
                    if (mk < bestmk || (mk == bestmk && a > bestabs)) { bestmk = mk; bestabs = a; bp = r[k]; bq = j; }
                }
                if (++examined >= 4 && bp >= 0) break;
            }
            if (bp >= 0 && (bestmk <= (long long)(c - 1) * (c - 1) || examined >= 4)) break;
            for (int i = rb.head[c]; i >= 0; i = rb.next[i]) {
                for (int j : rowC[i]) {
                    int k = find_in_col(j, i);
                    if (k < 0) continue;
                    double a = std::fabs(colV[j][k]);
                    double cmax = col_max(j);
                    if (cmax < abs_tol || a < u * cmax) continue;
                    long long mk = (long long)(c - 1) * (long long)(colR[j].size() - 1);
                    if (mk < bestmk || (mk == bestmk && a > bestabs)) { bestmk = mk; bestabs = a; bp = i; bq = j; }
                }
                if (++examined >= 4 && bp >= 0) break;
            }
            if (bp >= 0 && (bestmk <= (long long)c * c || examined >= 4)) break;
        }
        if (bp < 0) break;  // remaining submatrix numerically singular

        const int p = bp, q = bq;
        int kq = find_in_col(q, p);
        const double piv = colV[q][kq];

        // U row: row p entries in other active columns
        Ucols.clear(); Uvals.clear();
        for (int j : rowC[p]) {
            if (j == q) continue;
            int k = find_in_col(j, p);
            Ucols.push_back(j); Uvals.push_back(colV[j][k]);
            colR[j][k] = colR[j].back(); colR[j].pop_back();
            colV[j][k] = colV[j].back(); colV[j].pop_back();
        }
        // L column: column q entries in other active rows
        Lrows.clear(); Lvals.clear();
        for (size_t k = 0; k < colR[q].size(); ++k) {
            int i = colR[q][k];
            if (i == p) continue;
            Lrows.push_back(i); Lvals.push_back(colV[q][k] / piv);
            auto& rc = rowC[i];
            for (size_t t = 0; t < rc.size(); ++t)
                if (rc[t] == q) { rc[t] = rc.back(); rc.pop_back(); break; }
        }
        cb.remove(q); rb.remove(p);
        colAct[q] = 0; rowAct[p] = 0;
        std::vector<int>().swap(colR[q]); std::vector<double>().swap(colV[q]);
        std::vector<int>().swap(rowC[p]);

        // Schur complement update
        for (size_t t = 0; t < Ucols.size(); ++t) {
            const int j = Ucols[t];
            const double upj = Uvals[t];
            auto& r = colR[j];
            auto& v = colV[j];
            for (size_t k = 0; k < r.size(); ++k) pos[r[k]] = (int)k;
            for (size_t s = 0; s < Lrows.size(); ++s) {
                const int i = Lrows[s];
                const double delta = -Lvals[s] * upj;
                if (pos[i] >= 0) v[pos[i]] += delta;
                else {
                    pos[i] = (int)r.size();
                    r.push_back(i); v.push_back(delta);
                    rowC[i].push_back(j);
                }
            }
            for (size_t k = 0; k < r.size(); ++k) pos[r[k]] = -1;
            cb.move(j, (int)r.size());
        }
        for (int i : Lrows) rb.move(i, (int)rowC[i].size());

        prow_.push_back(p); pcol_.push_back(q); pval_.push_back(piv);
        Lidx_.insert(Lidx_.end(), Lrows.begin(), Lrows.end());
        Lval_.insert(Lval_.end(), Lvals.begin(), Lvals.end());
        Lstart_.push_back((int)Lidx_.size());
        Uidx_.insert(Uidx_.end(), Ucols.begin(), Ucols.end());
        Uval_.insert(Uval_.end(), Uvals.begin(), Uvals.end());
        Ustart_.push_back((int)Uidx_.size());
    }
    for (int j = 0; j < m; ++j) if (colAct[j]) sing_pos_.push_back(j);
    for (int i = 0; i < m; ++i) if (rowAct[i]) unpiv_rows_.push_back(i);
    work_.assign(m, 0.0);
    return (int)sing_pos_.size();
}

void BasisLU::ftran(std::vector<double>& x) const {
    const int K = (int)prow_.size();
    for (int k = 0; k < K; ++k) {
        double xp = x[prow_[k]];
        if (xp == 0.0) continue;
        for (int t = Lstart_[k]; t < Lstart_[k + 1]; ++t) x[Lidx_[t]] -= Lval_[t] * xp;
    }
    std::vector<double>& out = work_;
    std::fill(out.begin(), out.end(), 0.0);
    for (int k = K - 1; k >= 0; --k) {
        double s = x[prow_[k]];
        for (int t = Ustart_[k]; t < Ustart_[k + 1]; ++t) s -= Uval_[t] * out[Uidx_[t]];
        out[pcol_[k]] = s / pval_[k];
    }
    const int E = (int)eta_r_.size();
    for (int e = 0; e < E; ++e) {
        const int r = eta_r_[e];
        double xr = out[r] / eta_piv_[e];
        out[r] = xr;
        if (xr == 0.0) continue;
        for (int t = eta_start_[e]; t < eta_start_[e + 1]; ++t) out[eta_idx_[t]] -= eta_val_[t] * xr;
    }
    x.swap(out);
    out.assign(m_, 0.0);
}

void BasisLU::btran(std::vector<double>& d) const {
    for (int e = (int)eta_r_.size() - 1; e >= 0; --e) {
        const int r = eta_r_[e];
        double s = d[r];
        for (int t = eta_start_[e]; t < eta_start_[e + 1]; ++t) s -= eta_val_[t] * d[eta_idx_[t]];
        d[r] = s / eta_piv_[e];
    }
    const int K = (int)prow_.size();
    std::vector<double>& y = work_;
    std::fill(y.begin(), y.end(), 0.0);
    for (int k = 0; k < K; ++k) {
        double t = d[pcol_[k]] / pval_[k];
        y[prow_[k]] = t;
        if (t == 0.0) continue;
        for (int s = Ustart_[k]; s < Ustart_[k + 1]; ++s) d[Uidx_[s]] -= Uval_[s] * t;
    }
    for (int k = K - 1; k >= 0; --k) {
        double s = y[prow_[k]];
        for (int t = Lstart_[k]; t < Lstart_[k + 1]; ++t) s -= Lval_[t] * y[Lidx_[t]];
        y[prow_[k]] = s;
    }
    d.swap(y);
    y.assign(m_, 0.0);
}

void BasisLU::update(int r, const std::vector<double>& alpha) {
    eta_r_.push_back(r);
    eta_piv_.push_back(alpha[r]);
    for (int i = 0; i < m_; ++i)
        if (i != r && std::fabs(alpha[i]) > 1e-14) { eta_idx_.push_back(i); eta_val_.push_back(alpha[i]); }
    eta_start_.push_back((int)eta_idx_.size());
}

}  // namespace nirnay
