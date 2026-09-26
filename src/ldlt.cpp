#include "ldlt.hpp"

#include <algorithm>
#include <cmath>

namespace nirnay {

// ---------------------------------------------------------------------------
// Approximate minimum degree ordering on the quotient graph.
//
// Variables are uneliminated nodes; elements are eliminated pivots whose
// adjacency forms a clique. For each variable we keep adjacent variables
// (adjV) and adjacent elements (adjE). When pivot p is eliminated the new
// element L_p = adjV(p) ∪ (∪_{e∈adjE(p)} L_e) \ {p} absorbs every element
// adjacent to p. Degrees are then updated with the AMD-style upper bound
//     d_i ≈ |adjV(i)| + |L_p \ {i}| + Σ_{e∈adjE(i), e≠p} |L_e \ L_p|
// capped by d_i(old) + |L_p| - 1 and the size of the remaining graph.
// ---------------------------------------------------------------------------
std::vector<int> amd_order(int n, const std::vector<int>& colptr, const std::vector<int>& rowidx) {
    std::vector<std::vector<int>> adjV(n), adjE(n), Le(n);
    for (int j = 0; j < n; ++j) {
        auto& a = adjV[j];
        for (int p = colptr[j]; p < colptr[j + 1]; ++p)
            if (rowidx[p] != j) a.push_back(rowidx[p]);
        std::sort(a.begin(), a.end());
        a.erase(std::unique(a.begin(), a.end()), a.end());
    }
    std::vector<signed char> status(n, 0);  // 0 variable, 1 element, 2 absorbed
    std::vector<int> deg(n), head(n + 1, -1), next(n, -1), prev(n, -1);
    auto bucket_insert = [&](int i) {
        int d = deg[i];
        next[i] = head[d]; prev[i] = -1;
        if (head[d] >= 0) prev[head[d]] = i;
        head[d] = i;
    };
    auto bucket_remove = [&](int i) {
        int d = deg[i];
        if (prev[i] >= 0) next[prev[i]] = next[i]; else head[d] = next[i];
        if (next[i] >= 0) prev[next[i]] = prev[i];
    };
    for (int i = 0; i < n; ++i) { deg[i] = (int)adjV[i].size(); bucket_insert(i); }

    std::vector<int> mark(n, 0), wflag(n, 0), wval(n, 0);
    int stamp = 0, estamp = 0, mindeg = 0;
    std::vector<int> order; order.reserve(n);
    std::vector<int> Lp;

    for (int k = 0; k < n; ++k) {
        while (mindeg < n && head[mindeg] < 0) ++mindeg;
        int p = head[mindeg];
        bucket_remove(p);
        order.push_back(p);
        status[p] = 1;

        ++stamp;
        Lp.clear();
        mark[p] = stamp;
        for (int v : adjV[p])
            if (status[v] == 0 && mark[v] != stamp) { mark[v] = stamp; Lp.push_back(v); }
        for (int e : adjE[p]) {
            if (status[e] != 1) continue;
            for (int v : Le[e])
                if (status[v] == 0 && mark[v] != stamp) { mark[v] = stamp; Lp.push_back(v); }
            status[e] = 2;
            std::vector<int>().swap(Le[e]);
        }
        std::vector<int>().swap(adjV[p]);
        std::vector<int>().swap(adjE[p]);
        Le[p] = Lp;

        // |L_e \ L_p| for every element adjacent to a member of L_p
        ++estamp;
        for (int i : Lp) {
            for (int e : adjE[i]) {
                if (status[e] != 1) continue;
                if (wflag[e] != estamp) {
                    wflag[e] = estamp;
                    auto& le = Le[e];
                    size_t w = 0;
                    for (size_t t = 0; t < le.size(); ++t)
                        if (status[le[t]] == 0) le[w++] = le[t];
                    le.resize(w);
                    wval[e] = (int)w;
                }
                wval[e] -= 1;
            }
        }
        const int lp_sz = (int)Lp.size();
        const int remaining = n - k - 1;
        for (int i : Lp) {
            auto& ae = adjE[i];
            size_t w = 0;
            long long ext = 0;
            for (size_t t = 0; t < ae.size(); ++t) {
                int e = ae[t];
                if (status[e] == 1) { ae[w++] = e; ext += std::max(wval[e], 0); }
            }
            ae.resize(w);
            ae.push_back(p);
            auto& av = adjV[i];
            w = 0;
            for (size_t t = 0; t < av.size(); ++t) {
                int v = av[t];
                if (status[v] == 0 && mark[v] != stamp && v != i) av[w++] = v;
            }
            av.resize(w);
            long long d = (long long)av.size() + (lp_sz - 1) + ext;
            d = std::min<long long>(d, (long long)deg[i] + lp_sz - 1);
            d = std::min<long long>(d, remaining);
            if (d < 0) d = 0;
            bucket_remove(i);
            deg[i] = (int)d;
            bucket_insert(i);
            if (d < mindeg) mindeg = (int)d;
        }
    }
    return order;
}

// ---------------------------------------------------------------------------
void LdlFactor::analyse(int n, const std::vector<int>& colptr, const std::vector<int>& rowidx,
                        const std::vector<signed char>& sign) {
    n_ = n;
    perm_ = amd_order(n, colptr, rowidx);
    pinv_.assign(n, 0);
    for (int k = 0; k < n; ++k) pinv_[perm_[k]] = k;
    psign_.resize(n);
    for (int k = 0; k < n; ++k) psign_[k] = sign[perm_[k]];

    // permuted upper triangle
    const int nz = colptr[n];
    Up_.assign(n + 1, 0);
    for (int j = 0; j < n; ++j)
        for (int p = colptr[j]; p < colptr[j + 1]; ++p) {
            int pi = pinv_[rowidx[p]], pj = pinv_[j];
            if (pi <= pj) Up_[pj + 1]++;
        }
    for (int j = 0; j < n; ++j) Up_[j + 1] += Up_[j];
    Ui_.resize(Up_[n]);
    Ux_.assign(Up_[n], 0.0);
    map_.assign(nz, -1);
    std::vector<int> nxt(Up_.begin(), Up_.end() - 1);
    for (int j = 0; j < n; ++j)
        for (int p = colptr[j]; p < colptr[j + 1]; ++p) {
            int pi = pinv_[rowidx[p]], pj = pinv_[j];
            if (pi <= pj) { int q = nxt[pj]++; Ui_[q] = pi; map_[p] = q; }
        }

    // elimination tree and column counts
    parent_.assign(n, -1);
    Lnz_.assign(n, 0);
    flag_.assign(n, -1);
    for (int k = 0; k < n; ++k) {
        flag_[k] = k;
        for (int p = Up_[k]; p < Up_[k + 1]; ++p) {
            int i = Ui_[p];
            if (i >= k) continue;
            for (; flag_[i] != k; i = parent_[i]) {
                if (parent_[i] == -1) parent_[i] = k;
                Lnz_[i]++;
                flag_[i] = k;
            }
        }
    }
    Lp_.assign(n + 1, 0);
    for (int k = 0; k < n; ++k) Lp_[k + 1] = Lp_[k] + Lnz_[k];
    Li_.resize(Lp_[n]);
    Lx_.resize(Lp_[n]);
    D_.assign(n, 0.0);
    Y_.assign(n, 0.0);
    work_.assign(n, 0.0);
    pattern_.assign(n, 0);
}

int LdlFactor::factor(const std::vector<double>& val, double tol, double reg) {
    const int n = n_;
    std::fill(Ux_.begin(), Ux_.end(), 0.0);
    for (size_t p = 0; p < map_.size(); ++p)
        if (map_[p] >= 0) Ux_[map_[p]] += val[p];

    int nreg = 0;
    for (int k = 0; k < n; ++k) {
        Y_[k] = 0.0;
        int top = n;
        flag_[k] = k;
        Lnz_[k] = 0;
        for (int p = Up_[k]; p < Up_[k + 1]; ++p) {
            int i = Ui_[p];
            Y_[i] += Ux_[p];
            int len = 0;
            for (; flag_[i] != k; i = parent_[i]) { pattern_[len++] = i; flag_[i] = k; }
            while (len > 0) pattern_[--top] = pattern_[--len];
        }
        double dk = Y_[k];
        Y_[k] = 0.0;
        for (; top < n; ++top) {
            int i = pattern_[top];
            double yi = Y_[i];
            Y_[i] = 0.0;
            int p2 = Lp_[i] + Lnz_[i];
            for (int p = Lp_[i]; p < p2; ++p) Y_[Li_[p]] -= Lx_[p] * yi;
            double lki = yi / D_[i];
            dk -= lki * yi;
            Li_[p2] = k;
            Lx_[p2] = lki;
            Lnz_[i]++;
        }
        if (psign_[k] * dk < tol) { dk = psign_[k] * reg; ++nreg; }
        D_[k] = dk;
    }
    return nreg;
}

void LdlFactor::solve(std::vector<double>& b) const {
    const int n = n_;
    std::vector<double>& x = work_;
    for (int k = 0; k < n; ++k) x[k] = b[perm_[k]];
    for (int j = 0; j < n; ++j) {
        double xj = x[j];
        if (xj == 0.0) continue;
        for (int p = Lp_[j]; p < Lp_[j + 1]; ++p) x[Li_[p]] -= Lx_[p] * xj;
    }
    for (int j = 0; j < n; ++j) x[j] /= D_[j];
    for (int j = n - 1; j >= 0; --j) {
        double s = x[j];
        for (int p = Lp_[j]; p < Lp_[j + 1]; ++p) s -= Lx_[p] * x[Li_[p]];
        x[j] = s;
    }
    for (int k = 0; k < n; ++k) b[perm_[k]] = x[k];
}

}  // namespace nirnay
