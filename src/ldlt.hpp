// NIRNAY — sparse symmetric LDL' factorisation for quasi-definite KKT systems.
//
//  * Fill-reducing ordering: approximate minimum degree on a quotient graph
//    (element absorption + approximate external degree), implemented here.
//  * Symbolic analysis: elimination tree + column counts (computed once).
//  * Numeric: up-looking LDL' with *signed dynamic regularisation*: a pivot
//    whose magnitude collapses is replaced by  sign_k * reg  where sign_k is
//    the expected inertia (-1 for primal block, +1 for dual block). With static
//    primal/dual regularisation the KKT matrix is quasi-definite, so any
//    symmetric ordering admits a stable LDL' without pivoting (Vanderbei 1995).
#pragma once
#include <vector>

namespace nirnay {

// Symmetric pattern given as full CSC (both triangles, diagonal included).
std::vector<int> amd_order(int n, const std::vector<int>& colptr, const std::vector<int>& rowidx);

class LdlFactor {
public:
    // Analyse pattern of symmetric matrix K (full CSC storage incl. diagonal).
    // `sign` gives expected pivot sign per original index (+1 / -1).
    void analyse(int n, const std::vector<int>& colptr, const std::vector<int>& rowidx,
                 const std::vector<signed char>& sign);
    // Numeric factorisation; `val` follows the full CSC layout passed to analyse().
    // Returns number of pivots that required dynamic regularisation.
    int factor(const std::vector<double>& val, double dyn_reg_tol, double dyn_reg_val);
    // Solve K x = b in place (b indexed by original indices).
    void solve(std::vector<double>& b) const;

    long long nnz_L() const { return Lp_.empty() ? 0 : Lp_.back(); }
    int n() const { return n_; }

private:
    int n_ = 0;
    std::vector<int> perm_, pinv_;
    // permuted upper-triangular CSC pattern + map from input nz -> permuted slot
    std::vector<int> Up_, Ui_;
    std::vector<int> map_;   // for each input nz: slot in Ux (or -1 if lower part)
    std::vector<double> Ux_;
    std::vector<int> parent_, Lnz_, Lp_, Li_;
    std::vector<double> Lx_, D_;
    std::vector<signed char> psign_;
    // workspace
    mutable std::vector<double> Y_, work_;
    std::vector<int> pattern_, flag_;
};

}  // namespace nirnay
