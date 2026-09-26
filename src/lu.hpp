// NIRNAY — sparse LU factorisation of the simplex basis.
//
//  * Markowitz pivot search (column + row candidates, count-ordered buckets)
//    with threshold partial pivoting (|a_pq| >= u * max_i |a_iq|, u = 0.1).
//  * Singularity detection: columns without an acceptable pivot are reported so
//    the caller can repair the basis with logical (slack) columns.
//  * Basis changes are handled with product-form (eta) updates; the caller
//    refactorises periodically or when numerical checks fail.
#pragma once
#include <vector>

namespace nirnay {

class BasisLU {
public:
    // Factor the m x m basis B given in CSC form (column k = basis position k).
    // Returns the number of singular positions (then see singular_positions()).
    int factor(int m, const std::vector<int>& Bp, const std::vector<int>& Bi, const std::vector<double>& Bx);

    const std::vector<int>& singular_positions() const { return sing_pos_; }
    const std::vector<int>& unpivoted_rows() const { return unpiv_rows_; }

    // B x = b : in rhs indexed by row, out solution indexed by basis position.
    void ftran(std::vector<double>& x) const;
    // B' y = d : in d indexed by basis position, out y indexed by row.
    void btran(std::vector<double>& y) const;
    // Replace basis position r; alpha = B^{-1} a_q (indexed by position).
    void update(int r, const std::vector<double>& alpha);

    int num_updates() const { return (int)eta_r_.size(); }
    long long factor_nnz() const { return (long long)Lval_.size() + (long long)Uval_.size(); }

private:
    int m_ = 0;
    // pivot sequence
    std::vector<int> prow_, pcol_;
    std::vector<double> pval_;
    // L etas: for pivot k rows Lidx[Lstart[k]..Lstart[k+1]) with multipliers
    std::vector<int> Lstart_, Lidx_;
    std::vector<double> Lval_;
    // U rows: for pivot k columns Uidx[Ustart[k]..) with values
    std::vector<int> Ustart_, Uidx_;
    std::vector<double> Uval_;
    // PFI etas
    std::vector<int> eta_r_, eta_start_, eta_idx_;
    std::vector<double> eta_piv_, eta_val_;
    std::vector<int> sing_pos_, unpiv_rows_;
    mutable std::vector<double> work_;
};

}  // namespace nirnay
