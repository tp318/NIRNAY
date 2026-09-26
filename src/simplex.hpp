// NIRNAY — bounded dual simplex method.
//
// Computational form  [A  -I] [x; r] = 0,  lo <= (x, r) <= up  (r = row activities).
//
//  * Pricing: dual steepest edge (exact reference weights ||e_r' B^{-1}||^2 updated per pivot)
//  * Ratio test: Harris two-pass with bound flipping (long-step / BFRT)
//  * Degeneracy: randomised cost perturbation + cost shifting, removed at the end
//  * Dual phase 1: auxiliary box-bounded subproblem (Fourer 1994 / Koberstein 2005)
//  * Numerical safeguards: row/column pivot consistency check, periodic and
//    on-demand refactorisation, singular-basis repair with logical columns
//  * Warm start: basis get/set, bound changes and appended rows (for branch & cut)
#pragma once
#include <cstdint>
#include <vector>

#include "common.hpp"
#include "lu.hpp"
#include "model.hpp"

namespace nirnay {

struct SimplexOptions {
    double tol_primal = 1e-7;
    double tol_dual = 1e-7;
    long long iter_limit = 50000000;
    double time_limit = 1e30;
    bool perturb = true;
    double cutoff = kInf;   // stop when dual objective exceeds (for branch & bound)
    int refactor_freq = 100;
};

enum class SimplexResult { Optimal, Infeasible, DualInfeasible, Cutoff, IterLimit, TimeLimit, Numerical };

class DualSimplex {
public:
    enum : signed char { BASIC = 0, AT_LO = 1, AT_UP = 2, AT_ZERO = 3, FIXED = 4 };
    struct Basis {
        std::vector<int> head;
        std::vector<signed char> status;
    };

    void load(const Model& mdl);  // model must be an LP (Q ignored); assumed scaled by caller
    SimplexResult solve(const SimplexOptions& opt);

    int rows() const { return m_; }
    int cols() const { return n_; }
    int vars() const { return N_; }
    void set_col_bounds(int j, double lo, double up);
    double lo(int j) const { return lo_[j]; }
    double up(int j) const { return up_[j]; }
    void add_rows(const std::vector<std::vector<std::pair<int, double>>>& rows,
                  const std::vector<double>& rlo, const std::vector<double>& rup);
    // Delete rows whose logical variable is basic (basis stays valid).
    void remove_rows(const std::vector<int>& rows);

    Basis basis() const { return {head_, st_}; }
    void set_basis(const Basis& b);

    double objective() const;                 // c'x + offset (unperturbed costs)
    const std::vector<double>& values() const { return x_; }  // all N variables
    std::vector<double> primal() const { return std::vector<double>(x_.begin(), x_.begin() + n_); }
    std::vector<double> row_duals() const { return y_; }
    std::vector<double> reduced_costs() const { return std::vector<double>(d_.begin(), d_.begin() + n_); }
    long long iterations() const { return iters_; }
    int status_of(int j) const { return st_[j]; }
    int position_of(int j) const { return pos_[j]; }
    const std::vector<int>& head() const { return head_; }
    const CscMatrix& matrix() const { return A_; }
    // Tableau row of basic position r:  x_B[r] + sum_{j nonbasic} alpha_j x_j = 0.
    void tableau_row(int r, std::vector<double>& alpha);

private:
    int n_ = 0, m_ = 0, N_ = 0;
    CscMatrix A_, AT_;
    std::vector<double> c_, cw_, lo_, up_, x_, y_, d_, w_;
    std::vector<int> head_, pos_;
    std::vector<signed char> st_;
    double offset_ = 0.0;
    BasisLU lu_;
    bool factored_ = false;
    long long iters_ = 0;
    uint64_t rng_ = 0x9E3779B97F4A7C15ull;
    Timer timer_;

    // pivot-row workspace
    std::vector<double> arow_;
    std::vector<int> touched_;
    std::vector<char> is_touched_;

    void col_scatter_add(int j, double scale, std::vector<double>& v) const;
    double col_dot(int j, const std::vector<double>& v) const;
    bool refactor();
    void compute_primal();
    void compute_dual();
    void normalize_status(int j);
    void set_nonbasic_value(int j);
    int flip_and_count_dual_infeasible(double tol, bool& flipped);
    SimplexResult iterate(const SimplexOptions& opt, bool allow_cutoff);
    bool phase1(const SimplexOptions& opt);
    double rand01();
};

}  // namespace nirnay
