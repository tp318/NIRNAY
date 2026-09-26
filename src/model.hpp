// NIRNAY — core problem representation.
//
//   minimize    c'x + 0.5 x'Qx + offset
//   subject to  row_lo <= A x <= row_up
//               col_lo <=  x  <= col_up
//               x_j integer for j with is_int[j]
//
// Maximisation problems are negated on input; `maximize` records the
// original sense so reported objectives can be flipped back.
#pragma once
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace nirnay {

constexpr double kInf = std::numeric_limits<double>::infinity();

inline bool is_finite(double v) { return v > -kInf && v < kInf; }

// Compressed sparse column matrix.
struct CscMatrix {
    int m = 0, n = 0;
    std::vector<int> colptr;   // size n+1
    std::vector<int> rowidx;   // size nnz
    std::vector<double> val;   // size nnz

    int nnz() const { return colptr.empty() ? 0 : colptr[n]; }
    void resize_empty(int rows, int cols) {
        m = rows; n = cols;
        colptr.assign(cols + 1, 0);
        rowidx.clear(); val.clear();
    }
    // y += alpha * A x
    void mul_add(const double* x, double* y, double alpha = 1.0) const;
    // y += alpha * A' x
    void mul_t_add(const double* x, double* y, double alpha = 1.0) const;
    CscMatrix transpose() const;
    double max_abs() const;
};

// Build a CSC matrix from triplets; duplicates are summed, explicit zeros dropped.
CscMatrix csc_from_triplets(int m, int n, const std::vector<int>& ri,
                            const std::vector<int>& ci, const std::vector<double>& v);

struct Model {
    std::string name;
    int m = 0, n = 0;
    CscMatrix A;                      // m x n
    std::vector<double> c;            // n
    double obj_offset = 0.0;
    CscMatrix Q;                      // n x n, full symmetric storage (may be empty)
    std::vector<double> col_lo, col_up;
    std::vector<double> row_lo, row_up;
    std::vector<char> is_int;
    std::vector<std::string> col_names, row_names;
    bool maximize = false;

    bool has_q() const { return Q.nnz() > 0; }
    int num_int() const;
    // c'x + 0.5 x'Qx + offset  (in the internal minimisation sense)
    double objective(const std::vector<double>& x) const;
    // Row activities A x
    std::vector<double> activities(const std::vector<double>& x) const;
};

// Quality of a primal point w.r.t. the original model.
struct Violation {
    double max_row = 0, max_bound = 0, max_int = 0;
};
Violation check_primal(const Model& mdl, const std::vector<double>& x);

}  // namespace nirnay
