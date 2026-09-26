// NIRNAY — presolve / postsolve.
//
// Reductions (iterated to a fixed point):
//   * integer bound rounding, fixed columns (with quadratic cross terms)
//   * empty columns (fixed at the cost-optimal bound; detects unboundedness)
//   * empty / free rows, singleton rows -> column bounds
//   * activity-based redundant-row removal and infeasibility detection
//   * forcing rows (all columns fixed at the activity-attaining bounds)
// Postsolve restores the primal solution exactly and row duals for every
// reduction except forcing rows (flagged via duals_valid).
#pragma once
#include "common.hpp"
#include "model.hpp"

namespace nirnay {

struct PresolveStats {
    int rows_removed = 0, cols_removed = 0, bounds_tightened = 0;
    int singleton_rows = 0, forcing_rows = 0, redundant_rows = 0, fixed_cols = 0, empty_cols = 0;
};

class Presolver {
public:
    // level 0: mandatory only (fixed columns, free rows); level 1: full.
    Status run(const Model& original, int level);
    const Model& reduced() const { return red_; }
    const PresolveStats& stats() const { return st_; }
    // Map a reduced-space solution back to the original model.
    void postsolve(const std::vector<double>& xr, const std::vector<double>& yr,
                   std::vector<double>& x, std::vector<double>& y, bool& duals_valid) const;

private:
    enum class Op { FixCol, DropRow, SingletonRow, ForcingRow };
    struct Rec { Op op; int row; int col; double a; double old_lo, old_up, new_lo, new_up; };

    const Model* orig_ = nullptr;
    Model red_;
    PresolveStats st_;
    std::vector<int> col_map_, row_map_;  // reduced -> original
    std::vector<double> fix_val_;
    std::vector<Rec> stack_;
    bool forcing_used_ = false;
};

}  // namespace nirnay
