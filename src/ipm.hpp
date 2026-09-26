// NIRNAY — primal-dual interior point method for LP and convex QP.
//
//   min c'x + 0.5 x'Qx   s.t.  Ax - w = 0 (inequality rows), Ax = b (equality rows),
//                              l <= x <= u,  row_lo <= w <= row_up
//
// Mehrotra predictor-corrector with Gondzio centrality correctors, solved via
// the regularised quasi-definite augmented system
//
//   [ -(Q + Θ^{-1} + ρI)   Â' ] [dx]   [r1]
//   [        Â             δI ] [dy] = [r2]
//
// factorised with NIRNAY's own sparse LDL' (ldlt.hpp) and polished by
// iterative refinement against the unregularised system.
#pragma once
#include "common.hpp"
#include "model.hpp"

namespace nirnay {

struct IpmOptions {
    double tol = 1e-8;
    int max_iter = 300;
    double time_limit = 1e30;
    int max_correctors = 2;
};

// Requires: no fixed columns (lo == up) and no free rows. Model assumed scaled.
LpSolution solve_ipm(const Model& mdl, const IpmOptions& opt);

}  // namespace nirnay
