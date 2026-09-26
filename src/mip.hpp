// NIRNAY — LP-based branch-and-cut for MILP.
//
//  * Node LPs: NIRNAY dual simplex, warm-started from the parent basis (bound
//    changes keep the basis dual feasible), objective cutoff at the incumbent.
//  * Root cutting planes: Gomory mixed-integer (GMI) cuts from optimal tableau rows,
//    with coefficient-range and efficacy safeguards, several rounds.
//  * Branching: reliability pseudocost branching (strong branching on unreliable
//    candidates, product score), infeasible-child bound tightening.
//  * Node selection: best-bound with depth-first plunging.
//  * Primal heuristics: lock-based simple rounding, fractional diving,
//    fix-integers-and-solve-LP.
//  * Integral-objective detection for stronger pruning.
#pragma once
#include "common.hpp"
#include "model.hpp"

namespace nirnay {

struct MipOptions {
    double rel_gap = 1e-4;
    double abs_gap = 1e-6;
    long long node_limit = 10000000;
    double time_limit = 1e30;
    bool cuts = true;
    bool heuristics = true;
    int sb_candidates = 8;      // strong-branching candidates per node
    int sb_reliability = 4;     // pseudocost reliability threshold
    int sb_iters = 60;          // dual simplex iteration cap per strong-branching child
    int cut_rounds = 15;
};

struct MipResult {
    Status status = Status::NotSolved;
    std::vector<double> x;      // in the (presolved, unscaled) model space
    double obj = 0, bound = 0;  // internal minimisation sense, incl. offset
    long long nodes = 0, lp_iterations = 0;
    int cuts = 0;
    double root_bound = 0, root_bound_after_cuts = 0;
    bool trouble = false;
};

MipResult solve_mip(const Model& mdl, const MipOptions& opt);

}  // namespace nirnay
