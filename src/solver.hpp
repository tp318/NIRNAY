// NIRNAY — public solver API.
#pragma once
#include <string>
#include <vector>

#include "common.hpp"
#include "model.hpp"
#include "presolve.hpp"

namespace nirnay {

struct SolverOptions {
    std::string method = "auto";   // auto | ipm | simplex | pdhg
    std::string device = "auto";   // auto | cpu | gpu   (PDHG backend)
    int presolve = 1;              // 0 = mandatory reductions only, 1 = full
    double time_limit = 1e30;      // seconds
    double tol = 1e-8;             // IPM / simplex optimality tolerance
    double pdhg_tol = -1;          // PDHG relative KKT tolerance; <= 0: 1e-4 with crossover, 1e-6 without
    int pdhg_max_iter = 1000000;
    std::string pdhg_alg = "halpern";   // halpern | pdlp
    double pdhg_reflection = 1.0;
    int pdhg_check = 0;              // 0 = adaptive
    bool pdhg_compress = true, pdhg_graphs = true;
    std::string pdhg_precision = "auto";  // fp64 | mixed | auto
    std::string crossover = "auto";  // after PDHG: on | off | auto (on when rows + cols <= 500k)
    double mip_gap = 1e-4;         // relative MIP gap
    long long node_limit = 10000000;
    bool cuts = true;
    bool heuristics = true;
    int threads = 0;               // 0 = default
    int sb_candidates = 8, sb_reliability = 4, sb_iters = 60, cut_rounds = 15;
};

// KKT certificate evaluated on the ORIGINAL (unscaled, un-presolved) model.
struct KktReport {
    double primal_inf = 0;   // max abs violation of rows and bounds
    double dual_inf = 0;     // max abs dual sign/stationarity violation
    double rel_gap = 0;      // |primal - dual| / (1 + |primal|)
    double dual_obj = 0;
    bool available = false;
};

struct SolveResult {
    Status status = Status::NotSolved;
    std::string method;
    double objective = 0;         // in the model's ORIGINAL sense (max stays max)
    double bound = 0;             // best dual bound (MIP), original sense
    double mip_gap = 0;
    std::vector<double> x, y, z;  // original-space primal, row duals, reduced costs
    bool duals_valid = false;
    Violation viol;
    KktReport kkt;
    long long iterations = 0;
    long long nodes = 0;
    int cuts_added = 0;
    double time_total = 0, time_presolve = 0, time_solve = 0;
    int orig_rows = 0, orig_cols = 0, red_rows = 0, red_cols = 0, int_cols = 0;
    long long orig_nnz = 0;
    PresolveStats pstats;
    std::string note;
};

SolveResult solve(const Model& model, const SolverOptions& opt);

}  // namespace nirnay
