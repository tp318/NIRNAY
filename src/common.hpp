// NIRNAY — shared solver types.
#pragma once
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

namespace nirnay {

enum class Status {
    NotSolved,
    Optimal,
    Feasible,              // MIP: incumbent found, gap not closed (limit hit)
    Infeasible,
    Unbounded,
    InfeasibleOrUnbounded,
    TimeLimit,
    IterationLimit,
    NodeLimit,
    NumericalError,
};

inline const char* status_name(Status s) {
    switch (s) {
        case Status::NotSolved: return "NOT_SOLVED";
        case Status::Optimal: return "OPTIMAL";
        case Status::Feasible: return "FEASIBLE";
        case Status::Infeasible: return "INFEASIBLE";
        case Status::Unbounded: return "UNBOUNDED";
        case Status::InfeasibleOrUnbounded: return "INFEASIBLE_OR_UNBOUNDED";
        case Status::TimeLimit: return "TIME_LIMIT";
        case Status::IterationLimit: return "ITERATION_LIMIT";
        case Status::NodeLimit: return "NODE_LIMIT";
        case Status::NumericalError: return "NUMERICAL_ERROR";
    }
    return "?";
}

class Timer {
public:
    Timer() : t0_(std::chrono::steady_clock::now()) {}
    double seconds() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
    }
    void reset() { t0_ = std::chrono::steady_clock::now(); }
private:
    std::chrono::steady_clock::time_point t0_;
};

// Solution of a continuous problem in the representation handed to the engine.
struct LpSolution {
    Status status = Status::NotSolved;
    std::vector<double> x;   // primal (n)
    std::vector<double> y;   // row duals (m); sign: y >= 0 for active lower row bound (min)
    std::vector<double> z;   // reduced costs (n)
    double obj = 0.0;        // primal objective in engine space
    double dual_obj = 0.0;
    int iterations = 0;
    double pinf = 0, dinf = 0, gap = 0;  // final relative residuals (engine space)
};

extern int g_verbose;
#define NLOG(...) do { if (::nirnay::g_verbose > 0) { std::printf(__VA_ARGS__); std::fflush(stdout); } } while (0)
#define NLOG2(...) do { if (::nirnay::g_verbose > 1) { std::printf(__VA_ARGS__); std::fflush(stdout); } } while (0)

}  // namespace nirnay
