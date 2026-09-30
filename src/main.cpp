// NIRNAY — command-line interface.
//
//   nirnay <model.mps> [options]
//
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "mps.hpp"
#include "solver.hpp"

#ifndef NIRNAY_VERSION
#define NIRNAY_VERSION "0.1.0"
#endif

using namespace nirnay;

namespace {

void usage() {
    std::printf(
        "NIRNAY " NIRNAY_VERSION " optimisation engine (LP / convex QP / MILP)\n"
        "usage: nirnay [solve] <model.mps|.qps> [options]\n"
        "  --threads <n>                    CPU threads for parallel kernels (default: all cores)\n"
        "  --version\n"
        "  --method auto|ipm|simplex|pdhg   continuous engine (default auto)\n"
        "  --device auto|cpu|gpu            PDHG backend (default auto)\n"
        "  --presolve on|off                (default on)\n"
        "  --time-limit <sec>\n"
        "  --tol <eps>                      IPM/simplex tolerance (default 1e-8)\n"
        "  --pdhg-tol <eps>                 PDHG relative KKT tolerance (default 1e-4)\n"
        "  --pdhg-alg halpern|pdlp          PDHG variant (default halpern)\n"
        "  --crossover on|off|auto          simplex crossover after PDHG (auto: rows + cols <= 500k)\n"
        "  --pdhg-check <n>                 iterations between KKT checks (default adaptive)\n"
        "  --pdhg-compress on|off           GPU: lossless compressed matrix (default on)\n"
        "  --pdhg-graphs on|off             GPU: CUDA-graph iteration batches (default on)\n"
        "  --gap <rel>                      MIP relative gap (default 1e-4)\n"
        "  --node-limit <n>\n"
        "  --no-cuts  --no-heuristics\n"
        "  --json <file>                    write machine-readable result\n"
        "  --sol <file>                     write primal solution\n"
        "  --quiet | --verbose\n");
}

std::string json_num(double v) {
    if (std::isnan(v)) return "null";
    if (std::isinf(v)) return v > 0 ? "1e308" : "-1e308";
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.17g", v);
    return buf;
}

std::string json_escape(const std::string& s) {
    std::string o;
    for (char ch : s) {
        if (ch == '"' || ch == '\\') { o += '\\'; o += ch; }
        else if ((unsigned char)ch < 0x20) o += ' ';
        else o += ch;
    }
    return o;
}

void write_json(const std::string& path, const std::string& file, const SolveResult& r) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", path.c_str()); return; }
    std::fprintf(f, "{\n");
    std::fprintf(f, "  \"solver\": \"NIRNAY\",\n");
    std::fprintf(f, "  \"file\": \"%s\",\n", json_escape(file).c_str());
    std::fprintf(f, "  \"status\": \"%s\",\n", status_name(r.status));
    std::fprintf(f, "  \"method\": \"%s\",\n", json_escape(r.method).c_str());
    std::fprintf(f, "  \"objective\": %s,\n", json_num(r.objective).c_str());
    std::fprintf(f, "  \"bound\": %s,\n", json_num(r.bound).c_str());
    std::fprintf(f, "  \"mip_gap\": %s,\n", json_num(r.mip_gap).c_str());
    std::fprintf(f, "  \"time_total\": %s,\n", json_num(r.time_total).c_str());
    std::fprintf(f, "  \"time_presolve\": %s,\n", json_num(r.time_presolve).c_str());
    std::fprintf(f, "  \"time_solve\": %s,\n", json_num(r.time_solve).c_str());
    std::fprintf(f, "  \"iterations\": %lld,\n", r.iterations);
    std::fprintf(f, "  \"nodes\": %lld,\n", r.nodes);
    std::fprintf(f, "  \"cuts\": %d,\n", r.cuts_added);
    std::fprintf(f, "  \"rows\": %d, \"cols\": %d, \"nnz\": %lld, \"int_cols\": %d,\n", r.orig_rows, r.orig_cols, r.orig_nnz, r.int_cols);
    std::fprintf(f, "  \"presolved_rows\": %d, \"presolved_cols\": %d,\n", r.red_rows, r.red_cols);
    std::fprintf(f, "  \"max_row_violation\": %s,\n", json_num(r.viol.max_row).c_str());
    std::fprintf(f, "  \"max_bound_violation\": %s,\n", json_num(r.viol.max_bound).c_str());
    std::fprintf(f, "  \"max_int_violation\": %s,\n", json_num(r.viol.max_int).c_str());
    std::fprintf(f, "  \"kkt_available\": %s,\n", r.kkt.available ? "true" : "false");
    std::fprintf(f, "  \"kkt_primal_inf\": %s,\n", json_num(r.kkt.primal_inf).c_str());
    std::fprintf(f, "  \"kkt_dual_inf\": %s,\n", json_num(r.kkt.dual_inf).c_str());
    std::fprintf(f, "  \"kkt_rel_gap\": %s,\n", json_num(r.kkt.rel_gap).c_str());
    std::fprintf(f, "  \"note\": \"%s\"\n", json_escape(r.note).c_str());
    std::fprintf(f, "}\n");
    std::fclose(f);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 1; }
    std::string file, json, solfile;
    SolverOptions opt;
    int first = 1;
    if (std::strcmp(argv[1], "solve") == 0) first = 2;  // `nirnay solve model.mps` == `nirnay model.mps`
    for (int a = first; a < argc; ++a) {
        std::string s = argv[a];
        auto next = [&]() -> std::string {
            if (a + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", s.c_str()); std::exit(1); }
            return argv[++a];
        };
        if (s == "--help" || s == "-h") { usage(); return 0; }
        else if (s == "--version") { std::printf("NIRNAY %s\n", NIRNAY_VERSION); return 0; }
        else if (s == "--method") opt.method = next();
        else if (s == "--device") opt.device = next();
        else if (s == "--presolve") opt.presolve = next() == "off" ? 0 : 1;
        else if (s == "--time-limit") opt.time_limit = std::atof(next().c_str());
        else if (s == "--tol") opt.tol = std::atof(next().c_str());
        else if (s == "--pdhg-tol") opt.pdhg_tol = std::atof(next().c_str());
        else if (s == "--pdhg-alg") opt.pdhg_alg = next();
        else if (s == "--crossover") opt.crossover = next();
        else if (s == "--pdhg-precision") opt.pdhg_precision = next();
        else if (s == "--pdhg-reflect") opt.pdhg_reflection = std::atof(next().c_str());
        else if (s == "--pdhg-check") opt.pdhg_check = std::atoi(next().c_str());
        else if (s == "--pdhg-compress") opt.pdhg_compress = next() != "off";
        else if (s == "--pdhg-graphs") opt.pdhg_graphs = next() != "off";
        else if (s == "--pdhg-max-iter") opt.pdhg_max_iter = std::atoi(next().c_str());
        else if (s == "--gap") opt.mip_gap = std::atof(next().c_str());
        else if (s == "--node-limit") opt.node_limit = std::atoll(next().c_str());
        else if (s == "--no-cuts") opt.cuts = false;
        else if (s == "--no-heuristics") opt.heuristics = false;
        else if (s == "--threads") opt.threads = std::atoi(next().c_str());
        else if (s == "--sb-cands") opt.sb_candidates = std::atoi(next().c_str());
        else if (s == "--sb-rel") opt.sb_reliability = std::atoi(next().c_str());
        else if (s == "--sb-iters") opt.sb_iters = std::atoi(next().c_str());
        else if (s == "--cut-rounds") opt.cut_rounds = std::atoi(next().c_str());
        else if (s == "--json") json = next();
        else if (s == "--sol") solfile = next();
        else if (s == "--quiet") g_verbose = 0;
        else if (s == "--verbose") g_verbose = 2;
        else if (!s.empty() && s[0] == '-') { std::fprintf(stderr, "unknown option %s\n", s.c_str()); return 1; }
        else file = s;
    }
    if (file.empty()) { usage(); return 1; }
#ifdef _OPENMP
    if (opt.threads > 0) omp_set_num_threads(opt.threads);
#endif

    Model mdl;
    Timer tr;
    try {
        mdl = read_mps(file);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
    NLOG("NIRNAY | model %s: %d rows, %d cols, %d nnz, %d integer, %s%s  (read %.3fs)\n",
         mdl.name.c_str(), mdl.m, mdl.n, mdl.A.nnz(), mdl.num_int(), mdl.has_q() ? "QP, " : "",
         mdl.maximize ? "maximize" : "minimize", tr.seconds());

    SolveResult r = solve(mdl, opt);

    std::printf("Status        : %s\n", status_name(r.status));
    std::printf("Method        : %s\n", r.method.c_str());
    if (!r.x.empty()) {
        std::printf("Objective     : %.12e\n", r.objective);
        if (r.int_cols > 0) std::printf("Best bound    : %.12e   (gap %.4g%%, nodes %lld, cuts %d)\n", r.bound, 100 * r.mip_gap, r.nodes, r.cuts_added);
        std::printf("Violations    : row %.2e  bound %.2e  integrality %.2e\n", r.viol.max_row, r.viol.max_bound, r.viol.max_int);
        if (r.kkt.available)
            std::printf("KKT (orig.)   : rel.primal %.2e  rel.dual %.2e  rel.gap %.2e\n", r.kkt.primal_inf, r.kkt.dual_inf, r.kkt.rel_gap);
    }
    std::printf("Iterations    : %lld\n", r.iterations);
    std::printf("Time          : %.3fs (presolve %.3fs, solve %.3fs)\n", r.time_total, r.time_presolve, r.time_solve);
    if (!r.note.empty()) std::printf("Note          : %s\n", r.note.c_str());

    if (!json.empty()) write_json(json, file, r);
    if (!solfile.empty() && !r.x.empty()) {
        FILE* f = std::fopen(solfile.c_str(), "w");
        if (f) {
            std::fprintf(f, "# NIRNAY solution  status %s  objective %.17g\n", status_name(r.status), r.objective);
            for (int j = 0; j < mdl.n; ++j)
                std::fprintf(f, "%s %.17g\n", mdl.col_names.empty() ? ("C" + std::to_string(j)).c_str() : mdl.col_names[j].c_str(), r.x[j]);
            if (r.duals_valid) {
                std::fprintf(f, "# row duals\n");
                for (int i = 0; i < mdl.m; ++i)
                    std::fprintf(f, "%s %.17g\n", mdl.row_names.empty() ? ("R" + std::to_string(i)).c_str() : mdl.row_names[i].c_str(), r.y[i]);
            }
            std::fclose(f);
        }
    }
    return 0;
}
