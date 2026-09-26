#include "mps.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace nirnay {

namespace {

enum class Sec { None, Name, ObjSense, Rows, Columns, Rhs, Ranges, Bounds, Quad, End };

void split(const std::string& line, std::vector<std::string>& tok) {
    tok.clear();
    size_t i = 0, n = line.size();
    while (i < n) {
        while (i < n && (line[i] == ' ' || line[i] == '\t' || line[i] == '\r')) ++i;
        if (i >= n) break;
        size_t s = i;
        while (i < n && line[i] != ' ' && line[i] != '\t' && line[i] != '\r') ++i;
        tok.emplace_back(line.substr(s, i - s));
    }
}

double num(const std::string& s, int lineno) {
    char* end = nullptr;
    double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str()) {
        // Some generators write Infinity / inf
        if (s == "Inf" || s == "inf" || s == "Infinity" || s == "+inf") return kInf;
        if (s == "-Inf" || s == "-inf" || s == "-Infinity") return -kInf;
        throw std::runtime_error("MPS line " + std::to_string(lineno) + ": bad number '" + s + "'");
    }
    if (v >= 1e30) return kInf;
    if (v <= -1e30) return -kInf;
    return v;
}

}  // namespace

Model read_mps(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);

    Model mdl;
    std::unordered_map<std::string, int> rowmap, colmap;
    std::string obj_row;
    std::vector<char> row_type;  // 'E','L','G'
    std::vector<double> rhs;
    std::vector<double> range;
    std::vector<char> has_range;
    std::vector<int> ti, tj;
    std::vector<double> tv;
    std::vector<int> qi, qj;
    std::vector<double> qv;
    bool in_int_marker = false;
    bool quad_full = false;  // QMATRIX: both triangles supplied
    std::vector<char> bound_lo_set;

    Sec sec = Sec::None;
    std::string line;
    std::vector<std::string> tok;
    int lineno = 0;

    auto col_index = [&](const std::string& name) -> int {
        auto it = colmap.find(name);
        if (it != colmap.end()) return it->second;
        int j = (int)mdl.col_names.size();
        colmap.emplace(name, j);
        mdl.col_names.push_back(name);
        mdl.c.push_back(0.0);
        mdl.col_lo.push_back(0.0);
        mdl.col_up.push_back(kInf);
        mdl.is_int.push_back(in_int_marker ? 1 : 0);
        bound_lo_set.push_back(0);
        return j;
    };
    auto find_col = [&](const std::string& name) -> int {
        auto it = colmap.find(name);
        if (it == colmap.end())
            throw std::runtime_error("MPS line " + std::to_string(lineno) + ": unknown column " + name);
        return it->second;
    };

    while (std::getline(in, line)) {
        ++lineno;
        if (line.empty() || line[0] == '*') continue;
        bool header = !(line[0] == ' ' || line[0] == '\t');
        split(line, tok);
        if (tok.empty()) continue;
        if (header) {
            const std::string& h = tok[0];
            if (h == "NAME") { sec = Sec::Name; if (tok.size() > 1) mdl.name = tok[1]; continue; }
            if (h == "OBJSENSE") {
                sec = Sec::ObjSense;
                if (tok.size() > 1) { mdl.maximize = (tok[1] == "MAX" || tok[1] == "MAXIMIZE"); sec = Sec::None; }
                continue;
            }
            if (h == "OBJSENCE") { sec = Sec::ObjSense; continue; }
            if (h == "ROWS") { sec = Sec::Rows; continue; }
            if (h == "COLUMNS") { sec = Sec::Columns; continue; }
            if (h == "RHS") { sec = Sec::Rhs; continue; }
            if (h == "RANGES") { sec = Sec::Ranges; continue; }
            if (h == "BOUNDS") { sec = Sec::Bounds; continue; }
            if (h == "QUADOBJ" || h == "QSECTION") { sec = Sec::Quad; quad_full = false; continue; }
            if (h == "QMATRIX") { sec = Sec::Quad; quad_full = true; continue; }
            if (h == "ENDATA") { sec = Sec::End; break; }
            if (h == "MAX" || h == "MAXIMIZE") { mdl.maximize = true; continue; }
            if (h == "MIN" || h == "MINIMIZE") { continue; }
            throw std::runtime_error("MPS line " + std::to_string(lineno) + ": unsupported section " + h);
        }
        switch (sec) {
            case Sec::ObjSense:
                mdl.maximize = (tok[0] == "MAX" || tok[0] == "MAXIMIZE");
                break;
            case Sec::Rows: {
                if (tok.size() < 2) throw std::runtime_error("MPS line " + std::to_string(lineno) + ": bad ROWS entry");
                char t = (char)std::toupper(tok[0][0]);
                if (t == 'N') {
                    if (obj_row.empty()) obj_row = tok[1];
                    else rowmap.emplace(tok[1], -2);  // free row: ignored
                } else {
                    int i = (int)row_type.size();
                    rowmap.emplace(tok[1], i);
                    mdl.row_names.push_back(tok[1]);
                    row_type.push_back(t);
                    rhs.push_back(0.0); range.push_back(0.0); has_range.push_back(0);
                }
                break;
            }
            case Sec::Columns: {
                if (tok.size() >= 3 && tok[1] == "'MARKER'") {
                    if (tok[2] == "'INTORG'") in_int_marker = true;
                    else if (tok[2] == "'INTEND'") in_int_marker = false;
                    break;
                }
                if (tok.size() < 3 || tok.size() % 2 == 0)
                    throw std::runtime_error("MPS line " + std::to_string(lineno) + ": bad COLUMNS entry");
                int j = col_index(tok[0]);
                for (size_t k = 1; k + 1 < tok.size(); k += 2) {
                    double v = num(tok[k + 1], lineno);
                    if (tok[k] == obj_row) { mdl.c[j] += v; continue; }
                    auto it = rowmap.find(tok[k]);
                    if (it == rowmap.end())
                        throw std::runtime_error("MPS line " + std::to_string(lineno) + ": unknown row " + tok[k]);
                    if (it->second < 0) continue;
                    ti.push_back(it->second); tj.push_back(j); tv.push_back(v);
                }
                break;
            }
            case Sec::Rhs:
            case Sec::Ranges: {
                size_t start = (tok.size() % 2 == 1) ? 1 : 0;  // optional set name
                for (size_t k = start; k + 1 < tok.size(); k += 2) {
                    double v = num(tok[k + 1], lineno);
                    if (tok[k] == obj_row) {
                        if (sec == Sec::Rhs) mdl.obj_offset = -v;
                        continue;
                    }
                    auto it = rowmap.find(tok[k]);
                    if (it == rowmap.end())
                        throw std::runtime_error("MPS line " + std::to_string(lineno) + ": unknown row " + tok[k]);
                    if (it->second < 0) continue;
                    if (sec == Sec::Rhs) rhs[it->second] = v;
                    else { range[it->second] = v; has_range[it->second] = 1; }
                }
                break;
            }
            case Sec::Bounds: {
                std::string bt = tok[0];
                bool needs_val = !(bt == "FR" || bt == "MI" || bt == "PL" || bt == "BV");
                std::string cname; double v = 0.0;
                if (needs_val) {
                    if (tok.size() >= 4) { cname = tok[2]; v = num(tok[3], lineno); }
                    else if (tok.size() == 3) { cname = tok[1]; v = num(tok[2], lineno); }
                    else throw std::runtime_error("MPS line " + std::to_string(lineno) + ": bad BOUNDS entry");
                } else {
                    if (tok.size() >= 3) cname = tok[2];
                    else if (tok.size() == 2) cname = tok[1];
                    else throw std::runtime_error("MPS line " + std::to_string(lineno) + ": bad BOUNDS entry");
                }
                int j = find_col(cname);
                if (bt == "UP") {
                    mdl.col_up[j] = v;
                    if (v < 0 && mdl.col_lo[j] == 0.0 && !bound_lo_set[j]) mdl.col_lo[j] = -kInf;
                } else if (bt == "LO") { mdl.col_lo[j] = v; bound_lo_set[j] = 1; }
                else if (bt == "FX") { mdl.col_lo[j] = v; mdl.col_up[j] = v; bound_lo_set[j] = 1; }
                else if (bt == "FR") { mdl.col_lo[j] = -kInf; mdl.col_up[j] = kInf; bound_lo_set[j] = 1; }
                else if (bt == "MI") { mdl.col_lo[j] = -kInf; bound_lo_set[j] = 1; }
                else if (bt == "PL") { mdl.col_up[j] = kInf; }
                else if (bt == "BV") { mdl.col_lo[j] = 0; mdl.col_up[j] = 1; mdl.is_int[j] = 1; bound_lo_set[j] = 1; }
                else if (bt == "LI") { mdl.col_lo[j] = v; mdl.is_int[j] = 1; bound_lo_set[j] = 1; }
                else if (bt == "UI") {
                    mdl.col_up[j] = v; mdl.is_int[j] = 1;
                    if (v < 0 && mdl.col_lo[j] == 0.0 && !bound_lo_set[j]) mdl.col_lo[j] = -kInf;
                }
                else throw std::runtime_error("MPS line " + std::to_string(lineno) + ": unsupported bound type " + bt);
                break;
            }
            case Sec::Quad: {
                if (tok.size() < 3) throw std::runtime_error("MPS line " + std::to_string(lineno) + ": bad QUADOBJ entry");
                int a = find_col(tok[0]), b = find_col(tok[1]);
                double v = num(tok[2], lineno);
                qi.push_back(a); qj.push_back(b); qv.push_back(v);
                if (!quad_full && a != b) { qi.push_back(b); qj.push_back(a); qv.push_back(v); }
                break;
            }
            default:
                break;
        }
    }

    mdl.n = (int)mdl.col_names.size();
    mdl.m = (int)row_type.size();
    mdl.A = csc_from_triplets(mdl.m, mdl.n, ti, tj, tv);
    mdl.Q = qv.empty() ? CscMatrix{} : csc_from_triplets(mdl.n, mdl.n, qi, qj, qv);
    if (qv.empty()) mdl.Q.resize_empty(mdl.n, mdl.n);
    mdl.row_lo.resize(mdl.m); mdl.row_up.resize(mdl.m);
    for (int i = 0; i < mdl.m; ++i) {
        double r = rhs[i], R = range[i];
        switch (row_type[i]) {
            case 'E':
                if (!has_range[i]) { mdl.row_lo[i] = r; mdl.row_up[i] = r; }
                else if (R >= 0) { mdl.row_lo[i] = r; mdl.row_up[i] = r + R; }
                else { mdl.row_lo[i] = r + R; mdl.row_up[i] = r; }
                break;
            case 'L':
                mdl.row_up[i] = r;
                mdl.row_lo[i] = has_range[i] ? r - std::fabs(R) : -kInf;
                break;
            case 'G':
                mdl.row_lo[i] = r;
                mdl.row_up[i] = has_range[i] ? r + std::fabs(R) : kInf;
                break;
            default:
                throw std::runtime_error("unknown row type");
        }
    }
    // Binary default for integer columns declared by marker with no bounds is [0, inf)
    if (mdl.maximize) {
        for (double& v : mdl.c) v = -v;
        for (double& v : mdl.Q.val) v = -v;
        mdl.obj_offset = -mdl.obj_offset;
    }
    return mdl;
}

void write_mps(const Model& mdl, const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) throw std::runtime_error("cannot write " + path);
    std::fprintf(f, "NAME %s\n", mdl.name.empty() ? "NIRNAY" : mdl.name.c_str());
    if (mdl.maximize) std::fprintf(f, "OBJSENSE\n    MAX\n");
    std::fprintf(f, "ROWS\n N  OBJ\n");
    auto rname = [&](int i) { return mdl.row_names.size() == (size_t)mdl.m ? mdl.row_names[i] : "R" + std::to_string(i); };
    auto cname = [&](int j) { return mdl.col_names.size() == (size_t)mdl.n ? mdl.col_names[j] : "C" + std::to_string(j); };
    const double sgn = mdl.maximize ? -1.0 : 1.0;
    for (int i = 0; i < mdl.m; ++i) {
        const char* t = (mdl.row_lo[i] == mdl.row_up[i]) ? "E" : (is_finite(mdl.row_lo[i]) ? "G" : "L");
        std::fprintf(f, " %s  %s\n", t, rname(i).c_str());
    }
    std::fprintf(f, "COLUMNS\n");
    bool in_int = false;
    int mk = 0;
    for (int j = 0; j < mdl.n; ++j) {
        bool isint = !mdl.is_int.empty() && mdl.is_int[j];
        if (isint != in_int) {
            std::fprintf(f, "    M%d  'MARKER'  '%s'\n", mk++, isint ? "INTORG" : "INTEND");
            in_int = isint;
        }
        if (mdl.c[j] != 0.0) std::fprintf(f, "    %s  OBJ  %.17g\n", cname(j).c_str(), sgn * mdl.c[j]);
        for (int p = mdl.A.colptr[j]; p < mdl.A.colptr[j + 1]; ++p)
            std::fprintf(f, "    %s  %s  %.17g\n", cname(j).c_str(), rname(mdl.A.rowidx[p]).c_str(), mdl.A.val[p]);
        if (mdl.c[j] == 0.0 && mdl.A.colptr[j] == mdl.A.colptr[j + 1])
            std::fprintf(f, "    %s  OBJ  0\n", cname(j).c_str());
    }
    if (in_int) std::fprintf(f, "    M%d  'MARKER'  'INTEND'\n", mk++);
    std::fprintf(f, "RHS\n");
    if (mdl.obj_offset != 0.0) std::fprintf(f, "    RHS  OBJ  %.17g\n", -sgn * mdl.obj_offset);
    for (int i = 0; i < mdl.m; ++i) {
        double r = (mdl.row_lo[i] == mdl.row_up[i]) ? mdl.row_lo[i]
                   : (is_finite(mdl.row_lo[i]) ? mdl.row_lo[i] : mdl.row_up[i]);
        if (r != 0.0 && is_finite(r)) std::fprintf(f, "    RHS  %s  %.17g\n", rname(i).c_str(), r);
    }
    std::fprintf(f, "RANGES\n");
    for (int i = 0; i < mdl.m; ++i)
        if (mdl.row_lo[i] != mdl.row_up[i] && is_finite(mdl.row_lo[i]) && is_finite(mdl.row_up[i]))
            std::fprintf(f, "    RNG  %s  %.17g\n", rname(i).c_str(), mdl.row_up[i] - mdl.row_lo[i]);
    std::fprintf(f, "BOUNDS\n");
    for (int j = 0; j < mdl.n; ++j) {
        double lo = mdl.col_lo[j], up = mdl.col_up[j];
        std::string cn = cname(j);
        if (lo == up) { std::fprintf(f, " FX BND  %s  %.17g\n", cn.c_str(), lo); continue; }
        if (!is_finite(lo) && !is_finite(up)) { std::fprintf(f, " FR BND  %s\n", cn.c_str()); continue; }
        if (!is_finite(lo)) std::fprintf(f, " MI BND  %s\n", cn.c_str());
        else if (lo != 0.0) std::fprintf(f, " LO BND  %s  %.17g\n", cn.c_str(), lo);
        if (is_finite(up)) std::fprintf(f, " UP BND  %s  %.17g\n", cn.c_str(), up);
        else if (!mdl.is_int.empty() && mdl.is_int[j]) std::fprintf(f, " PL BND  %s\n", cn.c_str());
    }
    if (mdl.has_q()) {
        std::fprintf(f, "QUADOBJ\n");
        for (int j = 0; j < mdl.n; ++j)
            for (int p = mdl.Q.colptr[j]; p < mdl.Q.colptr[j + 1]; ++p) {
                int i = mdl.Q.rowidx[p];
                if (i >= j) std::fprintf(f, "    %s  %s  %.17g\n", cname(j).c_str(), cname(i).c_str(), sgn * mdl.Q.val[p]);
            }
    }
    std::fprintf(f, "ENDATA\n");
    std::fclose(f);
}

}  // namespace nirnay
