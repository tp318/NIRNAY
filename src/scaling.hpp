// NIRNAY — matrix scaling (geometric-mean passes + equilibration, powers of two).
//
// Scale factors are rounded to powers of two so that scaling and unscaling are
// exact in binary floating point (no rounding error is introduced).
#pragma once
#include "model.hpp"

namespace nirnay {

struct Scaling {
    std::vector<double> row;  // R: A' = R A C
    std::vector<double> col;  // C: x = C x'
    double obj = 1.0;         // c' = obj * C c
    bool identity = true;
};

// keep_int_cols: integer columns get scale 1 so integrality is preserved.
Scaling compute_scaling(const Model& mdl, bool keep_int_cols, int geo_passes = 8);
Model apply_scaling(const Model& mdl, const Scaling& s);
// Map engine-space vectors back to the unscaled model.
void unscale_primal(const Scaling& s, std::vector<double>& x);
void unscale_dual(const Scaling& s, std::vector<double>& y, std::vector<double>& z);

}  // namespace nirnay
