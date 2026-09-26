// NIRNAY — MPS / QPS reader and writer (free and whitespace-separated fixed format).
#pragma once
#include <string>

#include "model.hpp"

namespace nirnay {

// Reads an (optionally QPS) MPS file. Throws std::runtime_error on malformed input.
Model read_mps(const std::string& path);
void write_mps(const Model& mdl, const std::string& path);

}  // namespace nirnay
