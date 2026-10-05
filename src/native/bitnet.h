// Native CPU runtime for the BitNet b1.58 decision model (see bitnet_avx2.cc).
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace bitnet {

struct Config {
  int layers = 0, hidden = 0, heads = 0, intermediate = 0, bidir_layers = 0;
  int head_width = 0, head_layers = 0, vocab = 0, max_context = 0;
  float rms_eps = 1e-5f;
};

// A weight as listed in config.json: stored in weights.bin (offset/length) or inline (data).
struct Source {
  std::string dtype;  // "float" or "int8"
  std::vector<int64_t> shape;
  int64_t offset = -1, length = 0;
  std::vector<float> data;
};

class Model;

// Implemented in bitnet_avx2.cc: call only when the CPU has AVX2 and FMA (bitnet_wrap.cc checks).
Model* Load(const std::string& weights_path, const std::map<std::string, Source>& tensors,
            const Config& config, int threads, bool vnni);
// Logits for the markers (option rows): ids/positions/segments describe one case of n tokens.
std::vector<float> Forward(Model& model, const int32_t* ids, const int32_t* positions,
                           const int32_t* segments, int n, const int32_t* markers, int m);
void Destroy(Model* model);

}  // namespace bitnet
