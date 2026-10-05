// BitNet b1.58 decision model on CPU, written for this one architecture:
//   * ternary GEMMs on int8: per-token absmax activations (uint8 + 128) x {-1, 0, 1} weights with
//     AVX-VNNI vpdpbusd (AVX2 maddubs fallback), 6x16 register tiles, fused epilogues (residual add,
//     SwiGLU on interleaved gate/up columns), weights repacked once at load;
//   * RMSNorm fused with the per-token quantization (the float activation is never stored);
//   * attention that only computes the keys a query may see (shared prefix + its own option branch);
//   * the fp32 choice head on packed-panel AVX2/FMA GEMMs, its last layer only on the option rows;
//   * a thread pool that hands out small work items, so hybrid P/E cores balance themselves.
// The arithmetic is the reference's (utils_quant.py / decision_head.py); only float summation order
// differs. Compiled with AVX2+FMA (scripts/build-native.mjs); entered only after bitnet_wrap.cc
// checked the CPU.
#include "bitnet.h"

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <chrono>

#if defined(_MSC_VER)
#define BITNET_VNNI
#else
#define BITNET_VNNI __attribute__((target("avxvnni")))
#endif

namespace bitnet {
namespace {

constexpr int MR = 6;   // rows (tokens) per register tile
constexpr int NR = 16;  // columns per tile / packed panel
constexpr float kNeg = -3.402823466e+38f;

// ------------------------------------------------------------------------------------- memory

struct Free { void operator()(void* p) const {
#if defined(_MSC_VER)
  _aligned_free(p);
#else
  std::free(p);
#endif
} };

template <class T>
struct Buf {
  std::unique_ptr<T, Free> p;
  size_t n = 0;
  T* get() const { return p.get(); }
  T& operator[](size_t i) const { return p.get()[i]; }
  void resize(size_t count) {  // grow-only, 64-byte aligned, zeroed
    if (count <= n) return;
    size_t bytes = (count * sizeof(T) + 63) & ~size_t{63};
#if defined(_MSC_VER)
    void* q = _aligned_malloc(bytes, 64);
#else
    void* q = std::aligned_alloc(64, bytes);
#endif
    if (!q) throw std::bad_alloc();
    std::memset(q, 0, bytes);
    p.reset(static_cast<T*>(q));
    n = count;
  }
};

// ---------------------------------------------------------------------------------- thread pool

class Pool {
 public:
  explicit Pool(int threads) {
    for (int i = 1; i < threads; ++i) workers_.emplace_back([this] { Loop(); });
  }
  ~Pool() {
    { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; gen_.fetch_add(1); }
    wake_.notify_all();
    for (auto& t : workers_) t.join();
  }
  int size() const { return static_cast<int>(workers_.size()) + 1; }

  // Runs fn(0..items-1) on every thread; returns when all items are done.
  void For(int items, const std::function<void(int)>& fn) {
    if (items <= 0) return;
    if (workers_.empty() || items == 1) {
      for (int i = 0; i < items; ++i) fn(i);
      return;
    }
    Region region{&fn, items};
    current_.store(&region, std::memory_order_seq_cst);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      gen_.fetch_add(1, std::memory_order_release);
    }
    if (sleeping_.load(std::memory_order_acquire)) wake_.notify_all();
    Work(region);
    while (region.done.load(std::memory_order_acquire) < items) _mm_pause();
    // A late worker either saw `region` before this store (and is counted in inside_) or sees null.
    current_.store(nullptr, std::memory_order_seq_cst);
    while (inside_.load(std::memory_order_seq_cst)) _mm_pause();
  }

 private:
  struct Region {
    const std::function<void(int)>* fn;
    int items;
    std::atomic<int> next{0}, done{0};
  };

  static void Work(Region& r) {
    for (int i; (i = r.next.fetch_add(1, std::memory_order_relaxed)) < r.items;) {
      (*r.fn)(i);
      r.done.fetch_add(1, std::memory_order_release);
    }
  }

  void Loop() {
    uint64_t seen = gen_.load();
    for (;;) {
      for (int spin = 0; gen_.load(std::memory_order_acquire) == seen; ++spin) {
        if (spin < 40000) { _mm_pause(); continue; }  // a forward runs ~200 regions back to back
        std::unique_lock<std::mutex> lock(mutex_);
        sleeping_.fetch_add(1);
        wake_.wait(lock, [&] { return gen_.load() != seen; });
        sleeping_.fetch_sub(1);
        break;
      }
      seen = gen_.load(std::memory_order_acquire);
      if (stop_) return;
      inside_.fetch_add(1, std::memory_order_seq_cst);
      if (Region* r = current_.load(std::memory_order_seq_cst)) Work(*r);
      inside_.fetch_sub(1, std::memory_order_seq_cst);
    }
  }

  std::vector<std::thread> workers_;
  std::atomic<Region*> current_{nullptr};
  std::atomic<uint64_t> gen_{0};
  std::atomic<int> inside_{0}, sleeping_{0};
  std::mutex mutex_;
  std::condition_variable wake_;
  bool stop_ = false;
};

// --------------------------------------------------------------------------------- vector math

inline float Hsum(__m256 v) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_movehdup_ps(s));
  return _mm_cvtss_f32(s);
}

inline float Hmax(__m256 v) {
  __m128 s = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_max_ps(s, _mm_movehl_ps(s, s));
  s = _mm_max_ss(s, _mm_movehdup_ps(s));
  return _mm_cvtss_f32(s);
}

// exp(x) for x <= 88.7: Cephes polynomial on the reduced argument (relative error ~2e-7).
inline __m256 Exp(__m256 x) {
  const __m256 live = _mm256_cmp_ps(x, _mm256_set1_ps(-87.3f), _CMP_GE_OQ);  // exp underflows to 0 below
  x = _mm256_max_ps(x, _mm256_set1_ps(-87.3f));
  x = _mm256_min_ps(x, _mm256_set1_ps(88.7f));
  __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                             _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375f), x);
  r = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.12194440e-4f), r);
  __m256 p = _mm256_set1_ps(1.9875691500e-4f);
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507e-3f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073e-3f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894e-2f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459e-1f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201e-1f));
  p = _mm256_fmadd_ps(_mm256_mul_ps(p, r), r, _mm256_add_ps(r, _mm256_set1_ps(1.0f)));
  __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
  return _mm256_and_ps(_mm256_mul_ps(p, _mm256_castsi256_ps(e)), live);
}

// -------------------------------------------------------------------------- ternary int8 GEMM

// B (ternary, int8 [K, N]) in panels of NR columns: panel p holds [K/4][NR][4] bytes, the layout
// vpdpbusd reads (4 consecutive k of one column per 32-bit lane).
struct Ternary {
  int K = 0, N = 0;
  Buf<int8_t> packed;
  Buf<int32_t> colsum128;  // 128 * sum_k B[k][n]: the uint8 activations carry a +128 offset
  Buf<float> scale;        // per packed column: the tensor's 1 / weight_quant scale
  Buf<float> bias;         // choice-head layers only
};

enum class Epilogue { Store, Add, SwiGLU, Relu };

template <int R>
BITNET_VNNI inline void TernaryTileVnni(const uint8_t* a, size_t lda, const int8_t* b, int K, __m256i acc[R][2]) {
  for (int r = 0; r < R; ++r) acc[r][0] = acc[r][1] = _mm256_setzero_si256();
  for (int k = 0; k < K; k += 4, b += 4 * NR) {
    const __m256i b0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(b));
    const __m256i b1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(b + 32));
    for (int r = 0; r < R; ++r) {
      const __m256i av = _mm256_castps_si256(_mm256_broadcast_ss(reinterpret_cast<const float*>(a + r * lda + k)));
      acc[r][0] = _mm256_dpbusd_avx_epi32(acc[r][0], av, b0);
      acc[r][1] = _mm256_dpbusd_avx_epi32(acc[r][1], av, b1);
    }
  }
}

template <int R>
inline void TernaryTileAvx2(const uint8_t* a, size_t lda, const int8_t* b, int K, __m256i acc[R][2]) {
  const __m256i ones = _mm256_set1_epi16(1);
  for (int r = 0; r < R; ++r) acc[r][0] = acc[r][1] = _mm256_setzero_si256();
  for (int k = 0; k < K; k += 4, b += 4 * NR) {
    const __m256i b0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(b));
    const __m256i b1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(b + 32));
    for (int r = 0; r < R; ++r) {
      const __m256i av = _mm256_castps_si256(_mm256_broadcast_ss(reinterpret_cast<const float*>(a + r * lda + k)));
      // |weight| <= 1, so the int16 pair sums (<= 510) never saturate
      acc[r][0] = _mm256_add_epi32(acc[r][0], _mm256_madd_epi16(_mm256_maddubs_epi16(av, b0), ones));
      acc[r][1] = _mm256_add_epi32(acc[r][1], _mm256_madd_epi16(_mm256_maddubs_epi16(av, b1), ones));
    }
  }
}

inline __m256 Silu(__m256 g) {  // g * sigmoid(g)
  const __m256 one = _mm256_set1_ps(1.0f);
  return _mm256_mul_ps(g, _mm256_div_ps(one, _mm256_add_ps(one, Exp(_mm256_sub_ps(_mm256_setzero_ps(), g)))));
}

// Dequantize a tile and write it: y = (acc - 128 colsum) * rowscale * colscale.
template <int R>
inline void TernaryStore(__m256i acc[R][2], const Ternary& w, int col, const float* rowscale,
                         float* out, size_t ldo, Epilogue mode) {
  const __m256i c0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(w.colsum128.get() + col));
  const __m256i c1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(w.colsum128.get() + col + 8));
  const __m256 s0 = _mm256_load_ps(w.scale.get() + col), s1 = _mm256_load_ps(w.scale.get() + col + 8);
  for (int r = 0; r < R; ++r) {
    const __m256 rs = _mm256_set1_ps(rowscale[r]);
    __m256 y0 = _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(acc[r][0], c0)), s0), rs);
    __m256 y1 = _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(acc[r][1], c1)), s1), rs);
    if (w.bias.n) {
      y0 = _mm256_add_ps(y0, _mm256_loadu_ps(w.bias.get() + col));
      y1 = _mm256_add_ps(y1, _mm256_loadu_ps(w.bias.get() + col + 8));
    }
    if (mode == Epilogue::Relu) {
      y0 = _mm256_max_ps(y0, _mm256_setzero_ps());
      y1 = _mm256_max_ps(y1, _mm256_setzero_ps());
    }
    float* o = out + r * ldo;
    if (mode == Epilogue::SwiGLU) {  // panel = 8 gate columns then the same 8 up columns
      _mm256_storeu_ps(o + col / 2, _mm256_mul_ps(Silu(y0), y1));
    } else if (mode == Epilogue::Add) {
      _mm256_storeu_ps(o + col, _mm256_add_ps(_mm256_loadu_ps(o + col), y0));
      _mm256_storeu_ps(o + col + 8, _mm256_add_ps(_mm256_loadu_ps(o + col + 8), y1));
    } else {
      _mm256_storeu_ps(o + col, y0);
      _mm256_storeu_ps(o + col + 8, y1);
    }
  }
}

template <int R>
inline void TernaryRows(const uint8_t* a, size_t lda, const Ternary& w, int panel, const float* rowscale,
                        float* out, size_t ldo, Epilogue mode, bool vnni) {
  __m256i acc[R][2];
  const int8_t* b = w.packed.get() + static_cast<size_t>(panel) * w.K * NR;
  if (vnni) TernaryTileVnni<R>(a, lda, b, w.K, acc);
  else TernaryTileAvx2<R>(a, lda, b, w.K, acc);
  TernaryStore<R>(acc, w, panel * NR, rowscale, out, ldo, mode);
}

// Work item = one 16-column panel x up to kRowBlock tokens (16 register tiles). Small items keep hybrid
// CPUs balanced: an E-core runs an item ~3x slower than a P-core, so a coarse item taken last by an
// E-core would stall the whole region. The row block (<= 96 x K bytes of activations) stays in L2 and
// the panel in L1 while its row tiles stream past.
constexpr int kRowBlock = 16 * MR;

void TernaryGemm(Pool& pool, const uint8_t* a, const float* rowscale, int M, const Ternary& w,
                 float* out, size_t ldo, Epilogue mode, bool vnni) {
  const int panels = w.N / NR, blocks = (M + kRowBlock - 1) / kRowBlock;
  pool.For(panels * blocks, [&](int item) {
    const int p = item % panels, r0 = (item / panels) * kRowBlock, r1 = std::min(M, r0 + kRowBlock);
    int r = r0;
    for (; r + MR <= r1; r += MR)
      TernaryRows<MR>(a + r * static_cast<size_t>(w.K), w.K, w, p, rowscale + r, out + r * ldo, ldo, mode, vnni);
    switch (r1 - r) {
#define TAIL(n) case n: TernaryRows<n>(a + r * static_cast<size_t>(w.K), w.K, w, p, rowscale + r, out + r * ldo, ldo, mode, vnni); break;
      TAIL(1) TAIL(2) TAIL(3) TAIL(4) TAIL(5)
#undef TAIL
      default: break;
    }
  });
}

// RMSNorm (x * rsqrt(mean(x^2) + eps) * w) of each token, then activation_quant: uint8 = round(v *
// 127 / absmax) + 128 and rowscale = absmax / 127.
void NormQuant(Pool& pool, const float* x, int M, int K, const float* weight, float eps,
               uint8_t* out, float* rowscale) {
  const int per = 4;
  pool.For((M + per - 1) / per, [&](int item) {
    for (int t = item * per; t < std::min(M, (item + 1) * per); ++t) {
      const float* xr = x + static_cast<size_t>(t) * K;
      __m256 ss = _mm256_setzero_ps();
      for (int k = 0; k < K; k += 8) {
        const __m256 v = _mm256_loadu_ps(xr + k);
        ss = _mm256_fmadd_ps(v, v, ss);
      }
      const __m256 inv = _mm256_set1_ps(1.0f / std::sqrt(Hsum(ss) / K + eps));
      const __m256 sign = _mm256_set1_ps(-0.0f);
      __m256 mx = _mm256_setzero_ps();
      for (int k = 0; k < K; k += 8) {
        const __m256 v = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(xr + k), inv), _mm256_loadu_ps(weight + k));
        mx = _mm256_max_ps(mx, _mm256_andnot_ps(sign, v));
      }
      const float absmax = std::max(Hmax(mx), 1e-5f);
      const __m256 s = _mm256_set1_ps(127.0f / absmax);
      const __m256i offset = _mm256_set1_epi32(128);
      uint8_t* o = out + static_cast<size_t>(t) * K;
      for (int k = 0; k < K; k += 16) {
        __m256 v0 = _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(xr + k), inv), _mm256_loadu_ps(weight + k)), s);
        __m256 v1 = _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(xr + k + 8), inv), _mm256_loadu_ps(weight + k + 8)), s);
        __m256i i0 = _mm256_add_epi32(_mm256_cvtps_epi32(_mm256_round_ps(v0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)), offset);
        __m256i i1 = _mm256_add_epi32(_mm256_cvtps_epi32(_mm256_round_ps(v1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)), offset);
        __m256i w16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(i0, i1), 0xD8);
        __m128i w8 = _mm_packus_epi16(_mm256_castsi256_si128(w16), _mm256_extracti128_si256(w16, 1));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(o + k), w8);
      }
      rowscale[t] = absmax / 127.0f;
    }
  });
}

// Per-token absmax int8 quantization (uint8 + 128) of x [M][K] (row stride ldx) into out [M][K].
void Quant(Pool& pool, const float* x, size_t ldx, int M, int K, uint8_t* out, float* rowscale) {
  pool.For((M + 3) / 4, [&](int item) {
    for (int t = item * 4; t < std::min(M, item * 4 + 4); ++t) {
      const float* xr = x + t * ldx;
      const __m256 sign = _mm256_set1_ps(-0.0f);
      __m256 mx = _mm256_setzero_ps();
      for (int k = 0; k < K; k += 8) mx = _mm256_max_ps(mx, _mm256_andnot_ps(sign, _mm256_loadu_ps(xr + k)));
      const float absmax = std::max(Hmax(mx), 1e-5f);
      const __m256 s = _mm256_set1_ps(127.0f / absmax);
      const __m256i offset = _mm256_set1_epi32(128);
      uint8_t* o = out + static_cast<size_t>(t) * K;
      for (int k = 0; k < K; k += 16) {
        __m256i i0 = _mm256_add_epi32(_mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(_mm256_loadu_ps(xr + k), s), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)), offset);
        __m256i i1 = _mm256_add_epi32(_mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(_mm256_loadu_ps(xr + k + 8), s), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)), offset);
        __m256i w16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(i0, i1), 0xD8);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(o + k), _mm_packus_epi16(_mm256_castsi256_si128(w16), _mm256_extracti128_si256(w16, 1)));
      }
      rowscale[t] = absmax / 127.0f;
    }
  });
}

// -------------------------------------------------------------------------------- fp32 GEMMs

// C[R][16] (+)= A[R][K] (row stride lda) * B[K][16] (row stride ldb)
template <int R>
inline void FloatTile(const float* a, size_t lda, const float* b, size_t ldb, int K, __m256 acc[R][2]) {
  for (int r = 0; r < R; ++r) acc[r][0] = acc[r][1] = _mm256_setzero_ps();
  for (int k = 0; k < K; ++k, b += ldb) {
    const __m256 b0 = _mm256_loadu_ps(b), b1 = _mm256_loadu_ps(b + 8);
    for (int r = 0; r < R; ++r) {
      const __m256 av = _mm256_broadcast_ss(a + r * lda + k);
      acc[r][0] = _mm256_fmadd_ps(av, b0, acc[r][0]);
      acc[r][1] = _mm256_fmadd_ps(av, b1, acc[r][1]);
    }
  }
}

// Dense fp32 weight [K, N] (x @ W.T as exported) packed in NR-column panels.
struct Dense {
  int K = 0, N = 0;
  Buf<float> packed, bias;
};

enum class Act { None, Relu };

template <int R>
inline void DenseRows(const float* a, size_t lda, const Dense& w, int panel, float* out, size_t ldo,
                      Act act, bool add) {
  __m256 acc[R][2];
  FloatTile<R>(a, lda, w.packed.get() + static_cast<size_t>(panel) * w.K * NR, NR, w.K, acc);
  const int col = panel * NR;
  const __m256 b0 = _mm256_loadu_ps(w.bias.get() + col), b1 = _mm256_loadu_ps(w.bias.get() + col + 8);
  for (int r = 0; r < R; ++r) {
    __m256 y0 = _mm256_add_ps(acc[r][0], b0), y1 = _mm256_add_ps(acc[r][1], b1);
    if (act == Act::Relu) {
      y0 = _mm256_max_ps(y0, _mm256_setzero_ps());
      y1 = _mm256_max_ps(y1, _mm256_setzero_ps());
    }
    float* o = out + r * ldo + col;
    if (add) {
      y0 = _mm256_add_ps(_mm256_loadu_ps(o), y0);
      y1 = _mm256_add_ps(_mm256_loadu_ps(o + 8), y1);
    }
    _mm256_storeu_ps(o, y0);
    _mm256_storeu_ps(o + 8, y1);
  }
}

// out[M][N] (+)= act(a[M][K] @ W + bias)
void DenseGemm(Pool& pool, const float* a, size_t lda, int M, const Dense& w, float* out, size_t ldo,
               Act act = Act::None, bool add = false) {
  const int panels = w.N / NR, blocks = (M + kRowBlock - 1) / kRowBlock;
  pool.For(panels * blocks, [&](int item) {
    const int p = item % panels, r0 = (item / panels) * kRowBlock, r1 = std::min(M, r0 + kRowBlock);
    int r = r0;
    for (; r + MR <= r1; r += MR) DenseRows<MR>(a + r * lda, lda, w, p, out + r * ldo, ldo, act, add);
    switch (r1 - r) {
#define TAIL(n) case n: DenseRows<n>(a + r * lda, lda, w, p, out + r * ldo, ldo, act, add); break;
      TAIL(1) TAIL(2) TAIL(3) TAIL(4) TAIL(5)
#undef TAIL
      default: break;
    }
  });
}

// ---------------------------------------------------------------------------------- attention

// One block of up to MR queries of one head (same segment), against the shared keys [a0, a0 + nk_a)
// (a0 a multiple of 16) and, for an option branch, its own keys [b0, b0 + nk_b). keys_t: the head's
// keys in tiles of 16, [key / 16][d][16] (zero padded); values: the head's [key][d]. Query r sees shared
// keys first_key[r]..last_key[r] (sliding window and causality) and its branch's keys up to last_key[r].
template <int R>
void AttentionBlock(const float* q, size_t ldq, const float* keys_t, const float* values, int d, int a0,
                    int nk_a, int b0, int nk_b, const int* first_key, const int* last_key, float scale,
                    float* scores, size_t lds, float* out, size_t ldo) {
  // The branch range starts at its tile too: the keys before b0 there belong to other segments (masked).
  const int skip_b = nk_b ? b0 % 16 : 0;
  b0 -= skip_b;
  nk_b += skip_b;
  // scores[r][0..nk_a) = q . k for the shared keys, scores[r][nk_a..) for the branch's own keys
  const int ranges[2][2] = {{a0, nk_a}, {b0, nk_b}};
  int at = 0;
  for (const auto& range : ranges) {
    for (int j = 0; j < range[1]; j += NR) {
      __m256 acc[R][2];
      FloatTile<R>(q, ldq, keys_t + static_cast<size_t>((range[0] + j) / 16) * d * 16, 16, d, acc);
      for (int r = 0; r < R; ++r) {
        _mm256_storeu_ps(scores + r * lds + at + j, _mm256_mul_ps(acc[r][0], _mm256_set1_ps(scale)));
        _mm256_storeu_ps(scores + r * lds + at + j + 8, _mm256_mul_ps(acc[r][1], _mm256_set1_ps(scale)));
      }
    }
    at += range[1];
  }
  const int total = nk_a + nk_b;
  for (int r = 0; r < R; ++r) {
    float* s = scores + r * lds;
    // mask keys past the query (causal). Score i is key i (shared range) or key b0 + i - nk_a (the
    // branch's own range); keys grow with i in each range, so only each range's tail needs masking.
    for (int i = std::max(0, last_key[r] + 1 - a0); i < nk_a; ++i) s[i] = kNeg;
    for (int i = 0; i < std::min(nk_a, first_key[r] - a0); ++i) s[i] = kNeg;
    if (nk_b) for (int i = nk_a + std::max(0, last_key[r] + 1 - b0); i < total; ++i) s[i] = kNeg;
    for (int i = nk_a; i < nk_a + skip_b; ++i) s[i] = kNeg;
    for (int i = total; i < (total + 7) / 8 * 8; ++i) s[i] = kNeg;
    __m256 mx = _mm256_set1_ps(kNeg);
    for (int i = 0; i < total; i += 8) mx = _mm256_max_ps(mx, _mm256_loadu_ps(s + i));
    const __m256 m = _mm256_set1_ps(Hmax(mx));
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < total; i += 8) {
      const __m256 e = Exp(_mm256_sub_ps(_mm256_loadu_ps(s + i), m));
      _mm256_storeu_ps(s + i, e);
      sum = _mm256_add_ps(sum, e);
    }
    const __m256 inv = _mm256_set1_ps(1.0f / Hsum(sum));
    for (int i = 0; i < total; i += 8) _mm256_storeu_ps(s + i, _mm256_mul_ps(_mm256_loadu_ps(s + i), inv));
  }
  // out[r][0..d) = sum_i p[r][i] * v(key i)
  for (int c = 0; c < d; c += NR) {
    __m256 acc[R][2], part[R][2];
    FloatTile<R>(scores, lds, values + static_cast<size_t>(a0) * d + c, d, nk_a, acc);
    if (nk_b) {
      FloatTile<R>(scores + nk_a, lds, values + static_cast<size_t>(b0) * d + c, d, nk_b, part);
      for (int r = 0; r < R; ++r) {
        acc[r][0] = _mm256_add_ps(acc[r][0], part[r][0]);
        acc[r][1] = _mm256_add_ps(acc[r][1], part[r][1]);
      }
    }
    for (int r = 0; r < R; ++r) {
      _mm256_storeu_ps(out + r * ldo + c, acc[r][0]);
      _mm256_storeu_ps(out + r * ldo + c + 8, acc[r][1]);
    }
  }
}

// Lays out the keys and values of every head for AttentionBlock: keys_t[h] = [tile][d][16] (keys t of
// tile t / 16, zero padded to `tiles` tiles) and values[h] = [t][d], both contiguous per head.
void PackKeysValues(Pool& pool, const float* k, const float* v, size_t ld, int M, int heads, int d,
                    int tiles, float* keys_t, float* values) {
  pool.For(heads * 2, [&](int item) {
    const int h = item / 2;
    if (item % 2) {
      float* dst = values + static_cast<size_t>(h) * M * d;
      for (int t = 0; t < M; ++t) std::memcpy(dst + static_cast<size_t>(t) * d, v + t * ld + h * d, d * sizeof(float));
      return;
    }
    float* dst = keys_t + static_cast<size_t>(h) * tiles * d * 16;
    std::memset(dst, 0, static_cast<size_t>(tiles) * d * 16 * sizeof(float));
    for (int t = 0; t < M; ++t) {
      const float* src = k + t * ld + h * d;
      float* tile = dst + static_cast<size_t>(t / 16) * d * 16 + t % 16;
      for (int j = 0; j < d; ++j) tile[j * 16] = src[j];
    }
  });
}

struct QueryBlock { int start, count, keys_a_start, keys_a, branch_start, keys_b; };

// BITNET_PROFILE=N: time per stage, averaged and printed every N forwards.
struct Profile {
  const bool on = std::getenv("BITNET_PROFILE") != nullptr;
  double ms[16] = {};
  int runs = 0;
  std::chrono::steady_clock::time_point last;
  void Start() { if (on) last = std::chrono::steady_clock::now(); }
  void Mark(int stage) {
    if (!on) return;
    const auto now = std::chrono::steady_clock::now();
    ms[stage] += std::chrono::duration<double, std::milli>(now - last).count();
    last = now;
  }
  void End() {
    static const int every = std::max(1, std::atoi(on ? std::getenv("BITNET_PROFILE") : "1"));
    if (!on || ++runs % every) return;
    static const char* names[] = {"embed", "norm+quant", "ternary gemm", "rope", "attention", "final norm",
                                  "head gemm", "head attention", "head layernorm", "scorer"};
    std::fprintf(stderr, "[bitnet] ms per forward:");
    for (int i = 0; i < 10; ++i) std::fprintf(stderr, " %s %.1f", names[i], ms[i] / runs);
    std::fputc('\n', stderr);
  }
};
Profile profile;

// Score rows of the attention block a thread is working on.
float* Scratch(size_t count) {
  thread_local Buf<float> buffer;
  buffer.resize(count);
  return buffer.get();
}

// --------------------------------------------------------------------------------- the model

struct Layer {
  Buf<float> input_norm, inner_norm, post_norm, ffn_norm;
  Ternary qkv, o, gate_up, down;
};

// A choice-head linear layer: int8 (7-bit per-channel weights, per-token int8 activations, on the same
// VNNI kernel as the ternary layers) or exact fp32 (BITNET_EXACT_HEAD=1).
struct HeadLinear {
  Dense f32;
  Ternary i8;
};

struct HeadLayer {
  Buf<float> norm1_w, norm1_b, norm2_w, norm2_b;
  HeadLinear kv, q, out, linear1, linear2;
};

}  // namespace

class Model {
 public:
  Config cfg;
  bool vnni = false;
  std::unique_ptr<Pool> pool;
  Buf<float> embed, rope_cos, rope_sin, final_norm;
  int rope_rows = 0;      // positions in the stored RoPE table; later ones are computed
  float inv_freq[128] = {};
  std::vector<Layer> layers;
  bool int8_head = true;
  HeadLinear proj, scorer1;
  std::vector<HeadLayer> head;
  Buf<float> scorer_norm_w, scorer_norm_b, scorer3_w;
  float scorer3_b = 0;

  // workspace (grown on demand)
  Buf<float> h, qkv, attn, mlp, keys_t, values, rowscale, z, x, kv, qh, ff, zq, xq;
  Buf<uint8_t> act;
  std::mutex busy;
};

namespace {

struct Reader {
  FILE* file;
  const std::map<std::string, Source>& tensors;

  const Source& Get(const std::string& name) const {
    auto it = tensors.find(name);
    if (it == tensors.end()) throw std::runtime_error("weights: missing tensor " + name);
    return it->second;
  }
  template <class T>
  std::vector<T> Raw(const std::string& name) const {
    const Source& s = Get(name);
    if (s.offset < 0) {
      if constexpr (std::is_same_v<T, float>) return s.data;
      throw std::runtime_error("weights: inline int8 tensor " + name);
    }
    std::vector<T> out(static_cast<size_t>(s.length) / sizeof(T));
#if defined(_MSC_VER)
    _fseeki64(file, s.offset, SEEK_SET);
#else
    fseeko(file, s.offset, SEEK_SET);
#endif
    if (std::fread(out.data(), 1, static_cast<size_t>(s.length), file) != static_cast<size_t>(s.length))
      throw std::runtime_error("weights: short read for " + name);
    return out;
  }
  bool Has(const std::string& name) const { return tensors.count(name) != 0; }
  // A float tensor, stored as fp32 or packed as fp16 (scripts/pack-model.py: name + ".f16").
  std::vector<float> FloatValues(const std::string& name) const {
    if (Has(name)) return Raw<float>(name);
    if (Get(name + ".f16").offset < 0) return Get(name + ".f16").data;  // tiny: inline in config.json
    const auto half = Raw<uint16_t>(name + ".f16");
    std::vector<float> out(half.size());
    size_t i = 0;
    for (; i + 8 <= half.size(); i += 8)
      _mm256_storeu_ps(out.data() + i, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(half.data() + i))));
    for (; i < half.size(); ++i) out[i] = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(half[i])));
    return out;
  }
  std::vector<int64_t> Shape(const std::string& name) const {
    return Get(Has(name) ? name : name + ".f16").shape;
  }
  void Floats(const std::string& name, Buf<float>& out) const {
    auto v = FloatValues(name);
    out.resize(v.size() + 16);
    std::memcpy(out.get(), v.data(), v.size() * sizeof(float));
  }
  // Ternary int8 [K, N], stored as int8 or packed 4 per byte (name + ".2bit": column 4j+s in bits 2s).
  std::vector<int8_t> TernaryValues(const std::string& name, int& K, int& N) const {
    if (Has(name)) {
      K = static_cast<int>(Get(name).shape[0]);
      N = static_cast<int>(Get(name).shape[1]);
      return Raw<int8_t>(name);
    }
    const auto packed = Raw<uint8_t>(name + ".2bit");
    K = static_cast<int>(Get(name + ".2bit").shape[0]);
    N = static_cast<int>(Get(name + ".2bit").shape[1]) * 4;
    std::vector<int8_t> out(packed.size() * 4);
    for (size_t i = 0; i < packed.size(); ++i)
      for (int s = 0; s < 4; ++s) out[i * 4 + s] = static_cast<int8_t>((packed[i] >> (2 * s) & 3) - 1);
    return out;
  }
};

// Ternary [K, N] int8 + per-column scale -> panels. perm maps packed column -> source column.
void PackTernary(const Reader& rd, const std::string& name, Ternary& w, bool swiglu) {
  const auto b = rd.TernaryValues(name + ".ternary", w.K, w.N);
  if (w.K % 4 || w.N % NR) throw std::runtime_error("weights: unsupported shape for " + name);
  const auto scale = rd.Raw<float>(name + ".scale");
  auto perm = [&](int c) {
    if (!swiglu) return c;
    const int p = c / NR, j = c % NR, half = w.N / 2;
    return j < 8 ? p * 8 + j : half + p * 8 + (j - 8);
  };
  w.packed.resize(static_cast<size_t>(w.K) * w.N);
  w.colsum128.resize(w.N);
  w.scale.resize(w.N);
  for (int c = 0; c < w.N; ++c) {
    const int s = perm(c);
    int8_t* panel = w.packed.get() + static_cast<size_t>(c / NR) * w.K * NR;
    int sum = 0;
    for (int k = 0; k < w.K; ++k) {
      const int8_t v = b[static_cast<size_t>(k) * w.N + s];
      panel[(k / 4) * 4 * NR + (c % NR) * 4 + (k % 4)] = v;
      sum += v;
    }
    w.colsum128[c] = 128 * sum;
    w.scale[c] = scale[s];
  }
}

void PackDense(const Reader& rd, const std::string& weight, const std::string& bias, Dense& w) {
  const auto shape = rd.Shape(weight);
  w.K = static_cast<int>(shape[0]);
  w.N = static_cast<int>(shape[1]);
  if (w.N % NR) throw std::runtime_error("weights: unsupported shape for " + weight);
  const auto b = rd.FloatValues(weight);
  w.packed.resize(static_cast<size_t>(w.K) * w.N);
  for (int k = 0; k < w.K; ++k)
    for (int c = 0; c < w.N; ++c)
      w.packed[static_cast<size_t>(c / NR) * w.K * NR + static_cast<size_t>(k) * NR + c % NR] = b[static_cast<size_t>(k) * w.N + c];
  rd.Floats(bias, w.bias);
}

// fp32 [K, N] -> 7-bit per-column weights (|q| <= 63, so the AVX2 maddubs pairs never saturate).
void PackInt8(const Reader& rd, const std::string& weight, const std::string& bias, Ternary& w) {
  const auto shape = rd.Shape(weight);
  w.K = static_cast<int>(shape[0]);
  w.N = static_cast<int>(shape[1]);
  if (w.K % 4 || w.N % NR) throw std::runtime_error("weights: unsupported shape for " + weight);
  const auto b = rd.FloatValues(weight);
  w.packed.resize(static_cast<size_t>(w.K) * w.N);
  w.colsum128.resize(w.N);
  w.scale.resize(w.N);
  for (int c = 0; c < w.N; ++c) {
    float absmax = 0;
    for (int k = 0; k < w.K; ++k) absmax = std::max(absmax, std::fabs(b[static_cast<size_t>(k) * w.N + c]));
    const float step = absmax > 0 ? absmax / 63.0f : 1.0f;
    int8_t* panel = w.packed.get() + static_cast<size_t>(c / NR) * w.K * NR;
    int sum = 0;
    for (int k = 0; k < w.K; ++k) {
      const int q = static_cast<int>(std::nearbyint(b[static_cast<size_t>(k) * w.N + c] / step));
      const int8_t v = static_cast<int8_t>(std::clamp(q, -63, 63));
      panel[(k / 4) * 4 * NR + (c % NR) * 4 + (k % 4)] = v;
      sum += v;
    }
    w.colsum128[c] = 128 * sum;
    w.scale[c] = step;
  }
  rd.Floats(bias, w.bias);
}

void PackHead(const Reader& rd, const std::string& weight, const std::string& bias, HeadLinear& w, bool int8) {
  if (int8) PackInt8(rd, weight, bias, w.i8);
  else PackDense(rd, weight, bias, w.f32);
}

void LayerNorm(Pool& pool, const float* x, size_t ldx, int M, int K, const float* w, const float* b, float* out, size_t ldo) {
  pool.For((M + 7) / 8, [&](int item) {
    for (int t = item * 8; t < std::min(M, item * 8 + 8); ++t) {
      const float* xr = x + t * ldx;
      __m256 s = _mm256_setzero_ps();
      for (int k = 0; k < K; k += 8) s = _mm256_add_ps(s, _mm256_loadu_ps(xr + k));
      const float mean = Hsum(s) / K;
      const __m256 m = _mm256_set1_ps(mean);
      __m256 v = _mm256_setzero_ps();
      for (int k = 0; k < K; k += 8) {
        const __m256 c = _mm256_sub_ps(_mm256_loadu_ps(xr + k), m);
        v = _mm256_fmadd_ps(c, c, v);
      }
      const __m256 inv = _mm256_set1_ps(1.0f / std::sqrt(Hsum(v) / K + 1e-5f));
      float* o = out + t * ldo;
      for (int k = 0; k < K; k += 8) {
        const __m256 c = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(xr + k), m), inv);
        _mm256_storeu_ps(o + k, _mm256_add_ps(_mm256_mul_ps(c, _mm256_loadu_ps(w + k)), _mm256_loadu_ps(b + k)));
      }
    }
  });
}

}  // namespace

Model* Load(const std::string& path, const std::map<std::string, Source>& tensors, const Config& cfg,
            int threads, bool vnni) {
  FILE* file = std::fopen(path.c_str(), "rb");
  if (!file) throw std::runtime_error("cannot open " + path);
  std::unique_ptr<FILE, int (*)(FILE*)> guard(file, std::fclose);
  auto model = std::make_unique<Model>();
  model->cfg = cfg;
  model->vnni = vnni;
  model->pool = std::make_unique<Pool>(std::max(1, threads));
  const Reader rd{file, tensors};
  rd.Floats("embed", model->embed);
  rd.Floats("rope.cos", model->rope_cos);
  rd.Floats("rope.sin", model->rope_sin);
  {
    const int d = cfg.hidden / cfg.heads;
    model->rope_rows = static_cast<int>(rd.Shape("rope.cos")[0]);
    for (int j = 0; j < d / 2; ++j)  // BitnetRotaryEmbedding: 1 / base ** (arange(0, d, 2) / d), float32
      model->inv_freq[j] = 1.0f / std::pow(10000.0f, static_cast<float>(2 * j) / static_cast<float>(d));
  }
  rd.Floats("model.norm", model->final_norm);
  model->layers.resize(cfg.layers);
  for (int i = 0; i < cfg.layers; ++i) {
    const std::string p = "model.layers." + std::to_string(i) + ".";
    Layer& l = model->layers[i];
    rd.Floats(p + "input_layernorm", l.input_norm);
    rd.Floats(p + "inner_attn_ln", l.inner_norm);
    rd.Floats(p + "post_attention_layernorm", l.post_norm);
    rd.Floats(p + "ffn_layernorm", l.ffn_norm);
    PackTernary(rd, p + "q_v", l.qkv, false);
    PackTernary(rd, p + "o", l.o, false);
    PackTernary(rd, p + "gate_up", l.gate_up, true);
    PackTernary(rd, p + "down", l.down, false);
  }
  model->int8_head = !cfg.exact_head;
  const bool i8 = model->int8_head;
  PackHead(rd, "choice_head.proj.weight.T", "choice_head.proj.bias", model->proj, i8);
  model->head.resize(cfg.head_layers);
  for (int i = 0; i < cfg.head_layers; ++i) {
    const std::string p = "choice_head.encoder.layers." + std::to_string(i) + ".";
    HeadLayer& l = model->head[i];
    rd.Floats(p + "norm1.weight", l.norm1_w);
    rd.Floats(p + "norm1.bias", l.norm1_b);
    rd.Floats(p + "norm2.weight", l.norm2_w);
    rd.Floats(p + "norm2.bias", l.norm2_b);
    PackHead(rd, p + "kv.T", p + "kv.b", l.kv, i8);
    PackHead(rd, p + "q.T", p + "q.b", l.q, i8);
    PackHead(rd, p + "self_attn.out_proj.weight.T", p + "self_attn.out_proj.bias", l.out, i8);
    PackHead(rd, p + "linear1.weight.T", p + "linear1.bias", l.linear1, i8);
    PackHead(rd, p + "linear2.weight.T", p + "linear2.bias", l.linear2, i8);
  }
  rd.Floats("choice_head.scorer.0.weight", model->scorer_norm_w);
  rd.Floats("choice_head.scorer.0.bias", model->scorer_norm_b);
  PackHead(rd, "choice_head.scorer.1.weight.T", "choice_head.scorer.1.bias", model->scorer1, i8);
  rd.Floats("choice_head.scorer.3.weight.T", model->scorer3_w);
  model->scorer3_b = rd.FloatValues("choice_head.scorer.3.bias").at(0);
  return model.release();
}

void Destroy(Model* model) { delete model; }

namespace {

// Self-attention of one layer. q/k/v: rows of `qkv` (stride ldq) at column offsets 0, hidden, 2*hidden.
void Attend(Model& m, const float* qkv, size_t ldq, int qcol, int kcol, int vcol, int heads, int d,
            const std::vector<QueryBlock>& blocks, const int* row_of, int keys, float* out, size_t ldo,
            const std::vector<int>& first_key, const std::vector<int>& last_key) {
  const int tiles = (keys + 15) / 16 + 2;  // slack: an aligned 16-key chunk may run past the last key
  m.keys_t.resize(static_cast<size_t>(heads) * tiles * d * 16);
  m.values.resize(static_cast<size_t>(heads) * keys * d + 16);
  PackKeysValues(*m.pool, qkv + kcol, qkv + vcol, ldq, keys, heads, d, tiles, m.keys_t.get(), m.values.get());
  const size_t lds = static_cast<size_t>(tiles) * 16 + 16;
  const int items = static_cast<int>(blocks.size()) * heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(d));
  m.pool->For(items, [&](int item) {
    const int h = item % heads;
    const QueryBlock& b = blocks[item / heads];
    float* s = Scratch(MR * lds);
    const float* q = qkv + static_cast<size_t>(row_of[b.start]) * ldq + qcol + h * d;
    const float* kt = m.keys_t.get() + static_cast<size_t>(h) * tiles * d * 16;
    const float* v = m.values.get() + static_cast<size_t>(h) * keys * d;
    float* o = out + static_cast<size_t>(row_of[b.start]) * ldo + h * d;
    const int* fk = first_key.data() + b.start;
    const int* lk = last_key.data() + b.start;
    switch (b.count) {
#define CASE(n) case n: AttentionBlock<n>(q, ldq, kt, v, d, b.keys_a_start, b.keys_a, b.branch_start, b.keys_b, fk, lk, scale, s, lds, o, ldo); break;
      CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6)
#undef CASE
      default: break;
    }
  });
}

}  // namespace

namespace {

// out[M][N] (+)= act(a[M][K] @ W + bias) with the head layer's int8 or fp32 weights.
void HeadGemm(Model& m, const float* a, size_t lda, int M, const HeadLinear& w, float* out, size_t ldo,
              Act act = Act::None, bool add = false) {
  if (!m.int8_head) return DenseGemm(*m.pool, a, lda, M, w.f32, out, ldo, act, add);
  Quant(*m.pool, a, lda, M, w.i8.K, m.act.get(), m.rowscale.get());
  TernaryGemm(*m.pool, m.act.get(), m.rowscale.get(), M, w.i8, out, ldo,
              add ? Epilogue::Add : act == Act::Relu ? Epilogue::Relu : Epilogue::Store, m.vnni);
}

}  // namespace

std::vector<float> Forward(Model& m, const int32_t* ids, const int32_t* positions, const int32_t* segments,
                           int n, const int32_t* markers, int count) {
  std::lock_guard<std::mutex> lock(m.busy);
  const Config& c = m.cfg;
  const int H = c.hidden, I = c.intermediate, heads = c.heads, d = H / heads;
  if (n <= 0 || n > c.max_context) throw std::runtime_error("token count out of range");
  for (int i = 0; i < n; ++i)
    if (ids[i] < 0 || ids[i] >= c.vocab || positions[i] < 0 || positions[i] >= c.max_context)
      throw std::runtime_error("token id or position out of range");
  for (int i = 0; i < count; ++i)
    if (markers[i] < 0 || markers[i] >= n) throw std::runtime_error("marker out of range");
  Pool& pool = *m.pool;
  m.h.resize(static_cast<size_t>(n) * H);
  m.act.resize(static_cast<size_t>(n) * I);
  m.rowscale.resize(n);
  m.qkv.resize(static_cast<size_t>(n) * 3 * H);
  m.attn.resize(static_cast<size_t>(n) * H);
  m.mlp.resize(static_cast<size_t>(n) * I);
  profile.Start();
  float* h = m.h.get();
  for (int t = 0; t < n; ++t) std::memcpy(h + static_cast<size_t>(t) * H, m.embed.get() + static_cast<size_t>(ids[t]) * H, H * sizeof(float));

  // Segments: 0 = shared prefix (tokens [0, prefix)), k > 0 = option branch k (contiguous spans).
  int prefix = 0;
  while (prefix < n && segments[prefix] == 0) ++prefix;
  std::vector<int> branch_start(n, 0), last_key(n);
  for (int t = prefix; t < n; ++t) branch_start[t] = (t > prefix && segments[t] == segments[t - 1]) ? branch_start[t - 1] : t;
  std::vector<int> rows(n);
  for (int t = 0; t < n; ++t) rows[t] = t;
  // Query blocks never cross a segment. causal: prefix query t sees [0, t]; a branch query sees the
  // prefix and its branch up to itself. bidirectional prefix: prefix queries see the whole prefix.
  // Every query sees only shared keys within `window` RoPE positions (prefix keys sit at their index),
  // so inputs longer than the pretraining context never attend farther than it.
  const int win = c.window;
  auto blocks_for = [&](bool bidir) {
    std::vector<QueryBlock> out;
    for (int t = 0; t < n;) {
      const bool in_prefix = t < prefix;
      int end = in_prefix ? prefix : t;
      if (!in_prefix) while (end < n && segments[end] == segments[t]) ++end;
      for (int s = t; s < end; s += MR) {
        const int cnt = std::min(MR, end - s);
        const int a0 = std::min(prefix, std::max(0, positions[s] - win + 1)) / 16 * 16;
        if (in_prefix) out.push_back({s, cnt, a0, (bidir ? std::min(prefix, s + cnt - 1 + win) : s + cnt) - a0, 0, 0});
        else out.push_back({s, cnt, a0, prefix - a0, branch_start[s], s + cnt - branch_start[s]});
      }
      t = end;
    }
    return out;
  };
  const auto causal_blocks = blocks_for(false), bidir_blocks = blocks_for(true);
  std::vector<int> first_key(n), last_causal(n), last_bidir(n);
  for (int t = 0; t < n; ++t) {
    first_key[t] = std::max(0, positions[t] - win + 1);
    last_causal[t] = t;
    last_bidir[t] = t < prefix ? std::min(prefix - 1, t + win - 1) : t;
  }

  const size_t ldq = static_cast<size_t>(3) * H;
  profile.Mark(0);
  for (int li = 0; li < c.layers; ++li) {
    Layer& L = m.layers[li];
    const bool bidir = li >= c.layers - c.bidir_layers;
    NormQuant(pool, h, n, H, L.input_norm.get(), c.rms_eps, m.act.get(), m.rowscale.get());
    profile.Mark(1);
    TernaryGemm(pool, m.act.get(), m.rowscale.get(), n, L.qkv, m.qkv.get(), ldq, Epilogue::Store, m.vnni);
    profile.Mark(2);
    // RoPE (rotate_half) on q and k
    pool.For((n + 15) / 16, [&](int item) {
      for (int t = item * 16; t < std::min(n, item * 16 + 16); ++t) {
        const float* cs = m.rope_cos.get() + static_cast<size_t>(positions[t]) * d;
        const float* sn = m.rope_sin.get() + static_cast<size_t>(positions[t]) * d;
        thread_local float far_cos[256], far_sin[256];
        if (positions[t] >= m.rope_rows) {  // past the stored table: the model's own RoPE formula
          for (int j = 0; j < d / 2; ++j) {
            const float f = static_cast<float>(positions[t]) * m.inv_freq[j];
            far_cos[j] = far_cos[j + d / 2] = std::cos(f);
            far_sin[j] = far_sin[j + d / 2] = std::sin(f);
          }
          cs = far_cos;
          sn = far_sin;
        }
        for (int part = 0; part < 2; ++part) {
          float* row = m.qkv.get() + t * ldq + part * H;
          for (int hh = 0; hh < heads; ++hh) {
            float* x = row + hh * d;
            for (int j = 0; j < d / 2; j += 8) {
              const __m256 x1 = _mm256_loadu_ps(x + j), x2 = _mm256_loadu_ps(x + j + d / 2);
              const __m256 c1 = _mm256_loadu_ps(cs + j), c2 = _mm256_loadu_ps(cs + j + d / 2);
              const __m256 s1 = _mm256_loadu_ps(sn + j), s2 = _mm256_loadu_ps(sn + j + d / 2);
              _mm256_storeu_ps(x + j, _mm256_sub_ps(_mm256_mul_ps(x1, c1), _mm256_mul_ps(x2, s1)));
              _mm256_storeu_ps(x + j + d / 2, _mm256_add_ps(_mm256_mul_ps(x2, c2), _mm256_mul_ps(x1, s2)));
            }
          }
        }
      }
    });
    profile.Mark(3);
    Attend(m, m.qkv.get(), ldq, 0, H, 2 * H, heads, d, bidir ? bidir_blocks : causal_blocks, rows.data(), n,
           m.attn.get(), H, first_key, bidir ? last_bidir : last_causal);
    profile.Mark(4);
    NormQuant(pool, m.attn.get(), n, H, L.inner_norm.get(), c.rms_eps, m.act.get(), m.rowscale.get());
    profile.Mark(1);
    TernaryGemm(pool, m.act.get(), m.rowscale.get(), n, L.o, h, H, Epilogue::Add, m.vnni);
    profile.Mark(2);
    NormQuant(pool, h, n, H, L.post_norm.get(), c.rms_eps, m.act.get(), m.rowscale.get());
    profile.Mark(1);
    TernaryGemm(pool, m.act.get(), m.rowscale.get(), n, L.gate_up, m.mlp.get(), I, Epilogue::SwiGLU, m.vnni);
    profile.Mark(2);
    NormQuant(pool, m.mlp.get(), n, I, L.ffn_norm.get(), c.rms_eps, m.act.get(), m.rowscale.get());
    profile.Mark(1);
    TernaryGemm(pool, m.act.get(), m.rowscale.get(), n, L.down, h, H, Epilogue::Add, m.vnni);
    profile.Mark(2);
  }
  // final RMSNorm (float output)
  pool.For((n + 15) / 16, [&](int item) {
    for (int t = item * 16; t < std::min(n, item * 16 + 16); ++t) {
      float* x = h + static_cast<size_t>(t) * H;
      __m256 ss = _mm256_setzero_ps();
      for (int k = 0; k < H; k += 8) { const __m256 v = _mm256_loadu_ps(x + k); ss = _mm256_fmadd_ps(v, v, ss); }
      const __m256 inv = _mm256_set1_ps(1.0f / std::sqrt(Hsum(ss) / H + c.rms_eps));
      for (int k = 0; k < H; k += 8)
        _mm256_storeu_ps(x + k, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + k), inv), _mm256_loadu_ps(m.final_norm.get() + k)));
    }
  });

  profile.Mark(5);
  // ---- choice head (fp32): projection, norm-first encoder layers, scorer on the markers
  const int W = c.head_width, hh = W / 64;
  m.z.resize(static_cast<size_t>(n) * W);
  m.x.resize(static_cast<size_t>(n) * W);
  m.kv.resize(static_cast<size_t>(n) * 2 * W);
  m.qh.resize(static_cast<size_t>(n) * W);
  m.ff.resize(static_cast<size_t>(n) * 4 * W);
  m.zq.resize(static_cast<size_t>(std::max(count, 1)) * W);
  m.xq.resize(static_cast<size_t>(std::max(count, 1)) * W);
  HeadGemm(m, h, H, n, m.proj, m.z.get(), W);
  profile.Mark(6);
  std::vector<QueryBlock> all_blocks, marker_blocks;
  for (int s = 0; s < n; s += MR) all_blocks.push_back({s, std::min(MR, n - s), 0, n, 0, 0});
  for (int s = 0; s < count; s += MR) marker_blocks.push_back({s, std::min(MR, count - s), 0, n, 0, 0});
  std::vector<int> full_key(std::max(n, count), n - 1), no_window(std::max(n, count), 0), marker_rows(count);
  for (int i = 0; i < count; ++i) marker_rows[i] = i;
  float* z = m.z.get();
  int rows_now = n;
  for (int li = 0; li < c.head_layers; ++li) {
    HeadLayer& L = m.head[li];
    const bool last = li == c.head_layers - 1;
    LayerNorm(pool, z, W, n, W, L.norm1_w.get(), L.norm1_b.get(), m.x.get(), W);
    profile.Mark(8);
    HeadGemm(m, m.x.get(), W, n, L.kv, m.kv.get(), 2 * W);
    float* zr = z;
    const float* xr = m.x.get();
    if (last) {  // only the option rows reach the scorer
      for (int i = 0; i < count; ++i) {
        std::memcpy(m.zq.get() + static_cast<size_t>(i) * W, z + static_cast<size_t>(markers[i]) * W, W * sizeof(float));
        std::memcpy(m.xq.get() + static_cast<size_t>(i) * W, m.x.get() + static_cast<size_t>(markers[i]) * W, W * sizeof(float));
      }
      zr = m.zq.get();
      xr = m.xq.get();
      rows_now = count;
    }
    HeadGemm(m, xr, W, rows_now, L.q, m.qh.get(), W);
    profile.Mark(6);
    // attention: queries in qh (stride W), keys/values in kv (stride 2W); output into ff (stride W)
    {
      const int tiles = (n + 15) / 16 + 2;
      m.keys_t.resize(static_cast<size_t>(hh) * tiles * 64 * 16);
      m.values.resize(static_cast<size_t>(hh) * n * 64 + 16);
      PackKeysValues(pool, m.kv.get(), m.kv.get() + W, 2 * W, n, hh, 64, tiles, m.keys_t.get(), m.values.get());
      const auto& blocks = last ? marker_blocks : all_blocks;
      const size_t lds = static_cast<size_t>(tiles) * 16 + 16;
      const int items = static_cast<int>(blocks.size()) * hh;
      const float scale = 1.0f / 8.0f;
      pool.For(items, [&](int item) {
        const int head = item % hh;
        const QueryBlock& b = blocks[item / hh];
        float* s = Scratch(MR * lds);
        const float* q = m.qh.get() + static_cast<size_t>(b.start) * W + head * 64;
        const float* kt = m.keys_t.get() + static_cast<size_t>(head) * tiles * 64 * 16;
        const float* v = m.values.get() + static_cast<size_t>(head) * n * 64;
        float* o = m.ff.get() + static_cast<size_t>(b.start) * W + head * 64;
        switch (b.count) {
#define CASE(k) case k: AttentionBlock<k>(q, W, kt, v, 64, 0, n, 0, 0, no_window.data(), full_key.data(), scale, s, lds, o, W); break;
          CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6)
#undef CASE
          default: break;
        }
      });
    }
    profile.Mark(7);
    HeadGemm(m, m.ff.get(), W, rows_now, L.out, zr, W, Act::None, true);
    profile.Mark(6);
    LayerNorm(pool, zr, W, rows_now, W, L.norm2_w.get(), L.norm2_b.get(), m.x.get(), W);
    profile.Mark(8);
    HeadGemm(m, m.x.get(), W, rows_now, L.linear1, m.ff.get(), 4 * W, Act::Relu);
    HeadGemm(m, m.ff.get(), 4 * W, rows_now, L.linear2, zr, W, Act::None, true);
    profile.Mark(6);
    z = zr;
  }
  // scorer: LayerNorm -> Linear -> GELU (erf) -> Linear(1)
  LayerNorm(pool, z, W, count, W, m.scorer_norm_w.get(), m.scorer_norm_b.get(), m.x.get(), W);
  HeadGemm(m, m.x.get(), W, count, m.scorer1, m.ff.get(), W);
  std::vector<float> logits(count);
  for (int i = 0; i < count; ++i) {
    const float* s = m.ff.get() + static_cast<size_t>(i) * W;
    float sum = 0.0f;  // one dot product per option
    for (int k = 0; k < W; ++k) {
      const float g = 0.5f * s[k] * (1.0f + std::erf(s[k] * 0.70710678118654752f));
      sum += g * m.scorer3_w[k];
    }
    logits[i] = sum + m.scorer3_b;
  }
  profile.Mark(9);
  profile.End();
  return logits;
}

}  // namespace bitnet
