// N-API front of the native BitNet runtime (bitnet_avx2.cc): CPU detection, loading and forwards on
// libuv worker threads. This file is built without AVX flags so it is safe on any x86/ARM CPU.
#include <napi.h>

#include <atomic>
#include <memory>

#include "bitnet.h"

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__)
#include <cpuid.h>
#endif

namespace {

struct Cpu {
  bool supported = false, vnni = false;
};

Cpu Detect() {
  Cpu cpu;
#if defined(_M_X64) || defined(__x86_64__)
  unsigned r1[4] = {}, r7[4] = {}, r71[4] = {};
#if defined(_MSC_VER)
  __cpuidex(reinterpret_cast<int*>(r1), 1, 0);
  __cpuidex(reinterpret_cast<int*>(r7), 7, 0);
  __cpuidex(reinterpret_cast<int*>(r71), 7, 1);
  const unsigned long long xcr0 = (r1[2] >> 27 & 1) ? _xgetbv(0) : 0;
#else
  __cpuid_count(1, 0, r1[0], r1[1], r1[2], r1[3]);
  __cpuid_count(7, 0, r7[0], r7[1], r7[2], r7[3]);
  __cpuid_count(7, 1, r71[0], r71[1], r71[2], r71[3]);
  unsigned lo = 0, hi = 0;
  if (r1[2] >> 27 & 1) __asm__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
  const unsigned long long xcr0 = (static_cast<unsigned long long>(hi) << 32) | lo;
#endif
  const bool ymm = (xcr0 & 6) == 6;  // the OS saves YMM state
  const bool fma = r1[2] >> 12 & 1, avx = r1[2] >> 28 & 1, f16c = r1[2] >> 29 & 1, avx2 = r7[1] >> 5 & 1;
  cpu.supported = ymm && avx && fma && f16c && avx2;
  cpu.vnni = cpu.supported && (r71[0] >> 4 & 1);  // AVX-VNNI (Alder Lake / Zen 5 and later)
#endif
  return cpu;
}

const Cpu kCpu = Detect();

class BitnetModel : public Napi::ObjectWrap<BitnetModel> {
 public:
  static Napi::FunctionReference constructor;

  static void Init(Napi::Env env, Napi::Object exports) {
    Napi::Function cls = DefineClass(env, "BitnetModel", {
        StaticMethod("load", &BitnetModel::Load),
        InstanceMethod("run", &BitnetModel::Run),
        InstanceMethod("release", &BitnetModel::Release),
    });
    constructor = Napi::Persistent(cls);
    constructor.SuppressDestruct();
    exports.Set("BitnetModel", cls);
    exports.Set("bitnetCpu", Napi::Function::New(env, [](const Napi::CallbackInfo& info) {
      Napi::Object out = Napi::Object::New(info.Env());
      out.Set("supported", kCpu.supported);
      out.Set("kernel", kCpu.vnni ? "avx-vnni" : kCpu.supported ? "avx2" : "none");
      return out;
    }));
  }

  explicit BitnetModel(const Napi::CallbackInfo& info) : Napi::ObjectWrap<BitnetModel>(info) {
    if (info.Length() != 1 || !info[0].IsExternal()) {
      throw Napi::TypeError::New(info.Env(), "Use BitnetModel.load(weightsPath, tensors, config, threads).");
    }
    model_ = info[0].As<Napi::External<bitnet::Model>>().Data();
  }

  ~BitnetModel() override { Free(); }

  std::atomic<int> pending{0};
  bool released = false;
  bitnet::Model* model_ = nullptr;

  void Free() {
    if (model_ && pending.load() == 0) {
      bitnet::Destroy(model_);
      model_ = nullptr;
    }
  }

 private:
  // load(weightsPath: string, tensors: Record<string, {dtype, shape, offset?, length?, data?}>,
  //      config: {...}, threads: number) => Promise<BitnetModel>
  static Napi::Value Load(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (!kCpu.supported) throw Napi::Error::New(env, "The native BitNet runtime needs an x86-64 CPU with AVX2 and FMA.");
    if (info.Length() != 4 || !info[0].IsString() || !info[1].IsObject() || !info[2].IsObject() || !info[3].IsNumber()) {
      throw Napi::TypeError::New(env, "load(weightsPath, tensors, config, threads)");
    }
    std::map<std::string, bitnet::Source> tensors;
    Napi::Object table = info[1].As<Napi::Object>();
    Napi::Array names = table.GetPropertyNames();
    for (uint32_t i = 0; i < names.Length(); ++i) {
      const std::string name = names.Get(i).As<Napi::String>();
      Napi::Object t = table.Get(name).As<Napi::Object>();
      bitnet::Source s;
      s.dtype = t.Get("dtype").As<Napi::String>();
      Napi::Array shape = t.Get("shape").As<Napi::Array>();
      for (uint32_t j = 0; j < shape.Length(); ++j) s.shape.push_back(shape.Get(j).As<Napi::Number>().Int64Value());
      if (t.Has("offset")) {
        s.offset = t.Get("offset").As<Napi::Number>().Int64Value();
        s.length = t.Get("length").As<Napi::Number>().Int64Value();
      } else {
        Napi::Array data = t.Get("data").As<Napi::Array>();
        for (uint32_t j = 0; j < data.Length(); ++j) s.data.push_back(data.Get(j).As<Napi::Number>().FloatValue());
      }
      tensors.emplace(name, std::move(s));
    }
    Napi::Object c = info[2].As<Napi::Object>();
    auto num = [&](const char* key) { return c.Get(key).As<Napi::Number>(); };
    bitnet::Config config;
    config.layers = num("layers").Int32Value();
    config.hidden = num("hidden").Int32Value();
    config.heads = num("heads").Int32Value();
    config.intermediate = num("intermediate").Int32Value();
    config.bidir_layers = num("bidir_layers").Int32Value();
    config.head_width = num("head_width").Int32Value();
    config.head_layers = num("head_layers").Int32Value();
    config.vocab = num("vocab").Int32Value();
    config.max_context = num("max_context").Int32Value();
    config.rms_eps = num("rms_eps").FloatValue();
    const int threads = info[3].As<Napi::Number>().Int32Value();
    auto* worker = new LoadWorker(env, info[0].As<Napi::String>(), std::move(tensors), config, threads);
    worker->Queue();
    return worker->deferred.Promise();
  }

  // run(ids, positions, segments, markers: Int32Array) => Promise<Float32Array>
  Napi::Value Run(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (!model_ || released) throw Napi::Error::New(env, "The model was released.");
    std::vector<int32_t> arrays[4];
    for (int i = 0; i < 4; ++i) {
      if (!info[i].IsTypedArray() || info[i].As<Napi::TypedArray>().TypedArrayType() != napi_int32_array) {
        throw Napi::TypeError::New(env, "run(ids, positions, segments, markers) takes four Int32Arrays.");
      }
      Napi::Int32Array a = info[i].As<Napi::Int32Array>();
      arrays[i].assign(a.Data(), a.Data() + a.ElementLength());
    }
    if (arrays[1].size() != arrays[0].size() || arrays[2].size() != arrays[0].size()) {
      throw Napi::RangeError::New(env, "ids, positions and segments must have the same length.");
    }
    auto* worker = new RunWorker(env, this, std::move(arrays));
    worker->Queue();
    return worker->deferred.Promise();
  }

  Napi::Value Release(const Napi::CallbackInfo& info) {
    released = true;
    Free();
    return info.Env().Undefined();
  }

  class LoadWorker : public Napi::AsyncWorker {
   public:
    LoadWorker(Napi::Env env, std::string path, std::map<std::string, bitnet::Source> tensors,
               bitnet::Config config, int threads)
        : Napi::AsyncWorker(env), deferred(Napi::Promise::Deferred::New(env)), path_(std::move(path)),
          tensors_(std::move(tensors)), config_(config), threads_(threads) {}
    Napi::Promise::Deferred deferred;
    void Execute() override {
      try {
        model_ = bitnet::Load(path_, tensors_, config_, threads_, kCpu.vnni);
      } catch (const std::exception& e) {
        SetError(e.what());
      }
    }
    void OnOK() override {
      deferred.Resolve(constructor.New({Napi::External<bitnet::Model>::New(Env(), model_)}));
    }
    void OnError(const Napi::Error& e) override { deferred.Reject(e.Value()); }

   private:
    std::string path_;
    std::map<std::string, bitnet::Source> tensors_;
    bitnet::Config config_;
    int threads_;
    bitnet::Model* model_ = nullptr;
  };

  class RunWorker : public Napi::AsyncWorker {
   public:
    RunWorker(Napi::Env env, BitnetModel* owner, std::vector<int32_t> (&&arrays)[4])
        : Napi::AsyncWorker(env), deferred(Napi::Promise::Deferred::New(env)), owner_(owner),
          ref_(Napi::Persistent(owner->Value())) {
      for (int i = 0; i < 4; ++i) arrays_[i] = std::move(arrays[i]);
      owner_->pending.fetch_add(1);
    }
    Napi::Promise::Deferred deferred;
    void Execute() override {
      try {
        logits_ = bitnet::Forward(*owner_->model_, arrays_[0].data(), arrays_[1].data(), arrays_[2].data(),
                                  static_cast<int>(arrays_[0].size()), arrays_[3].data(),
                                  static_cast<int>(arrays_[3].size()));
      } catch (const std::exception& e) {
        SetError(e.what());
      }
    }
    void OnOK() override {
      Napi::Float32Array out = Napi::Float32Array::New(Env(), logits_.size());
      std::copy(logits_.begin(), logits_.end(), out.Data());
      Done();
      deferred.Resolve(out);
    }
    void OnError(const Napi::Error& e) override {
      Done();
      deferred.Reject(e.Value());
    }

   private:
    void Done() {
      owner_->pending.fetch_sub(1);
      if (owner_->released) owner_->Free();
    }
    BitnetModel* owner_;
    Napi::ObjectReference ref_;  // keeps the wrapper (and its model) alive while running
    std::vector<int32_t> arrays_[4];
    std::vector<float> logits_;
  };
};

Napi::FunctionReference BitnetModel::constructor;

}  // namespace

void InitBitnet(Napi::Env env, Napi::Object exports) { BitnetModel::Init(env, exports); }
