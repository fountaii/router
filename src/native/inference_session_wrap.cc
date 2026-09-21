// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_cxx_api.h"

#include "common.h"
#include "cuda_transfer.h"
#include "inference_session_wrap.h"
#include "ort_instance_data.h"
#include "ort_singleton_data.h"
#include "run_options_helper.h"
#include "session_options_helper.h"
#include "tensor_helper.h"
#include <string>

namespace {
size_t TensorByteSize(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return sizeof(float);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return sizeof(int64_t);
    default:
      throw std::runtime_error("CUDA graph only supports float32 and int64 tensors.");
  }
}
}  // namespace

Napi::Object InferenceSessionWrap::Init(Napi::Env env, Napi::Object exports) {
  // create ONNX runtime env
  const auto* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ORT_NAPI_THROW_ERROR_IF(api == nullptr, env, "ONNX Runtime API version mismatch.");
  Ort::InitApi(api);

  // initialize binding
  Napi::HandleScope scope(env);

  Napi::Function func = DefineClass(
      env, "InferenceSession",
      {InstanceMethod("loadModel", &InferenceSessionWrap::LoadModel),
       InstanceMethod("run", &InferenceSessionWrap::Run),
       InstanceMethod("dispose", &InferenceSessionWrap::Dispose),
       InstanceMethod("endProfiling", &InferenceSessionWrap::EndProfiling),
       InstanceAccessor("inputMetadata", &InferenceSessionWrap::GetMetadata, nullptr, napi_default, reinterpret_cast<void*>(true)),
       InstanceAccessor("outputMetadata", &InferenceSessionWrap::GetMetadata, nullptr, napi_default, reinterpret_cast<void*>(false))});

  OrtInstanceData::Create(env, func);

  exports.Set("InferenceSession", func);

  Napi::Function listSupportedBackends = Napi::Function::New(env, InferenceSessionWrap::ListSupportedBackends);
  exports.Set("listSupportedBackends", listSupportedBackends);

  Napi::Function initOrtOnce = Napi::Function::New(env, InferenceSessionWrap::InitOrtOnce);
  exports.Set("initOrtOnce", initOrtOnce);

  return exports;
}

Napi::Value InferenceSessionWrap::InitOrtOnce(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  int log_level = info[0].As<Napi::Number>().Int32Value();
  Napi::Function tensorConstructor = info[1].As<Napi::Function>();
  bool is_main_thread = info[2].As<Napi::Boolean>().Value();

  OrtInstanceData::InitOrt(env, log_level, tensorConstructor, is_main_thread);

  return env.Undefined();
}

InferenceSessionWrap::InferenceSessionWrap(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<InferenceSessionWrap>(info), initialized_(false), disposed_(false), session_(nullptr) {}

InferenceSessionWrap::~InferenceSessionWrap() {
  // If the ORT singleton has already been destroyed (e.g. during process shutdown when the
  // cleanup hook fires before N-API finalizers run), we must not call into ORT to
  // release owned ORT objects — doing so would crash. Intentionally leak in that case.
  if (!OrtSingletonData::GetOrtObjects()) {
    (void)session_.release();
  }
}

Napi::Value InferenceSessionWrap::LoadModel(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  ORT_NAPI_THROW_ERROR_IF(this->initialized_, env, "Model already loaded. Cannot load model multiple times.");
  ORT_NAPI_THROW_ERROR_IF(this->disposed_, env, "Session already disposed.");

  size_t argsLength = info.Length();
  ORT_NAPI_THROW_TYPEERROR_IF(argsLength == 0, env, "Expect argument: model file path.");

  try {
    ORT_NAPI_THROW_ERROR_IF(!OrtSingletonData::GetOrtObjects(), env, "Call initOrtOnce before loading a model.");
    Ort::SessionOptions sessionOptions;

    if (argsLength == 2 && info[0].IsString() && info[1].IsObject()) {
      Napi::String value = info[0].As<Napi::String>();

      auto jsOptions = info[1].As<Napi::Object>();
      ORT_NAPI_THROW_TYPEERROR_IF(jsOptions.Has("enableCudaGraph") && !jsOptions.Get("enableCudaGraph").IsBoolean(), env,
                                  "sessionOptions.enableCudaGraph must be a boolean value.");
      this->cudaGraph_ = jsOptions.Has("enableCudaGraph") && jsOptions.Get("enableCudaGraph").As<Napi::Boolean>().Value();
      if (this->cudaGraph_) {
        bool hasCuda = false;
        if (jsOptions.Has("executionProviders") && jsOptions.Get("executionProviders").IsArray()) {
          for (const auto& entry : jsOptions.Get("executionProviders").As<Napi::Array>()) {
            Napi::Value item = entry.second.AsValue();
            hasCuda = hasCuda || (item.IsString() && item.As<Napi::String>().Utf8Value() == "cuda");
          }
        }
        ORT_NAPI_THROW_TYPEERROR_IF(!hasCuda, env,
                                    "sessionOptions.enableCudaGraph requires executionProviders to include 'cuda'.");
      }
      ParseSessionOptions(jsOptions, sessionOptions);
      this->session_.reset(new Ort::Session(OrtSingletonData::GetOrtObjects()->env,
#ifdef _WIN32
                                            reinterpret_cast<const wchar_t*>(value.Utf16Value().c_str()),
#else
                                            value.Utf8Value().c_str(),
#endif
                                            sessionOptions));

    } else {
      ORT_NAPI_THROW_TYPEERROR(
          env,
          "Invalid argument: expected (modelPath, options).");
    }

    // cache input/output names
    Ort::AllocatorWithDefaultOptions allocator;

    size_t count = session_->GetInputCount();
    inputNames_.reserve(count);
    for (size_t i = 0; i < count; i++) {
      auto input_name = session_->GetInputNameAllocated(i, allocator);
      inputNames_.emplace_back(input_name.get());
    }

    count = session_->GetOutputCount();
    outputNames_.reserve(count);
    for (size_t i = 0; i < count; i++) {
      auto output_name = session_->GetOutputNameAllocated(i, allocator);
      outputNames_.emplace_back(output_name.get());
    }

  } catch (Napi::Error const& e) {
    throw e;
  } catch (std::exception const& e) {
    ORT_NAPI_THROW_ERROR(env, e.what());
  }
  this->initialized_ = true;
  return env.Undefined();
}

Napi::Value InferenceSessionWrap::GetMetadata(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ORT_NAPI_THROW_ERROR_IF(!this->initialized_, env, "Session is not initialized.");
  ORT_NAPI_THROW_ERROR_IF(this->disposed_, env, "Session already disposed.");

  Napi::EscapableHandleScope scope(env);
  auto& names = info.Data() != nullptr ? inputNames_ : outputNames_;
  auto array = Napi::Array::New(env, names.size());
  for (uint32_t i = 0; i < names.size(); i++) {
    Napi::Object obj = Napi::Object::New(env);
    obj.Set("name", names[i]);
    array.Set(i, Napi::Value::From(env, obj));
  }
  return scope.Escape(array);
}

void InferenceSessionWrap::BindDeviceBuffers(const std::vector<Ort::Value>& inputs) {
  auto memoryInfo = Ort::MemoryInfo("Cuda", OrtArenaAllocator, 0, OrtMemTypeDefault);
  deviceAllocator_ = std::make_unique<Ort::Allocator>(*session_, memoryInfo);
  ioBinding_ = std::make_unique<Ort::IoBinding>(*session_);

  for (size_t i = 0; i < inputs.size(); i++) {
    auto info = inputs[i].GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();
    deviceInputs_.emplace_back(
        Ort::Value::CreateTensor(*deviceAllocator_, shape.data(), shape.size(), info.GetElementType()));
    CopyHostToCuda(deviceInputs_.back().GetTensorMutableRawData(), inputs[i].GetTensorRawData(),
                   info.GetElementCount() * TensorByteSize(info.GetElementType()));
    boundInputShapes_.push_back(shape);
    ioBinding_->BindInput(inputNames_[i].c_str(), deviceInputs_[i]);
  }

  // As formas de saida so sao conhecidas depois de rodar uma vez, entao a primeira
  // execucao liga as saidas ao device e deixa o ORT alocar; os enderecos resultantes
  // sao os que o grafo capturado vai usar.
  auto* cudaMemInfo = static_cast<const OrtMemoryInfo*>(memoryInfo);
  for (const auto& name : outputNames_) {
    ioBinding_->BindOutput(name.c_str(), cudaMemInfo);
  }

  session_->Run(OrtSingletonData::GetOrtObjects()->default_run_options, *ioBinding_);

  // Refixa as saidas nos buffers que acabaram de ser alocados, para que o endereco
  // pare de mudar a cada chamada — requisito do CUDA graph.
  deviceOutputs_ = ioBinding_->GetOutputValues();
  ioBinding_->ClearBoundOutputs();
  for (size_t i = 0; i < outputNames_.size(); i++) {
    ioBinding_->BindOutput(outputNames_[i].c_str(), deviceOutputs_[i]);
  }
}

Napi::Value InferenceSessionWrap::Run(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ORT_NAPI_THROW_ERROR_IF(!this->initialized_, env, "Session is not initialized.");
  ORT_NAPI_THROW_ERROR_IF(this->disposed_, env, "Session already disposed.");
  ORT_NAPI_THROW_TYPEERROR_IF(info.Length() < 2, env, "Expect argument: inputs(feed) and outputs(fetch).");
  ORT_NAPI_THROW_TYPEERROR_IF(!info[0].IsObject() || !info[1].IsObject(), env,
                              "Expect inputs(feed) and outputs(fetch) to be objects.");
  ORT_NAPI_THROW_TYPEERROR_IF(info.Length() > 2 && (!info[2].IsObject() || info[2].IsNull()), env,
                              "'runOptions' must be an object.");

  Napi::EscapableHandleScope scope(env);

  auto feed = info[0].As<Napi::Object>();
  auto fetch = info[1].As<Napi::Object>();

  std::vector<const char*> inputNames_cstr;
  std::vector<Ort::Value> inputValues;
  std::vector<const char*> outputNames_cstr;
  std::vector<Ort::Value> outputValues;
  size_t inputIndex = 0;
  size_t outputIndex = 0;
  Ort::MemoryInfo cpuMemoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  try {
    for (auto& name : inputNames_) {
      if (feed.Has(name)) {
        inputIndex++;
        inputNames_cstr.push_back(name.c_str());
        auto value = feed.Get(name);
        inputValues.push_back(NapiValueToOrtValue(env, value, cpuMemoryInfo));
      }
    }
    for (auto& name : outputNames_) {
      if (fetch.Has(name)) {
        outputIndex++;
        outputNames_cstr.push_back(name.c_str());
        auto value = fetch.Get(name);
        outputValues.emplace_back(value.IsNull() ? Ort::Value{nullptr} : NapiValueToOrtValue(env, value, cpuMemoryInfo));
      }
    }

    Ort::RunOptions runOptions{nullptr};
    if (info.Length() > 2) {
      runOptions = Ort::RunOptions{};
      ParseRunOptions(info[2].As<Napi::Object>(), runOptions);
    }

    if (cudaGraph_) {
      ORT_NAPI_THROW_ERROR_IF(inputIndex != inputNames_.size(), env,
                              "The cuda graph path needs every input in 'feeds'.");
      if (!ioBinding_) {
        BindDeviceBuffers(inputValues);
      }

      // O grafo capturado congela os enderecos, entao a forma nao pode mudar depois
      // da primeira chamada: use um modelo de shape fixo e preencha com padding.
      for (size_t i = 0; i < inputIndex; i++) {
        auto shape = inputValues[i].GetTensorTypeAndShapeInfo().GetShape();
        ORT_NAPI_THROW_ERROR_IF(shape != boundInputShapes_[i], env,
                                "Input shape changed after the cuda graph was captured.");
      }

      // Reescreve os buffers do device no lugar. Ligar tensores de CPU tambem e
      // aceito pelo ORT, mas devolve o resultado da chamada anterior sem erro.
      for (size_t i = 0; i < inputIndex; i++) {
        const auto info = inputValues[i].GetTensorTypeAndShapeInfo();
        CopyHostToCuda(deviceInputs_[i].GetTensorMutableRawData(), inputValues[i].GetTensorRawData(),
                       info.GetElementCount() * TensorByteSize(info.GetElementType()));
      }
      session_->Run(runOptions == nullptr ? OrtSingletonData::GetOrtObjects()->default_run_options : runOptions,
                    *ioBinding_);

      Napi::Object result = Napi::Object::New(env);
      Ort::AllocatorWithDefaultOptions cpuAllocator;

      std::vector<Ort::Value> hostOutputs;
      hostOutputs.reserve(outputNames_.size());
      for (auto& deviceOutput : deviceOutputs_) {
        auto shapeInfo = deviceOutput.GetTensorTypeAndShapeInfo();
        auto shape = shapeInfo.GetShape();
        hostOutputs.emplace_back(
            Ort::Value::CreateTensor(cpuAllocator, shape.data(), shape.size(), shapeInfo.GetElementType()));
      }
      for (size_t i = 0; i < deviceOutputs_.size(); i++) {
        const auto info = deviceOutputs_[i].GetTensorTypeAndShapeInfo();
        CopyCudaToHost(hostOutputs[i].GetTensorMutableRawData(), deviceOutputs_[i].GetTensorRawData(),
                       info.GetElementCount() * TensorByteSize(info.GetElementType()));
      }

      for (size_t i = 0; i < outputNames_.size(); i++) {
        result.Set(outputNames_[i], OrtValueToNapiValue(env, std::move(hostOutputs[i])));
      }
      return scope.Escape(result);
    }

    session_->Run(runOptions == nullptr ? OrtSingletonData::GetOrtObjects()->default_run_options : runOptions,
                  inputIndex == 0 ? nullptr : &inputNames_cstr[0], inputIndex == 0 ? nullptr : &inputValues[0],
                  inputIndex, outputIndex == 0 ? nullptr : &outputNames_cstr[0],
                  outputIndex == 0 ? nullptr : &outputValues[0], outputIndex);

    Napi::Object result = Napi::Object::New(env);

    for (size_t i = 0; i < outputIndex; i++) {
      result.Set(outputNames_cstr[i], OrtValueToNapiValue(env, std::move(outputValues[i])));
    }
    return scope.Escape(result);
  } catch (Napi::Error const& e) {
    throw e;
  } catch (std::exception const& e) {
    ORT_NAPI_THROW_ERROR(env, e.what());
  }
}

Napi::Value InferenceSessionWrap::Dispose(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ORT_NAPI_THROW_ERROR_IF(!this->initialized_, env, "Session is not initialized.");
  ORT_NAPI_THROW_ERROR_IF(this->disposed_, env, "Session already disposed.");

  // IoBinding and its tensors borrow session-owned allocator state. Release
  // them before the session so process shutdown is safe in both Node and Bun.
  deviceOutputs_.clear();
  deviceInputs_.clear();
  ioBinding_.reset();
  deviceAllocator_.reset();
  boundInputShapes_.clear();
  this->session_.reset(nullptr);

  this->disposed_ = true;
  return env.Undefined();
}

Napi::Value InferenceSessionWrap::EndProfiling(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ORT_NAPI_THROW_ERROR_IF(!this->initialized_, env, "Session is not initialized.");
  ORT_NAPI_THROW_ERROR_IF(this->disposed_, env, "Session already disposed.");

  Napi::EscapableHandleScope scope(env);

  Ort::AllocatorWithDefaultOptions allocator;

  auto filename = session_->EndProfilingAllocated(allocator);
  Napi::String filenameValue = Napi::String::From(env, filename.get());
  return scope.Escape(filenameValue);
}

Napi::Value InferenceSessionWrap::ListSupportedBackends(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::EscapableHandleScope scope(env);
  Napi::Array result = Napi::Array::New(env);

  auto createObject = [&env](const std::string& name, const bool bundled) -> Napi::Object {
    Napi::Object result = Napi::Object::New(env);
    result.Set("name", name);
    result.Set("bundled", bundled);
    return result;
  };

  result.Set(uint32_t(0), createObject("cpu", true));

  // GetAvailableProviders carrega a biblioteca do provider (176 MB no caso do
  // CUDA), entao a resposta fica em cache: sem isso toda sessao de CPU pagava
  // esse carregamento e ficava ~60% mais lenta.
  static const bool hasCuda = [] {
    for (const auto& provider : Ort::GetAvailableProviders()) {
      if (provider == "CUDAExecutionProvider") return true;
    }
    return false;
  }();

  if (hasCuda) result.Set(result.Length(), createObject("cuda", true));

  return scope.Escape(result);
}
