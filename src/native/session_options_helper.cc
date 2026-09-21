// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_cxx_api.h"
#include <napi.h>

#include <cmath>
#include <unordered_map>

#include "common.h"
#include "session_options_helper.h"
const std::unordered_map<std::string, GraphOptimizationLevel> GRAPH_OPT_LEVEL_NAME_TO_ID_MAP = {
    {"disabled", ORT_DISABLE_ALL},
    {"basic", ORT_ENABLE_BASIC},
    {"extended", ORT_ENABLE_EXTENDED},
    {"all", ORT_ENABLE_ALL}};

static void ReleaseCudaOptions(OrtCUDAProviderOptionsV2* options) {
  Ort::GetApi().ReleaseCUDAProviderOptions(options);
}

const std::unordered_map<std::string, ExecutionMode> EXECUTION_MODE_NAME_TO_ID_MAP = {{"sequential", ORT_SEQUENTIAL},
                                                                                      {"parallel", ORT_PARALLEL}};

void ParseExecutionProviders(const Napi::Array epList, Ort::SessionOptions& sessionOptions, bool cudaGraph) {
  for (uint32_t i = 0; i < epList.Length(); i++) {
    auto value = epList.Get(i);
    ORT_NAPI_THROW_TYPEERROR_IF(!value.IsString(), epList.Env(),
                              "sessionOptions.executionProviders entries must be strings.");
    auto name = value.As<Napi::String>().Utf8Value();

    // cpu e o fallback implicito do ORT e nao precisa ser anexado
    if (name == "cpu") continue;

    if (name == "cuda") {
      // as contrib ops int8 do modelo quantizado sao so de CPU; o CUDA so ajuda
      // com um modelo float, senao o grafo parte e fica pior que a CPU
      if (!cudaGraph) {
        OrtCUDAProviderOptions cudaOptions;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
        continue;
      }

      // enable_cuda_graph so existe na variante V2, que recebe as opcoes por string
      OrtCUDAProviderOptionsV2* raw = nullptr;
      Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&raw));
      std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(&ReleaseCudaOptions)> cudaOptions(raw, ReleaseCudaOptions);

      const char* keys[] = {"enable_cuda_graph"};
      const char* values[] = {"1"};
      Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(cudaOptions.get(), keys, values, 1));
      sessionOptions.AppendExecutionProvider_CUDA_V2(*cudaOptions);
      continue;
    }

    ORT_NAPI_THROW_TYPEERROR_IF(true, epList.Env(),
                              "Only the cpu and cuda execution providers are supported by this binding.");
  }
}

void ParseSessionOptions(const Napi::Object options, Ort::SessionOptions& sessionOptions) {
  // Flat ORT configuration keys allow measured thread-pool tuning.
  if (options.Has("configEntries")) {
    auto value = options.Get("configEntries");
    ORT_NAPI_THROW_TYPEERROR_IF(!value.IsObject() || value.IsNull() || value.IsArray(), options.Env(),
                              "sessionOptions.configEntries must be an object of strings.");
    for (const auto& entry : value.As<Napi::Object>()) {
      auto key = entry.first.As<Napi::String>().Utf8Value();
      Napi::Value setting = entry.second;
      ORT_NAPI_THROW_TYPEERROR_IF(!setting.IsString(), options.Env(),
                                "sessionOptions.configEntries values must be strings.");
      sessionOptions.AddConfigEntry(key.c_str(), setting.As<Napi::String>().Utf8Value().c_str());
    }
  }
  // Execution provider
  const bool cudaGraph = options.Has("enableCudaGraph") && options.Get("enableCudaGraph").ToBoolean();

  if (options.Has("executionProviders")) {
    auto epsValue = options.Get("executionProviders");
    ORT_NAPI_THROW_TYPEERROR_IF(!epsValue.IsArray(), options.Env(),
                                "Invalid argument: sessionOptions.executionProviders must be an array.");
    ParseExecutionProviders(epsValue.As<Napi::Array>(), sessionOptions, cudaGraph);
  }

  // Intra threads number
  if (options.Has("intraOpNumThreads")) {
    auto numValue = options.Get("intraOpNumThreads");
    ORT_NAPI_THROW_TYPEERROR_IF(!numValue.IsNumber(), options.Env(),
                                "Invalid argument: sessionOptions.intraOpNumThreads must be a number.");
    double num = numValue.As<Napi::Number>().DoubleValue();
    ORT_NAPI_THROW_RANGEERROR_IF(std::floor(num) != num || num < 0 || num > 4294967295, options.Env(),
                                 "'intraOpNumThreads' is invalid: ", num);
    sessionOptions.SetIntraOpNumThreads(static_cast<int>(num));
  }

  // Inter threads number
  if (options.Has("interOpNumThreads")) {
    auto numValue = options.Get("interOpNumThreads");
    ORT_NAPI_THROW_TYPEERROR_IF(!numValue.IsNumber(), options.Env(),
                                "Invalid argument: sessionOptions.interOpNumThreads must be a number.");
    double num = numValue.As<Napi::Number>().DoubleValue();
    ORT_NAPI_THROW_RANGEERROR_IF(std::floor(num) != num || num < 0 || num > 4294967295, options.Env(),
                                 "'interOpNumThreads' is invalid: ", num);
    sessionOptions.SetInterOpNumThreads(static_cast<int>(num));
  }

  // Optimization level
  if (options.Has("graphOptimizationLevel")) {
    auto optLevelValue = options.Get("graphOptimizationLevel");
    ORT_NAPI_THROW_TYPEERROR_IF(!optLevelValue.IsString(), options.Env(),
                                "Invalid argument: sessionOptions.graphOptimizationLevel must be a string.");
    auto optLevelString = optLevelValue.As<Napi::String>().Utf8Value();
    auto v = GRAPH_OPT_LEVEL_NAME_TO_ID_MAP.find(optLevelString);
    ORT_NAPI_THROW_TYPEERROR_IF(v == GRAPH_OPT_LEVEL_NAME_TO_ID_MAP.end(), options.Env(),
                                "'graphOptimizationLevel' is not supported: ", optLevelString);
    sessionOptions.SetGraphOptimizationLevel(v->second);
  }

  // CPU memory arena
  if (options.Has("enableCpuMemArena")) {
    auto enableCpuMemArenaValue = options.Get("enableCpuMemArena");
    ORT_NAPI_THROW_TYPEERROR_IF(!enableCpuMemArenaValue.IsBoolean(), options.Env(),
                                "Invalid argument: sessionOptions.enableCpuMemArena must be a boolean value.");
    if (enableCpuMemArenaValue.As<Napi::Boolean>().Value()) {
      sessionOptions.EnableCpuMemArena();
    } else {
      sessionOptions.DisableCpuMemArena();
    }
  }

  // memory pattern
  if (options.Has("enableMemPattern")) {
    auto enableMemPatternValue = options.Get("enableMemPattern");
    ORT_NAPI_THROW_TYPEERROR_IF(!enableMemPatternValue.IsBoolean(), options.Env(),
                                "Invalid argument: sessionOptions.enableMemPattern must be a boolean value.");
    if (enableMemPatternValue.As<Napi::Boolean>().Value()) {
      sessionOptions.EnableMemPattern();
    } else {
      sessionOptions.DisableMemPattern();
    }
  }

  // execution mode
  if (options.Has("executionMode")) {
    auto executionModeValue = options.Get("executionMode");
    ORT_NAPI_THROW_TYPEERROR_IF(!executionModeValue.IsString(), options.Env(),
                                "Invalid argument: sessionOptions.executionMode must be a string.");
    auto executionModeString = executionModeValue.As<Napi::String>().Utf8Value();
    auto v = EXECUTION_MODE_NAME_TO_ID_MAP.find(executionModeString);
    ORT_NAPI_THROW_TYPEERROR_IF(v == EXECUTION_MODE_NAME_TO_ID_MAP.end(), options.Env(),
                                "'executionMode' is not supported: ", executionModeString);
    sessionOptions.SetExecutionMode(v->second);
  }

  // Log severity level
  if (options.Has("logSeverityLevel")) {
    auto logLevelValue = options.Get("logSeverityLevel");
    ORT_NAPI_THROW_TYPEERROR_IF(!logLevelValue.IsNumber(), options.Env(),
                                "Invalid argument: sessionOptions.logSeverityLevel must be a number.");
    double logLevelNumber = logLevelValue.As<Napi::Number>().DoubleValue();
    ORT_NAPI_THROW_RANGEERROR_IF(
        std::floor(logLevelNumber) != logLevelNumber || logLevelNumber < 0 || logLevelNumber > 4, options.Env(),
        "Invalid argument: sessionOptions.logSeverityLevel must be one of the following: 0, 1, 2, 3, 4.");

    sessionOptions.SetLogSeverityLevel(static_cast<int>(logLevelNumber));
  }

  // Profiling
  if (options.Has("enableProfiling")) {
    auto enableProfilingValue = options.Get("enableProfiling");
    ORT_NAPI_THROW_TYPEERROR_IF(!enableProfilingValue.IsBoolean(), options.Env(),
                                "Invalid argument: sessionOptions.enableProfiling must be a boolean value.");

    if (enableProfilingValue.As<Napi::Boolean>().Value()) {
      ORT_NAPI_THROW_TYPEERROR_IF(!options.Has("profileFilePrefix"), options.Env(),
                                  "Invalid argument: sessionOptions.profileFilePrefix is required"
                                  " when sessionOptions.enableProfiling is set to true.");
      auto profileFilePrefixValue = options.Get("profileFilePrefix");
      ORT_NAPI_THROW_TYPEERROR_IF(!profileFilePrefixValue.IsString(), options.Env(),
                                  "Invalid argument: sessionOptions.profileFilePrefix must be a string."
                                  " when sessionOptions.enableProfiling is set to true.");
#ifdef _WIN32
      auto str = profileFilePrefixValue.As<Napi::String>().Utf16Value();
      std::basic_string<ORTCHAR_T> profileFilePrefix = std::wstring{str.begin(), str.end()};
#else
      std::basic_string<ORTCHAR_T> profileFilePrefix = profileFilePrefixValue.As<Napi::String>().Utf8Value();
#endif
      sessionOptions.EnableProfiling(profileFilePrefix.c_str());
    } else {
      sessionOptions.DisableProfiling();
    }
  }

}
