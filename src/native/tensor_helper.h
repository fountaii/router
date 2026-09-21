// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <napi.h>
#include "onnxruntime_cxx_api.h"

Ort::Value NapiValueToOrtValue(Napi::Env env, Napi::Value value, OrtMemoryInfo* cpu_memory_info);
Napi::Value OrtValueToNapiValue(Napi::Env env, Ort::Value&& value);
