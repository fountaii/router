// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <napi.h>

#include "inference_session_wrap.h"

// Native CPU runtime of the BitNet decision model (bitnet_wrap.cc).
void InitBitnet(Napi::Env env, Napi::Object exports);

Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
  InferenceSessionWrap::Init(env, exports);
  InitBitnet(env, exports);
  return exports;
}

NODE_API_MODULE(onnxruntime, InitAll)
