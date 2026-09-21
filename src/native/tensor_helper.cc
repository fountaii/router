// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <cmath>
#include <cstring>
#include <limits>
#include "common.h"
#include "ort_instance_data.h"
#include "tensor_helper.h"

Ort::Value NapiValueToOrtValue(Napi::Env env, Napi::Value value, OrtMemoryInfo* memory) {
  ORT_NAPI_THROW_TYPEERROR_IF(!value.IsObject() || value.IsNull(), env, "Tensor must be an object.");
  auto tensor = value.As<Napi::Object>();
  auto location = tensor.Get("location");
  ORT_NAPI_THROW_TYPEERROR_IF(!location.IsString() || location.As<Napi::String>().Utf8Value() != "cpu",
                            env, "Tensor.location must be 'cpu'.");
  auto type = tensor.Get("type");
  ORT_NAPI_THROW_TYPEERROR_IF(!type.IsString(), env, "Tensor.type must be a string.");
  const auto name = type.As<Napi::String>().Utf8Value();
  ORT_NAPI_THROW_TYPEERROR_IF(name != "float32" && name != "int64", env, "Unsupported tensor type: ", name);
  const bool isFloat = name == "float32";
  auto data = tensor.Get("data");
  ORT_NAPI_THROW_TYPEERROR_IF(!data.IsTypedArray(), env, "Tensor.data must be a typed array.");
  auto array = data.As<Napi::TypedArray>();
  ORT_NAPI_THROW_TYPEERROR_IF(array.TypedArrayType() != (isFloat ? napi_float32_array : napi_bigint64_array),
                            env, "Tensor.data does not match Tensor.type.");
  auto shape = tensor.Get("dims");
  ORT_NAPI_THROW_TYPEERROR_IF(!shape.IsArray(), env, "Tensor.dims must be an array.");
  auto dimensions = shape.As<Napi::Array>();
  std::vector<int64_t> dims;
  size_t count = 1;
  for (uint32_t i = 0; i < dimensions.Length(); ++i) {
    auto dimension = dimensions.Get(i);
    ORT_NAPI_THROW_TYPEERROR_IF(!dimension.IsNumber(), env, "Tensor dimension must be a number.");
    double number = dimension.As<Napi::Number>().DoubleValue();
    ORT_NAPI_THROW_RANGEERROR_IF(!std::isfinite(number) || std::floor(number) != number || number < 0 || number > 4294967295.,
                                env, "Invalid tensor dimension.");
    auto dim = static_cast<size_t>(number);
    ORT_NAPI_THROW_RANGEERROR_IF(dim != 0 && count > std::numeric_limits<size_t>::max() / dim,
                                env, "Tensor size overflow.");
    count *= dim;
    dims.push_back(static_cast<int64_t>(dim));
  }
  ORT_NAPI_THROW_RANGEERROR_IF(count != array.ElementLength(), env, "Tensor shape does not match data length.");
  auto* bytes = static_cast<char*>(array.ArrayBuffer().Data());
  return Ort::Value::CreateTensor(memory, bytes ? bytes + array.ByteOffset() : nullptr, array.ByteLength(),
                                  dims.data(), dims.size(), isFloat ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT : ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
}

Napi::Value OrtValueToNapiValue(Napi::Env env, Ort::Value&& value) {
  Napi::EscapableHandleScope scope(env);
  ORT_NAPI_THROW_TYPEERROR_IF(!value.IsTensor(), env, "Only tensor outputs are supported.");
  const auto info = value.GetTensorTypeAndShapeInfo();
  const auto type = info.GetElementType();
  ORT_NAPI_THROW_TYPEERROR_IF(type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                            env, "Only float32 and int64 outputs are supported.");
  const bool isFloat = type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  const auto shape = info.GetShape();
  auto dims = Napi::Array::New(env, shape.size());
  for (uint32_t i = 0; i < shape.size(); ++i) dims.Set(i, Napi::Number::New(env, static_cast<double>(shape[i])));
  const auto count = info.GetElementCount();
  const auto bytes = count * (isFloat ? sizeof(float) : sizeof(int64_t));
  // JS owns this copy after Ort::Value and the session are released.
  auto buffer = Napi::ArrayBuffer::New(env, bytes);
  if (bytes) std::memcpy(buffer.Data(), value.GetTensorRawData(), bytes);
  napi_value data;
  NAPI_THROW_IF_FAILED(env, napi_create_typedarray(env, isFloat ? napi_float32_array : napi_bigint64_array,
                                                count, buffer, 0, &data), Napi::Value);
  return scope.Escape(OrtInstanceData::TensorConstructor(env).New({
      Napi::String::New(env, isFloat ? "float32" : "int64"), Napi::Value(env, data), dims}));
}
