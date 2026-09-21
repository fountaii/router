// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_cxx_api.h"

#include <memory>
#include <napi.h>

// class InferenceSessionWrap is a N-API object wrapper for native InferenceSession.
class InferenceSessionWrap : public Napi::ObjectWrap<InferenceSessionWrap> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);

  InferenceSessionWrap(const Napi::CallbackInfo& info);
  ~InferenceSessionWrap();

 private:
  /**
   * [sync] initialize ONNX Runtime once.
   *
   * This function must be called before any other functions.
   *
   * @param arg0 a number specifying the log level.
   *
   * @returns undefined
   */
  static Napi::Value InitOrtOnce(const Napi::CallbackInfo& info);

  /**
   * [sync] list supported backend list
   * @returns array with objects { "name": "cpu", requirementsInstalled: true }
   */
  static Napi::Value ListSupportedBackends(const Napi::CallbackInfo& info);

  /**
   * [sync] create the session.
   * @param arg0 a string (file path)
   * @returns nothing
   * @throw error if status code != 0
   */
  Napi::Value LoadModel(const Napi::CallbackInfo& info);

  // following functions have to be called after model is loaded.

  /**
   * [sync] get metadata of the model's inputs or outputs.
   * @param nothing
   * @returns an array of objects with keys: name
   * @throw nothing
   */
  Napi::Value GetMetadata(const Napi::CallbackInfo& info);

  /**
   * [sync] run the model.
   * @param arg0 input object: all keys must present, value is object
   * @param arg1 output object: at least one key must present, value can be null.
   * @returns an object that every output specified will present and value must be object
   * @throw error if status code != 0
   */
  Napi::Value Run(const Napi::CallbackInfo& info);

  /**
   * [sync] dispose the session.
   * @param nothing
   * @returns nothing
   * @throw nothing
   */
  Napi::Value Dispose(const Napi::CallbackInfo& info);

  /**
   * [sync] end the profiling.
   * @param nothing
   * @returns nothing
   * @throw nothing
   */
  Napi::Value EndProfiling(const Napi::CallbackInfo& info);

  // private members

  // session objects
  bool initialized_;
  bool disposed_;
  std::unique_ptr<Ort::Session> session_;

  // input/output metadata
  std::vector<std::string> inputNames_;
  std::vector<std::string> outputNames_;

  // Caminho de CUDA graph. O grafo capturado fixa os enderecos dos buffers, entao
  // os tensores do device sao alocados uma vez e reescritos no lugar por
  // cudaMemcpy. Ligar entradas de CPU parece funcionar mas devolve o
  // resultado da chamada anterior, sem erro — por isso este caminho e obrigatorio.
  bool cudaGraph_ = false;
  std::unique_ptr<Ort::Allocator> deviceAllocator_;
  std::unique_ptr<Ort::IoBinding> ioBinding_;
  std::vector<Ort::Value> deviceInputs_;
  std::vector<Ort::Value> deviceOutputs_;
  std::vector<std::vector<int64_t>> boundInputShapes_;

  // Prepara (uma vez) os buffers do device com as formas do primeiro run.
  void BindDeviceBuffers(const std::vector<Ort::Value>& inputs);

};
