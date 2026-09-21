// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_transfer.h"

#include <stdexcept>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace {
using CudaMemcpyHtoD = int(__cdecl*)(uintptr_t, const void*, size_t);
using CudaMemcpyDtoH = int(__cdecl*)(void*, uintptr_t, size_t);

struct CudaDriver {
  CudaMemcpyHtoD hostToDevice;
  CudaMemcpyDtoH deviceToHost;
};

const CudaDriver& GetCudaDriver() {
  static const CudaDriver driver = [] {
    // The driver API is supplied by Windows' NVIDIA driver. Unlike cudart, it
    // is not versioned and need not be shipped with the CUDA EP dependencies.
    HMODULE module = GetModuleHandleW(L"nvcuda.dll");
    if (!module) module = LoadLibraryW(L"nvcuda.dll");
    if (!module) throw std::runtime_error("CUDA graph requires the NVIDIA driver (nvcuda.dll).");
    auto hostToDevice = reinterpret_cast<CudaMemcpyHtoD>(GetProcAddress(module, "cuMemcpyHtoD_v2"));
    auto deviceToHost = reinterpret_cast<CudaMemcpyDtoH>(GetProcAddress(module, "cuMemcpyDtoH_v2"));
    if (!hostToDevice || !deviceToHost) throw std::runtime_error("NVIDIA driver does not export CUDA copy functions.");
    return CudaDriver{hostToDevice, deviceToHost};
  }();
  return driver;
}

void Check(int status) {
  if (status != 0) throw std::runtime_error("CUDA driver copy failed with CUDA error " + std::to_string(status) + ".");
}

void CopyHostToDevice(void* destination, const void* source, size_t bytes) {
  if (bytes == 0) return;
  Check(GetCudaDriver().hostToDevice(reinterpret_cast<uintptr_t>(destination), source, bytes));
}

void CopyDeviceToHost(void* destination, const void* source, size_t bytes) {
  if (bytes == 0) return;
  Check(GetCudaDriver().deviceToHost(destination, reinterpret_cast<uintptr_t>(source), bytes));
}
}  // namespace

void CopyHostToCuda(void* destination, const void* source, size_t bytes) {
  CopyHostToDevice(destination, source, bytes);
}

void CopyCudaToHost(void* destination, const void* source, size_t bytes) {
  CopyDeviceToHost(destination, source, bytes);
}
#else
void CopyHostToCuda(void*, const void*, size_t) {
  throw std::runtime_error("CUDA graph is only supported by this binding on Windows.");
}

void CopyCudaToHost(void*, const void*, size_t) {
  throw std::runtime_error("CUDA graph is only supported by this binding on Windows.");
}
#endif
