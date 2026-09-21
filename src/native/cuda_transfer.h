// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>

// Copies between host memory and CUDA device memory without linking the addon to
// cudart. The CUDA EP already loads cudart; resolving it at runtime keeps the
// CPU-only build free of a CUDA SDK dependency.
void CopyHostToCuda(void* destination, const void* source, size_t bytes);
void CopyCudaToHost(void* destination, const void* source, size_t bytes);
