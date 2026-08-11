// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// musify: MUSA test-only compatibility include — routes CUDA runtime API names used by
// CV-CUDA's checked-in C++ test sources to real MUSA runtime APIs when USE_MUSA builds
// the required in-tree test artifacts. This is not used by production sources.
#include <musa_runtime.h>

#define cudaError_t musaError_t
#define cudaSuccess musaSuccess
#define cudaStream_t musaStream_t
#define cudaStreamNonBlocking musaStreamNonBlocking
#define cudaStreamPerThread musaStreamPerThread
#define cudaStreamCreateWithFlags musaStreamCreateWithFlags
#define cudaMemsetAsync musaMemsetAsync
#define cudaEvent_t musaEvent_t
#define cudaMemcpyKind musaMemcpyKind
#define cudaErrorMemoryAllocation musaErrorMemoryAllocation
#define cudaErrorNotReady musaErrorNotReady
#define cudaErrorInvalidValue musaErrorInvalidValue
#define cudaErrorTextureFetchFailed musaErrorTextureFetchFailed
#define cudaErrorCudartUnloading musaErrorMusartUnloading
#define cudaMemcpyHostToDevice musaMemcpyHostToDevice
#define cudaMemcpyDeviceToHost musaMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice musaMemcpyDeviceToDevice
#define cudaStreamCreate musaStreamCreate
#define cudaStreamSynchronize musaStreamSynchronize
#define cudaStreamDestroy musaStreamDestroy
#define cudaStreamWaitEvent musaStreamWaitEvent
#define cudaEventDisableTiming musaEventDisableTiming
#define cudaEventCreateWithFlags musaEventCreateWithFlags
#define cudaEventDestroy musaEventDestroy
#define cudaEventRecord musaEventRecord
#define cudaEventQuery musaEventQuery
#define cudaEventSynchronize musaEventSynchronize
#define cudaMemcpy musaMemcpy
#define cudaMemcpy2D musaMemcpy2D
#define cudaMemcpyAsync musaMemcpyAsync
#define cudaMemcpy2DAsync musaMemcpy2DAsync
#define cudaMemset musaMemset
#define cudaMemset2D musaMemset2D
#define cudaMalloc musaMalloc
#define cudaMallocManaged musaMallocManaged
#define cudaFree musaFree
#define cudaGetErrorName musaGetErrorName
#define cudaGetErrorString musaGetErrorString
#define cudaGetLastError musaGetLastError
#define cudaDeviceSynchronize musaDeviceSynchronize
#define cudaGetDeviceCount musaGetDeviceCount

#define cudaMem musaMem
