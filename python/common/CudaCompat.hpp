/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVCV_PYTHON_CUDA_COMPAT_HPP
#define NVCV_PYTHON_CUDA_COMPAT_HPP

#ifdef NVCV_USE_MUSA
#    include <musa_runtime.h>
#    include <cvcuda/cuda_tools_musa/TypeTraits.hpp>

using cudaError_t           = musaError_t;
using cudaStream_t          = musaStream_t;
using cudaEvent_t           = musaEvent_t;
using cudaPointerAttributes = musaPointerAttributes;
using cudaStreamCallback_t  = musaStreamCallback_t;

#    define cudaSuccess musaSuccess
#    define cudaStreamNonBlocking musaStreamNonBlocking
#    define cudaEventDisableTiming musaEventDisableTiming
#    define cudaMemoryTypeUnregistered musaMemoryTypeUnregistered
#    define cudaMemcpyHostToDevice musaMemcpyHostToDevice
#    define cudaMemcpyDeviceToHost musaMemcpyDeviceToHost
#    define cudaErrorNotReady musaErrorNotReady
#    define cudaErrorCudartUnloading musaErrorMusartUnloading

inline const char *cudaGetErrorName(cudaError_t err)
{
    return musaGetErrorName(err);
}

inline const char *cudaGetErrorString(cudaError_t err)
{
    return musaGetErrorString(err);
}

inline cudaError_t cudaGetLastError()
{
    return musaGetLastError();
}

inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t *stream, unsigned int flags)
{
    return musaStreamCreateWithFlags(stream, flags);
}

inline cudaError_t cudaStreamGetFlags(cudaStream_t stream, unsigned int *flags)
{
    return musaStreamGetFlags(stream, flags);
}

inline cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
    return musaStreamSynchronize(stream);
}

inline cudaError_t cudaStreamDestroy(cudaStream_t stream)
{
    return musaStreamDestroy(stream);
}

inline cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags = 0)
{
    return musaStreamWaitEvent(stream, event, flags);
}

inline cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void *userData,
                                         unsigned int flags)
{
    return musaStreamAddCallback(stream, callback, userData, flags);
}

inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
{
    return musaEventCreateWithFlags(event, flags);
}

inline cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0)
{
    return musaEventRecord(event, stream);
}

inline cudaError_t cudaEventDestroy(cudaEvent_t event)
{
    return musaEventDestroy(event);
}

inline cudaError_t cudaEventQuery(cudaEvent_t event)
{
    return musaEventQuery(event);
}

inline cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
    return musaEventSynchronize(event);
}

inline cudaError_t cudaPointerGetAttributes(cudaPointerAttributes *attributes, const void *ptr)
{
    return musaPointerGetAttributes(attributes, ptr);
}

inline cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height,
                                musaMemcpyKind kind)
{
    return musaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
}

inline cudaError_t cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height)
{
    return musaMemset2D(devPtr, pitch, value, width, height);
}

inline cudaError_t cudaMemGetInfo(size_t *free, size_t *total)
{
    return musaMemGetInfo(free, total);
}

inline cudaError_t cudaGetDevice(int *device)
{
    return musaGetDevice(device);
}

namespace nvcv {
namespace cuda = musa;
}

#else
#    include <cuda_runtime.h>
#    include <cvcuda/cuda_tools/TypeTraits.hpp>
#endif

#endif // NVCV_PYTHON_CUDA_COMPAT_HPP
