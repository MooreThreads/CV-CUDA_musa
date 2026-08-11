/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Stream.hpp"

#include <musa.h>
#include <musa_runtime_api.h>
#include <nvcv/util_musa/CheckError.hpp>

namespace nvcv::util {

CudaStream CudaStream::Create(bool nonBlocking, int deviceId)
{
    musaStream_t stream;
    int          flags   = nonBlocking ? musaStreamNonBlocking : musaStreamDefault;
    int          prevDev = -1;
    if (deviceId >= 0)
    {
        NVCV_CHECK_THROW(musaGetDevice(&prevDev));
        NVCV_CHECK_THROW(musaSetDevice(deviceId));
    }
    auto err = musaStreamCreateWithFlags(&stream, flags);
    if (prevDev >= 0)
        NVCV_CHECK_THROW(musaSetDevice(prevDev));
    NVCV_CHECK_THROW(err);
    return CudaStream(stream);
}

CudaStream CudaStream::CreateWithPriority(bool nonBlocking, int priority, int deviceId)
{
    musaStream_t stream;
    int          flags   = nonBlocking ? musaStreamNonBlocking : musaStreamDefault;
    int          prevDev = -1;
    if (deviceId >= 0)
    {
        NVCV_CHECK_THROW(musaGetDevice(&prevDev));
        NVCV_CHECK_THROW(musaSetDevice(deviceId));
    }
    auto err = musaStreamCreateWithPriority(&stream, flags, priority);
    if (prevDev >= 0)
        NVCV_CHECK_THROW(musaSetDevice(prevDev));
    NVCV_CHECK_THROW(err);
    return CudaStream(stream);
}

void CudaStream::DestroyHandle(musaStream_t stream)
{
    auto err = musaStreamDestroy(stream);
    if (err != musaSuccess && err != musaErrorMusartUnloading)
    {
        NVCV_CHECK_THROW(err);
    }
}

} // namespace nvcv::util
