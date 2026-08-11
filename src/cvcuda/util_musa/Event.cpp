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

#include "Event.hpp"

#include <musa_runtime_api.h>
#include <nvcv/util_musa/CheckError.hpp>

namespace nvcv::util {

CudaEvent CudaEvent::Create(int deviceId)
{
    return CreateWithFlags(musaEventDisableTiming, deviceId);
}

CudaEvent CudaEvent::CreateWithFlags(unsigned flags, int deviceId)
{
    musaEvent_t event;
    int         prevDev = -1;
    if (deviceId >= 0)
    {
        NVCV_CHECK_THROW(musaGetDevice(&prevDev));
        NVCV_CHECK_THROW(musaSetDevice(deviceId));
    }
    auto err = musaEventCreateWithFlags(&event, flags);
    if (prevDev >= 0)
        NVCV_CHECK_THROW(musaSetDevice(prevDev));
    NVCV_CHECK_THROW(err);
    return CudaEvent(event);
}

void CudaEvent::DestroyHandle(musaEvent_t event)
{
    auto err = musaEventDestroy(event);
    if (err != musaSuccess && err != musaErrorMusartUnloading)
    {
        NVCV_CHECK_THROW(err);
    }
}

} // namespace nvcv::util
