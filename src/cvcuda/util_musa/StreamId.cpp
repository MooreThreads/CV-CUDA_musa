/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "StreamId.hpp"

#include <musa.h>
#include <musa_runtime.h>
#include <nvcv/Exception.hpp>

#include <sys/syscall.h>
#include <unistd.h>

using cuStreamGetId_t = MUresult(MUstream, unsigned long long *);

namespace {

inline int getTID()
{
    return syscall(SYS_gettid);
}

constexpr uint64_t MakeLegacyStreamId(int dev, int tid)
{
    return (uint64_t)dev << 32 | tid;
}

bool TryGetPseudoStreamId(MUstream stream, unsigned long long *id)
{
    if (stream == 0 || stream == MU_STREAM_LEGACY || stream == MU_STREAM_PER_THREAD)
    {
        int dev = -1;
        if (musaGetDevice(&dev) != musaSuccess)
            return false;
        *id = MakeLegacyStreamId(dev, stream == MU_STREAM_PER_THREAD ? getTID() : -1);
        return true;
    }
    return false;
}

} // namespace

#if MUSA_VERSION >= 12000

namespace {

MUresult cuStreamGetIdWithPseudoDefault(MUstream stream, unsigned long long *id)
{
    if (TryGetPseudoStreamId(stream, id))
        return MUSA_SUCCESS;
    return muStreamGetId(stream, id);
}

cuStreamGetId_t *_cuStreamGetId = cuStreamGetIdWithPseudoDefault;

bool _hasPreciseHint()
{
    return true;
}

} // namespace

#else

#    include <dlfcn.h>

namespace {

MUresult cuStreamGetIdFallback(MUstream stream, unsigned long long *id)
{
    // If the stream handle is a pseudohandle, use some special treatment....
    if (TryGetPseudoStreamId(stream, id))
    {
        return MUSA_SUCCESS;
    }
    else
    {
        // Otherwise just use the handle - it's not perfactly safe, but should do.
        *id = (uint64_t)stream;
        return MUSA_SUCCESS;
    }
}

cuStreamGetId_t *getRealStreamIdFunc()
{
    static cuStreamGetId_t *fn = []()
    {
        void *sym = nullptr;
        // If it fails, we'll just return nullptr.
        (void)cuGetProcAddress("cuStreamGetId", &sym, 12000, MU_GET_PROC_ADDRESS_DEFAULT);
        return (cuStreamGetId_t *)sym;
    }();
    return fn;
}

bool _hasPreciseHint()
{
    static bool ret = getRealStreamIdFunc() != nullptr;
    return ret;
}

MUresult cuStreamGetIdBootstrap(MUstream stream, unsigned long long *id);

cuStreamGetId_t *_cuStreamGetId = cuStreamGetIdBootstrap;

MUresult cuStreamGetIdBootstrap(MUstream stream, unsigned long long *id)
{
    cuStreamGetId_t *realFunc = getRealStreamIdFunc();
    if (realFunc)
        _cuStreamGetId = realFunc;
    else
        _cuStreamGetId = cuStreamGetIdFallback;

    return _cuStreamGetId(stream, id);
}

} // namespace

#endif

namespace nvcv::util {

bool IsCudaStreamIdHintUnambiguous()
{
    return _hasPreciseHint();
}

uint64_t GetCudaStreamIdHint(MUstream stream)
{
    static auto initResult = muInit(0);
    (void)initResult;
    unsigned long long id;
    MUresult           err = _cuStreamGetId(stream, &id);
    if (err != MUSA_SUCCESS)
    {
        switch (err)
        {
        case MUSA_ERROR_DEINITIALIZED:
            // This is most likely to happen during process teardown, so likely in a destructor
            // - we don't want to throw there and the stream equality is immaterial anyway at this point.
            return -1;
        case MUSA_ERROR_INVALID_VALUE:
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid stream handle");
        default:
        {
            const char *msg  = "";
            const char *name = "Unknown error";
            (void)muGetErrorString(err, &msg);
            (void)muGetErrorName(err, &name);
            throw nvcv::Exception(nvcv::Status::ERROR_INTERNAL, "MUSA error %s %i %s", name, err, msg);
        }
        }
    }
    return id;
}

} // namespace nvcv::util
