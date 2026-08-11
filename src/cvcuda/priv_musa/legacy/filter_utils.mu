/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../Assert.h"

#include <cvcuda/cuda_tools_musa/DropCast.hpp>     // for DropCast, etc.
#include <cvcuda/cuda_tools_musa/MathOps.hpp>      // for math operators
#include <cvcuda/cuda_tools_musa/MathWrappers.hpp> // for sqrt, etc.
#include <cvcuda/cuda_tools_musa/SaturateCast.hpp> // for SaturateCast, etc.
#include <cvcuda/cuda_tools_musa/TensorWrap.hpp>   // for TensorWrap, etc.

namespace nvcv::legacy::musa_op {

__global__ void computeMeanKernel(float *kernel_ptr, int k_size)
{
    float kernelVal = 1.0 / k_size;
    int   tid       = threadIdx.x;
    if (tid < k_size)
    {
        kernel_ptr[tid] = kernelVal;
    }
}

template<typename COORD_TYPE>
__device__ __forceinline__ float computeSingleGaussianValue(COORD_TYPE coord, int2 half, double2 sigma)
{
    float sx = 2.f * sigma.x * sigma.x;
    float sy = 2.f * sigma.y * sigma.y;
    float s  = 2.f * sigma.x * sigma.y * M_PI;

    float sum = 0.f;

    for (int y = -half.y; y <= half.y; ++y)
    {
        for (int x = -half.x; x <= half.x; ++x)
        {
            sum += musa::exp(-((x * x) / sx + (y * y) / sy)) / s;
        }
    }

    int x = coord.x - half.x;
    int y = coord.y - half.y;

    return musa::exp(-((x * x) / sx + (y * y) / sy)) / (s * sum);
}

__global__ void computeGaussianKernel(float *kernel, Size2D kernelSize, double2 sigma)
{
    int2 coord{static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x),
               static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y)};

    if (coord.x >= kernelSize.w || coord.y >= kernelSize.h)
    {
        return;
    }

    int2 half{kernelSize.w / 2, kernelSize.h / 2};

    kernel[coord.y * kernelSize.w + coord.x] = computeSingleGaussianValue(coord, half, sigma);
}

__global__ void computeMeanKernelVarShape(musa::Tensor3DWrap<float, int32_t> kernel,
                                          musa::Tensor1DWrap<int2, int32_t>  kernelSizeArr,
                                          musa::Tensor1DWrap<int2, int32_t>  kernelAnchorArr)
{
    int3 coord{static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x),
               static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y),
               static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z)};

    int2 kernelSize = kernelSizeArr[coord.z];

    if (coord.x >= kernelSize.x || coord.y >= kernelSize.y)
    {
        return;
    }

    bool kernelAnchorUpdated = false;
    int2 kernelAnchor        = kernelAnchorArr[coord.z];

    if (kernelAnchor.x < 0)
    {
        kernelAnchor.x      = kernelSize.x / 2;
        kernelAnchorUpdated = true;
    }

    if (kernelAnchor.y < 0)
    {
        kernelAnchor.y      = kernelSize.y / 2;
        kernelAnchorUpdated = true;
    }

    if (kernelAnchorUpdated)
    {
        kernelAnchorArr[coord.z] = kernelAnchor;
    }

    kernel[coord] = 1.f / (kernelSize.x * kernelSize.y);
}

__global__ void computeGaussianKernelVarShape(musa::Tensor3DWrap<float, int32_t> kernel, int dataKernelSize,
                                              Size2D maxKernelSize, musa::Tensor1DWrap<int2, int32_t> kernelSizeArr,
                                              musa::Tensor1DWrap<double2, int32_t> sigmaArr)
{
    int3 coord{static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x),
               static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y),
               static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z)};

    int2 kernelSize = kernelSizeArr[coord.z];

    double2 sigma = sigmaArr[coord.z];

    if (sigma.y <= 0)
        sigma.y = sigma.x;

    // automatic detection of kernel size from sigma
    if (kernelSize.x <= 0 && sigma.x > 0)
        kernelSize.x = musa::round<int>(sigma.x * dataKernelSize * 2 + 1) | 1;
    if (kernelSize.y <= 0 && sigma.y > 0)
        kernelSize.y = musa::round<int>(sigma.y * dataKernelSize * 2 + 1) | 1;

    NVCV_CUDA_ASSERT(kernelSize.x > 0 && (kernelSize.x % 2 == 1) && kernelSize.x <= maxKernelSize.w,
                     "E Wrong kernelSize.x = %d, expected > 0, odd and <= %d\n", kernelSize.x, maxKernelSize.w);
    NVCV_CUDA_ASSERT(kernelSize.y > 0 && (kernelSize.y % 2 == 1) && kernelSize.y <= maxKernelSize.h,
                     "E Wrong kernelSize.y = %d, expected > 0, odd and <= %d\n", kernelSize.y, maxKernelSize.h);

    if (coord.x >= kernelSize.x || coord.y >= kernelSize.y)
    {
        return;
    }

    int2 half{kernelSize.x / 2, kernelSize.y / 2};

    sigma.x = musa::max(sigma.x, 0.0);
    sigma.y = musa::max(sigma.y, 0.0);

    kernel[coord] = computeSingleGaussianValue(coord, half, sigma);
}

__global__ void computeMeanKernelVarShape(musa::Tensor3DWrap<float, int32_t> kernel,
                                          musa::Tensor1DWrap<int, int32_t>   blockSizeArr)
{
    int3 coord{static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x),
               static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y),
               static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z)};

    int blockSize = blockSizeArr[coord.z];

    if (coord.x >= blockSize || coord.y >= blockSize)
    {
        return;
    }

    kernel[coord] = 1.f / (blockSize * blockSize);
}

__global__ void computeGaussianKernelVarShape(musa::Tensor3DWrap<float, int32_t> kernel,
                                              musa::Tensor1DWrap<int, int32_t>   blockSizeArr)
{
    int3 coord{static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x),
               static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y),
               static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z)};

    int blockSize = blockSizeArr[coord.z];

    if (coord.x >= blockSize || coord.y >= blockSize)
    {
        return;
    }

    int2 half{blockSize / 2, blockSize / 2};

    double2 sigma;
    sigma.x = 0.3 * ((blockSize - 1) * 0.5 - 1) + 0.8;
    sigma.y = sigma.x;

    kernel[coord] = computeSingleGaussianValue(coord, half, sigma);
}

} // namespace nvcv::legacy::musa_op
