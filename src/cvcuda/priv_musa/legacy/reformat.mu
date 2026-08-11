/* Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2021-2023, Bytedance Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include "CvCudaLegacy.h"
#include "CvCudaLegacyHelpers.hpp"

#include "CvCudaUtils.muh"

#include <nvcv/Image.hpp>
#include <nvcv/ImageData.hpp>

#include <cassert>

namespace musa    = nvcv::musa;
namespace musa_op = nvcv::legacy::musa_op;

template<musa_op::DataFormat SrcFormat, class SrcWrapper, class DstWrapper>
__global__ void transformFormat(const SrcWrapper src, DstWrapper dst, int3 inout_size)
{
    int3 thrCoord = {static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x),
           static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y),
           static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z)};

    if (thrCoord.x >= inout_size.x || thrCoord.y >= inout_size.y)
        return;

    using DimType = musa::MakeType<int, SrcWrapper::kNumDimensions>;
    DimType srcCoord, dstCoord;

    for (int c = 0; c < inout_size.z; c++)
    {
        if constexpr (SrcFormat == musa_op::kNCHW)
        {
            srcCoord = {thrCoord.x, thrCoord.y, c, thrCoord.z};
            dstCoord = {c, thrCoord.x, thrCoord.y, thrCoord.z};
        }
        else if constexpr (SrcFormat == musa_op::kNHWC)
        {
            srcCoord = {c, thrCoord.x, thrCoord.y, thrCoord.z};
            dstCoord = {thrCoord.x, thrCoord.y, c, thrCoord.z};
        }
        else if constexpr (SrcFormat == musa_op::kCHW)
        {
            srcCoord = {thrCoord.x, thrCoord.y, c};
            dstCoord = {c, thrCoord.x, thrCoord.y};
        }
        else if constexpr (SrcFormat == musa_op::kHWC)
        {
            srcCoord = {c, thrCoord.x, thrCoord.y};
            dstCoord = {thrCoord.x, thrCoord.y, c};
        }

        dst[dstCoord] = src[srcCoord];
    }
}

namespace nvcv::legacy::musa_op {

template<musa_op::DataFormat input_format, typename data_type> // k(N)CHW k(N)HWC, uchar float
nvcv::legacy::musa_op::ErrorCode transform(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                    musaStream_t stream)
{
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    const int3 inout_size = {inAccess->numCols(), inAccess->numRows(), outAccess->numChannels()};

    dim3 block(32, 8);
    dim3 grid(musa_op::divUp(inout_size.x, block.x), musa_op::divUp(inout_size.y, block.y), inAccess->numSamples());

    auto inMaxStride  = inAccess->sampleStride() * inAccess->numSamples();
    auto outMaxStride = outAccess->sampleStride() * outAccess->numSamples();
    if (std::max(inMaxStride, outMaxStride) <= musa::TypeTraits<int32_t>::max)
    {
        musa::TensorNDWrap<const data_type, musa_op::FormatDimensions<input_format>, int32_t> src(inData);
        musa::TensorNDWrap<data_type, musa_op::FormatDimensions<input_format>, int32_t>       dst(outData);

        transformFormat<input_format><<<grid, block, 0, stream>>>(src, dst, inout_size);
    }
    else
    {
        LOG_ERROR("Input or output size exceeds " << musa::TypeTraits<int32_t>::max << ". Tensor is too large.");
        return nvcv::legacy::musa_op::ErrorCode::INVALID_PARAMETER;
    }

    checkKernelErrors();

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(musaDeviceSynchronize());
    checkCudaErrors(musaGetLastError());
#endif
    return nvcv::legacy::musa_op::ErrorCode::SUCCESS;
}

void Reformat::checkDataFormat(DataFormat format)
{
    NVCV_ASSERT(format == kNHWC || format == kHWC || format == kNCHW || format == kCHW);
}

ErrorCode Reformat::infer(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                          musaStream_t stream)
{
    DataFormat input_format  = helpers::GetLegacyDataFormat(inData.layout());
    DataFormat output_format = helpers::GetLegacyDataFormat(outData.layout());

    checkDataFormat(input_format);
    checkDataFormat(output_format);

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    if (inData.dtype() == outData.dtype() && inData.shape() == outData.shape())
    {
#ifdef CUDA_DEBUG_LOG
        LOG_ERROR("input_format == output_format, copy outputs from inputs");
#endif

        for (uint32_t i = 0; i < inAccess->numSamples(); ++i)
        {
            nvcv::Byte *inSampData  = inAccess->sampleData(i);
            nvcv::Byte *outSampData = outAccess->sampleData(i);

            for (int p = 0; p < inAccess->numPlanes(); ++p)
            {
                checkCudaErrors(musaMemcpy2DAsync(outAccess->planeData(p, outSampData), outAccess->rowStride(),
                                                  inAccess->planeData(p, inSampData), inAccess->rowStride(),
                                                  inAccess->numCols() * inAccess->colStride(), inAccess->numRows(),
                                                  musaMemcpyDeviceToDevice, stream));
            }
        }
        return SUCCESS;
    }

    // Only allow CHW <-> HWC or NCHW <-> NHWC reformats
    if ((input_format == musa_op::kNHWC) && !(output_format == musa_op::kNCHW)
        || (input_format == musa_op::kNCHW) && !(output_format == musa_op::kNHWC)
        || (input_format == musa_op::kHWC) && !(output_format == musa_op::kCHW)
        || (input_format == musa_op::kCHW) && !(output_format == musa_op::kHWC))
    {
        LOG_ERROR("Invalid combination of input format " << input_format << " and output format " << output_format);
        return nvcv::legacy::musa_op::ErrorCode::INVALID_DATA_FORMAT;
    }

    DataType data_type     = helpers::GetLegacyDataType(inData.dtype());
    DataType out_data_type = helpers::GetLegacyDataType(outData.dtype());

    if (data_type != out_data_type)
    {
        LOG_ERROR("DataType of input and output must be equal, but got " << data_type << " and " << out_data_type);
        return nvcv::legacy::musa_op::ErrorCode::INVALID_DATA_TYPE;
    }

    if (!(data_type == kCV_8U || data_type == kCV_8S || data_type == kCV_16U || data_type == kCV_16S
          || data_type == kCV_32S || data_type == kCV_32F || data_type == kCV_64F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return nvcv::legacy::musa_op::ErrorCode::INVALID_DATA_TYPE;
    }

    typedef ErrorCode (*transform_t)(const TensorDataStridedCuda &input, const TensorDataStridedCuda &output,
                                     musaStream_t stream);

    static const transform_t funcs[4][7] = {
        {transform<kNCHW, uchar>, transform<kNCHW, schar>, transform<kNCHW, ushort>, transform<kNCHW, short>,
         transform<kNCHW, int>, transform<kNCHW, float>, transform<kNCHW, double>},
        {transform<kNHWC, uchar>, transform<kNHWC, schar>, transform<kNHWC, ushort>, transform<kNHWC, short>,
         transform<kNHWC, int>, transform<kNHWC, float>, transform<kNHWC, double>},
        { transform<kCHW, uchar>,  transform<kCHW, schar>,  transform<kCHW, ushort>,  transform<kCHW, short>,
         transform<kCHW, int>,  transform<kCHW, float>,  transform<kCHW, double> },
        { transform<kHWC, uchar>,  transform<kHWC, schar>,  transform<kHWC, ushort>,  transform<kHWC, short>,
         transform<kHWC, int>,  transform<kHWC, float>,  transform<kHWC, double> }
    };

    transform_t func = funcs[input_format][data_type];
    return func(inData, outData, stream);
}

} // namespace nvcv::legacy::musa_op
