/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * @file CvCudaLegacyHelpers.hpp
 *
 * @brief Defines util functions for conversion between nvcv and legacy cv cuda
 */

#ifndef CV_CUDA_LEGACY_HELPERS_HPP
#define CV_CUDA_LEGACY_HELPERS_HPP

#include "CvCudaLegacy.h"

#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageBatchData.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorShapeInfo.hpp>

namespace nvcv::legacy::helpers {

musa_op::DataFormat GetLegacyDataFormat(int32_t numberChannels, int32_t numberPlanes, int32_t numberInBatch);

musa_op::DataType GetLegacyDataType(int32_t bpc, nvcv::DataKind kind);
musa_op::DataType GetLegacyDataType(DataType dtype);
musa_op::DataType GetLegacyDataType(ImageFormat fmt);

musa_op::DataFormat GetLegacyDataFormat(const TensorLayout &layout);
musa_op::DataFormat GetLegacyDataFormat(const ImageBatchVarShapeDataStridedCuda &imgBatch);
musa_op::DataFormat GetLegacyDataFormat(const ImageBatchVarShape &imgBatch);
musa_op::DataFormat GetLegacyDataFormat(const TensorDataStridedCuda &tensor);

musa_op::DataShape GetLegacyDataShape(const TensorShapeInfoImage &shapeInfo);

Size2D GetMaxImageSize(const TensorDataStridedCuda &tensor);
Size2D GetMaxImageSize(const ImageBatchVarShapeDataStridedCuda &imageBatch);

} // namespace nvcv::legacy::helpers

namespace nvcv::util {

inline bool CheckSucceeded(legacy::musa_op::ErrorCode err)
{
    return err == legacy::musa_op::SUCCESS;
}

NVCVStatus  TranslateError(legacy::musa_op::ErrorCode err);
const char *ToString(legacy::musa_op::ErrorCode err, const char **perrdescr = nullptr);

inline void PreprocessError(legacy::musa_op::ErrorCode)
{
}

} // namespace nvcv::util

#endif // CV_CUDA_LEGACY_HELPERS_HPP
