# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(CV_CUDA_PRIV_LEGACY_FILES CvCudaLegacyHelpers.cpp)

set(CV_CUDA_PRIV_LEGACY_OP_FILES
    filter_utils.mu
    custom_crop.mu
    reformat.mu
    resize_var_shape.mu
    convert_to.mu
    normalize.mu
    pad_and_stack.mu
    normalize_var_shape.mu
    copy_make_border.mu
    copy_make_border_var_shape.mu
    center_crop.mu
    rotate.mu
    rotate_var_shape.mu
    filter.mu
    erase.mu
    erase_var_shape.mu
    filter_var_shape.mu
    median_blur.mu
    bilateral_filter.mu
    bilateral_filter_var_shape.mu
    joint_bilateral_filter.mu
    joint_bilateral_filter_var_shape.mu
    median_blur_var_shape.mu
    morphology.mu
    cvt_color.mu
    warp.mu
    warp_var_shape.mu
    cvt_color_var_shape.mu
    composite.mu
    morphology_var_shape.mu
    channel_reorder_var_shape.mu
    flip.mu
    flip_or_copy_var_shape.mu
    composite_var_shape.mu
    custom_crop.mu
    reformat.mu
    resize_var_shape.mu
    convert_to.mu
    normalize.mu
    pad_and_stack.mu
    normalize_var_shape.mu
    gamma_contrast_var_shape.mu
    pillow_resize.mu
    pillow_resize_var_shape.mu
    threshold.mu
    adaptive_threshold.mu
    adaptive_threshold_var_shape.mu
    threshold_var_shape.mu
    threshold_util.mu
    box_blur.mu
    osd.mu
    textbackend/backend.cpp
    textbackend/stb.cpp
    random_resized_crop.mu
    random_resized_crop_var_shape.mu
    gaussian_noise.mu
    gaussian_noise_var_shape.mu
    gaussian_noise_util.mu
    inpaint.mu
    inpaint_var_shape.mu
    inpaint_utils.mu
    calc_hist.mu
    histogram_eq.mu
    histogram_eq_var_shape.mu
    min_area_rect.mu
)

add_compile_definitions(ENABLE_TEXT_BACKEND_STB)

# filter only one that matches the patern (case insensitive), should be set on the global level
# usage:
# set(CV_CUDA_SRC_PATERN medianblur median_blur"
#
# will compile only files relevant to themedian blur op "median_blur.mu;median_blur_var_shape.mu"
if (NOT "${CV_CUDA_SRC_PATERN}" STREQUAL "")
    foreach(PATTERN ${CV_CUDA_SRC_PATERN})
        string(TOLOWER ${PATTERN} PATTERN)
        foreach(FILENAME ${CV_CUDA_PRIV_LEGACY_OP_FILES})
            string(TOLOWER ${FILENAME} FILENAME_LOWERCASE)
            if (${FILENAME_LOWERCASE} MATCHES ${PATTERN})
                list(APPEND CV_CUDA_PRIV_LEGACY_FILES ${FILENAME})
            endif()
        endforeach()
    endforeach()
else()
    list(APPEND CV_CUDA_PRIV_LEGACY_FILES ${CV_CUDA_PRIV_LEGACY_OP_FILES})
endif()

add_library(cvcuda_legacy STATIC
    ${CV_CUDA_PRIV_LEGACY_FILES}
)

target_compile_definitions(cvcuda_legacy PUBLIC NVCV_USE_MUSA=1)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    target_compile_definitions(cvcuda_legacy PUBLIC NVCV_MUSA_ARM_PORTABLE_BACKEND=1)
endif()

target_link_libraries(cvcuda_legacy
    PUBLIC
        MUSA::musart
        nvcv_types
        nvcv_util
        cvcuda_headers
        -lrt
)
