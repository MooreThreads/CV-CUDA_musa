// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// musify: MUSA test-only compatibility include for CUDA driver header usage in
// CV-CUDA's checked-in C++ test sources. This routes driver API names to real
// MUSA driver/runtime APIs when USE_MUSA builds required in-tree test artifacts.
#include <musa.h>
#include <musa_runtime.h>

#define CUstream MUstream
#define cuStreamGetId muStreamGetId
#define cuStreamGetCtx muStreamGetCtx
#define CUDA_SUCCESS MUSA_SUCCESS
