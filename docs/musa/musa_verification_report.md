[//]: # "SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# MUSA Verification Report

This page lists the concrete verification cases used to validate the MUSA port.
The latest full verification round completed with no failures in the required V2 handoff set.

Back to [CV-CUDA With MUSA](musa.md).

See [Native CV-CUDA Benchmark Results: MUSA and CUDA](musa_native_benchmark.md) for the native CV-CUDA MUSA benchmark results, and [OpenCV CPU Versus CV-CUDA MUSA Benchmark Comparison](musa_opencv_comparison_benchmark.md) for the separate OpenCV comparison data collected alongside this verification work.

## Verification environment

The verification data on this page was collected on the following platform:

| Item | Value |
| --- | --- |
| OS | Ubuntu 22.04.4 LTS (Jammy Jellyfish) |
| CPU architecture | x86_64 |
| CPU | AMD Ryzen 7 5700G with Radeon Graphics |
| CPU cores / threads | 8 cores / 16 threads |
| CPU max frequency | 3.8 GHz |
| NUMA nodes | 1 |
| MUSA SDK / driver version | 5.1.0 / 5.1 |
| MUSA compiler | `/usr/local/musa/bin/mcc` |
| Validated device family | MT S5000 class target |
| Validated architecture setting | `CV_CUDA_MUSA_ARCH=mp_31` |
| Build path | Standard CMake flow with `USE_MUSA=ON` |

## Overall result

| Test binary | Scope | Passed | Failed | Skipped | Notes |
| --- | --- | --- | --- | --- | --- |
| `cvcuda_test_system_smoke` | CV-CUDA operator smoke tests | 25/25 | 0 | 0 | BndBox, BoxBlur, and OSD smoke coverage |
| `cvcuda_test_unit` | CV-CUDA runtime/unit coverage | 16/17 executed | 0 | 1 | One stream handle-reuse case skipped because handle reuse could not be triggered |
| `nvcv_test_types_unit` | NVCV type-system unit coverage | 679/679 | 0 | 0 | Core C++ template/type helpers and CUDA-style compatibility helpers |
| `nvcv_test_types_system` | NVCV C/C++ API system coverage | 2139/2139 | 0 | 0 | Version, status, format, allocator, image, tensor, and resource API coverage |
| `nvcv_test_cudatools_unit` | CUDA/MUSA legacy helper compatibility | 49/49 | 0 | 0 | Legacy format, data-type, error translation, and string conversion helpers |
| Configurable-arch smoke | Rebuild and smoke run with `CV_CUDA_MUSA_ARCH=mp_31` | 25/25 | 0 | 0 | Confirms the architecture selection path |

The required V2 result is 2,908 executed tests passed with 0 failures. The historical handoff summary records the same validation set as 2,907 passed after excluding the intentionally skipped stream handle-reuse case.

## CV-CUDA operator smoke cases

`cvcuda_test_system_smoke` passed all 25 smoke cases.

| Suite | Cases | Result |
| --- | --- | --- |
| `OpBndBox_Smoke` | `operator_creation`, `basic_functionality_rgb8`, `basic_functionality_rgba8`, `multiple_boxes`, `memory_management`, `edge_cases` | 6/6 passed |
| `OpBoxBlur_Smoke` | `operator_creation`, `basic_functionality_rgb8`, `basic_functionality_rgba8`, `multiple_boxes`, `various_kernel_sizes`, `memory_management`, `edge_cases`, `batch_processing` | 8/8 passed |
| `OpOSD_Smoke` | `operator_creation`, `rectangle_element`, `text_element`, `line_element`, `point_element`, `circle_element`, `multiple_elements`, `memory_management`, `edge_cases`, `batch_processing`, `various_image_sizes` | 11/11 passed |

## CV-CUDA unit cases

`cvcuda_test_unit` completed without failures.

| Suite | Cases | Result |
| --- | --- | --- |
| `WorkspaceMemAllocatorTest` | `Get`, `ExceedWorkspaceSize`, `AcquireRelease`, `Sync` | 4/4 passed |
| `WorkspaceAllocatorTest` | `Get` | 1/1 passed |
| `WorkspaceMemEstimatorTest` | `Add` | 1/1 passed |
| `WorkspaceEstimatorTest` | `Add` | 1/1 passed |
| `StreamIdTest` | `RegularAndDefault`, `PerThreadDefault` | 2/3 passed; `HandleReuse` skipped because handle reuse could not be triggered |
| `SimpleCacheTest` | `PutGet` | 1/1 passed |
| `StreamCacheItemAllocator` | `BasicTest` | 1/1 passed |
| `StreamOrderedCacheTest` | `InsertGet`, `FindNextReady`, `RemoveAllReady` | 3/3 passed |
| `PerStreamCacheTest` | `NoStream`, `TwoStream` | 2/2 passed |

## NVCV type and system coverage

The MUSA validation includes the broad NVCV type and API suites.
These suites confirm that the MUSA build keeps the CUDA-facing ABI and behavior expected by the CV-CUDA runtime.

| Test binary | Representative covered areas | Result |
| --- | --- | --- |
| `nvcv_test_types_unit` | MD5 hashing, value/type lists, typed tests, status/error macros, string buffers, static vectors, small vectors, span helpers, math/type utilities, and internal helper templates | 679/679 passed |
| `nvcv_test_types_system` | Version APIs, status APIs, color specs, data layout, image formats, data types, allocators, image batches, tensors, tensor batches, resources, and C/C++ API behavior | 2139/2139 passed |
| `nvcv_test_cudatools_unit` | Legacy data-format conversion, data-type conversion, image-batch format checking, error translation, and error string conversion | 49/49 passed |

## Configurable architecture validation

A separate build-and-smoke pass validated explicit architecture selection:

- `CV_CUDA_MUSA_ARCH=mp_31` was passed through to the MUSA compiler.
- The generated flags included `--offload-arch=mp_31`.
- The smoke build compiled MUSA legacy objects and completed `cvcuda_test_system_smoke` with 25/25 passed.

## Interpretation

The verified area is the native C/C++ MUSA path used by the required V2 handoff.
It covers the core NVCV runtime, CV-CUDA workspace/cache utilities, selected legacy operator smoke flows, CUDA-style helper compatibility, and architecture configuration.
Python package/runtime interoperability was not part of the required pass-case gate.
