[//]: # "SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# CV-CUDA With MUSA

This page summarizes the current MUSA support status for the CV-CUDA port.
Detailed verification pass cases and benchmark data are listed on independent Markdown pages:

- [MUSA Verification Report](musa_verification_report.md)
- [Native CV-CUDA Benchmark Results: MUSA and CUDA](musa_native_benchmark.md)
- [OpenCV CPU Versus CV-CUDA MUSA Benchmark Comparison](musa_opencv_comparison_benchmark.md)
- [MUSA and CUDA OpenCV Benchmark Comparison](musa_opencv_cuda_benchmark_comparison.md)


## Build and runtime status

The MUSA build path is working with the standard CMake flow:

- `USE_MUSA=ON` switches the project to the MUSA toolchain and MUSA source overlays.
- The validated compiler is `/usr/local/musa/bin/mcc` with `CMAKE_MUSA_STANDARD=17`.
- `CV_CUDA_MUSA_ARCH` is optional. The validated configuration used `mp_22` for S4000. For S5000, use `CV_CUDA_MUSA_ARCH=mp_31`.

## Build and usage on MUSA

### Environment

Install the MUSA toolkit and make sure the MUSA compiler is available. The validated environment used:

- MUSA SDK 5.1.0
- MUSA compiler: `/usr/local/musa/bin/mcc`
- validated offload architecture: `mp_22` for S4000 (`mp_31` should be used for S5000)

Recommended environment setup:

```bash
export MUSA_HOME=/usr/local/musa
export PATH=${MUSA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${MUSA_HOME}/lib:${MUSA_HOME}/lib64:${LD_LIBRARY_PATH}
```

Check the compiler:

```bash
which mcc
mcc --version
```

### Configure and build

Use one of the following build modes depending on whether you only need the MUSA libraries or also want to build the supported MUSA verification targets.
In both modes, pass `CV_CUDA_MUSA_ARCH` explicitly so the build only targets the GPU architecture in use:

- S4000: `CV_CUDA_MUSA_ARCH=mp_22`
- S5000: `CV_CUDA_MUSA_ARCH=mp_31`

#### Build mode 1: library build without tests

Use the project build wrapper when you only need the MUSA-built libraries. This is the recommended general build command because it avoids compiling test targets that are outside the current MUSA validation scope.

```bash
# Release library build for S4000
ci/build.sh release build-musa-s4000 \
  -DUSE_MUSA=ON \
  -DCV_CUDA_MUSA_ARCH=mp_22 \
  -DBUILD_TESTS=OFF \
  -DBUILD_PYTHON=OFF \
  -DBUILD_BENCH=OFF

# Release library build for S5000
ci/build.sh release build-musa-s5000 \
  -DUSE_MUSA=ON \
  -DCV_CUDA_MUSA_ARCH=mp_31 \
  -DBUILD_TESTS=OFF \
  -DBUILD_PYTHON=OFF \
  -DBUILD_BENCH=OFF
```

#### Build mode 2: supported MUSA verification targets

To reproduce the supported native MUSA verification set, configure a build tree with C++ tests enabled, then build only the validated targets explicitly. Do not use the build wrapper for this mode, because `ci/build.sh` builds the default `all` target.

```bash
# Configure supported-test build for S4000
cmake -S . -B build-musa-tests-s4000 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_MUSA=ON \
  -DCV_CUDA_MUSA_ARCH=mp_22 \
  -DBUILD_TESTS=ON \
  -DBUILD_TESTS_CPP=ON \
  -DBUILD_TESTS_PYTHON=OFF \
  -DBUILD_PYTHON=OFF \
  -DBUILD_BENCH=OFF

# Configure supported-test build for S5000
cmake -S . -B build-musa-tests-s5000 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_MUSA=ON \
  -DCV_CUDA_MUSA_ARCH=mp_31 \
  -DBUILD_TESTS=ON \
  -DBUILD_TESTS_CPP=ON \
  -DBUILD_TESTS_PYTHON=OFF \
  -DBUILD_PYTHON=OFF \
  -DBUILD_BENCH=OFF
```

Then build the supported MUSA verification targets:

```bash
# S4000
cmake --build build-musa-tests-s4000 --target \
  cvcuda_test_system_smoke \
  cvcuda_test_unit \
  nvcv_test_types_unit \
  nvcv_test_types_system \
  nvcv_test_cudatools_unit \
  -- -j$(nproc)

# S5000
cmake --build build-musa-tests-s5000 --target \
  cvcuda_test_system_smoke \
  cvcuda_test_unit \
  nvcv_test_types_unit \
  nvcv_test_types_system \
  nvcv_test_cudatools_unit \
  -- -j$(nproc)
```

The current MUSA verification scope is the supported native artifact set listed above. It is not the complete upstream CUDA test target set.
Avoid using only `-DUSE_MUSA=ON` for normal builds unless you intentionally want the default multi-architecture build. Without `CV_CUDA_MUSA_ARCH`, the compiler may try multiple offload targets such as `mp_21`, `mp_22`, `mp_31`, and `mp_32`, which is slower and may fail if an unsupported target is selected by the local MUSA toolkit.

To build benchmark targets as well:

```bash
# Release build with MUSA benchmarks for S4000
ci/build.sh release build-musa-bench-s4000 -DUSE_MUSA=ON -DCV_CUDA_MUSA_ARCH=mp_22 -DBUILD_TESTS=OFF -DBUILD_PYTHON=OFF -DBUILD_BENCH=ON

# Release build with MUSA benchmarks for S5000
ci/build.sh release build-musa-bench-s5000 -DUSE_MUSA=ON -DCV_CUDA_MUSA_ARCH=mp_31 -DBUILD_TESTS=OFF -DBUILD_PYTHON=OFF -DBUILD_BENCH=ON
```

### Run verification tests

After a build with tests enabled, run the generated test binaries from the build tree:

```bash
# CV-CUDA operator smoke tests
build-musa/bin/cvcuda_test_system_smoke

# CV-CUDA runtime/unit tests
build-musa/bin/cvcuda_test_unit

# NVCV type-system unit tests
build-musa/bin/nvcv_test_types_unit

# NVCV type-system API/system tests
build-musa/bin/nvcv_test_types_system

# CUDA/MUSA legacy helper compatibility tests
build-musa/bin/nvcv_test_cudatools_unit
```

The generated test runner can also be used when present:

```bash
build-musa/bin/run_tests.sh cvcuda,cpp
build-musa/bin/run_tests.sh nvcv,cpp
```

### Build and run benchmarks

Build this project with `BUILD_BENCH=ON` to enable benchmark targets. Two benchmark flows are documented:

- native CV-CUDA MUSA benchmark executables driven by the NVBench-compatible runner
- external OpenCV CPU versus CV-CUDA MUSA comparison harness

See [Native CV-CUDA Benchmark Results: MUSA and CUDA](musa_native_benchmark.md) for the native `-n 5` benchmark results, and [OpenCV CPU Versus CV-CUDA MUSA Benchmark Comparison](musa_opencv_comparison_benchmark.md) for the separate OpenCV comparison results.

### Use the MUSA build from C/C++

For C/C++ applications, link against the MUSA-built CV-CUDA libraries from the build or install tree. Use the public CV-CUDA/NVCV headers as usual; the MUSA build selects the MUSA implementation through the build configuration.

Typical runtime setup when running directly from a build tree:

```bash
export LD_LIBRARY_PATH=$PWD/build-musa-s5000/lib:${MUSA_HOME}/lib:${MUSA_HOME}/lib64:${LD_LIBRARY_PATH}
```

Then run applications or tests that use the built `nvcv_types` and `cvcuda` libraries.

### Notes and limitations

- `USE_MUSA=ON` is the required switch for the MUSA backend.
- The validated handoff focuses on the native C/C++ MUSA path. Python package/runtime interoperability was not part of the required verification gate.
- CUDA Python / CuPy style interop remains optional for the MUSA port status summarized here.

## Verification summary

| Item | Result |
| --- | --- |
| MUSA toolchain discovery | Pass |
| Build configuration | Pass |
| `cvcuda_test_system_smoke` | Pass: 25/25 |
| `cvcuda_test_unit` | Pass: 16 executed, 0 failed, 1 skipped |
| `nvcv_test_types_unit` | Pass: 679/679 |
| `nvcv_test_types_system` | Pass: 2139/2139 |
| `nvcv_test_cudatools_unit` | Pass: 49/49 |
| Configurable-arch smoke with `CV_CUDA_MUSA_ARCH=mp_22` | Pass: 25/25 |

See [MUSA Verification Pass Cases](musa_verification_report.md) for the concrete pass-case list.

## Supported areas

The port covers the core runtime and operator layers needed for the verified flow:

- core CV-CUDA / NVCV runtime
- MUSA runtime, driver, math, random, BLAS, SOLVER, and CUB/Thrust compatibility paths
- legacy operators used by smoke coverage, including BndBox, BoxBlur, and OSD
- Python bindings build support, while CUDA Python / CuPy style interop remains optional and not part of the required V2 handoff

## Notes from the analysis artifacts

- no standalone PTX was found in the project sources during the porting analysis
- `OpFindHomography` remains the main SDK hotspot because of the `syevjBatched` surface
- default row alignment and CUDA-style error-string compatibility were adjusted so existing CUDA-facing tests keep passing
