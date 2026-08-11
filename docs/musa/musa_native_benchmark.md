[//]: # "SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"

# Native CV-CUDA Benchmark Results: MUSA and CUDA

This page records the native CV-CUDA benchmark results collected with the official `bench/run_bench.py -n 5 <build>/bin` flow.
It compares the MUSA native benchmark run against a CUDA native benchmark reference run using the same official summary fields.
The comparison table lists only the operators common to both official benchmark summaries; CUDA-only `FindHomography` is omitted because it has no MUSA counterpart in this port.

The OpenCV CPU versus CV-CUDA MUSA comparison data is documented separately in [OpenCV CPU Versus CV-CUDA MUSA Benchmark Comparison](musa_opencv_comparison_benchmark.md).

Back to [CV-CUDA With MUSA](musa.md).

## Test environments

| Item | MUSA native run | CUDA native reference run |
| --- | --- | --- |
| Device | MT S5000 class target | NVIDIA GeForce RTX 4080 |
| Driver / SDK | MUSA SDK 5.1.0 / driver 5.1 | NVIDIA driver 590.48.01 |
| Compiler / toolkit | `/usr/local/musa/bin/mcc` | CUDA runtime environment `/usr/local/cuda-13.1` |
| Benchmark executables discovered | 50 | 51 |
| Requested benchmark passes | 5 | 5 |
| Successful operator summaries | 49 | 50 |
| Summarized runs across shared operators | 1010 | 1010 |

Notes:

- Both sides use the official CV-CUDA benchmark script `bench/run_bench.py -n 5 <build>/bin`.
- The MUSA benchmark build excludes `BenchFindHomography.cpp`; the CUDA build includes it, but it is omitted from the comparison table below.
- Both official summary CSV files contain the common fields `Benchmark`, `Mean GPU Time (sec)`, `Std Dev GPU Time (sec)`, and `Runs`, so the comparison table below uses only those fields.
- `Mean GPU Time (sec)` is the average GPU execution time across the measured runs for a benchmark case.
- `Std Dev GPU Time (sec)` is the standard deviation of those GPU execution times, which shows how stable the measurements are.
- `Runs` is the number of measured samples that contributed to the reported mean and standard deviation.
- Timing values below are GPU time in seconds.

## Build and run commands

### MUSA native benchmark

```bash
cmake -S . -B build-musa-bench-s5000 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_MUSA=ON \
  -DCV_CUDA_MUSA_ARCH=mp_31 \
  -DBUILD_TESTS=OFF \
  -DBUILD_PYTHON=OFF \
  -DBUILD_BENCH=ON

cmake --build build-musa-bench-s5000 --target bench_all -- -j$(nproc)
python3 bench/run_bench.py -n 5 build-musa-bench-s5000/bin
```

### CUDA native benchmark reference

```bash
cd /home/mt/data/yaguang/CV-CUDA
python3 bench/run_bench.py -n 5 build-cuda-bench-compare/bin
```

## Persisted data files

| Backend | File | Description |
| --- | --- | --- |
| MUSA | `build-musa-bench-minimal-s5000/musa_official_nvbench_run_n5.log` | Full console log for the official MUSA `-n 5` benchmark run |
| MUSA | `build-musa-bench-minimal-s5000/bin/bench_output.csv` | Raw aggregated output written by the official MUSA benchmark script |
| MUSA | `build-musa-bench-minimal-s5000/bin/bench_gpu_stats.csv` | Official MUSA summary statistics |
| CUDA | `build-cuda-bench-compare/bin/bench_output.csv` | Raw aggregated output written by the official CUDA benchmark script |
| CUDA | `build-cuda-bench-compare/bin/bench_gpu_stats.csv` | Official CUDA summary statistics |

## Native benchmark comparison (`-n 5`)

| Operator | Backend | Runs | Mean GPU Time (sec) | Std Dev GPU Time (sec) |
| --- | --- | ---: | ---: | ---: |
| AdaptiveThreshold | MUSA | 10 | 0.000272 | 0.000050 |
|  | CUDA | 10 | 0.000081 | 0.000007 |
| AdvCvtColor | MUSA | 5 | 0.000036 | 0.000000 |
|  | CUDA | 5 | 0.000030 | 0.000000 |
| AverageBlur | MUSA | 20 | 0.000532 | 0.000014 |
|  | CUDA | 20 | 0.000100 | 0.000006 |
| BilateralFilter | MUSA | 20 | 0.000390 | 0.000008 |
|  | CUDA | 20 | 0.000094 | 0.000001 |
| BndBox | MUSA | 20 | 0.000480 | 0.000340 |
|  | CUDA | 20 | 0.000111 | 0.000069 |
| BoxBlur | MUSA | 10 | 0.000091 | 0.000001 |
|  | CUDA | 10 | 0.000049 | 0.000002 |
| BrightnessContrast | MUSA | 20 | 0.000058 | 0.000017 |
|  | CUDA | 20 | 0.000027 | 0.000004 |
| CenterCrop | MUSA | 10 | 0.000019 | 0.000000 |
|  | CUDA | 10 | 0.000008 | 0.000000 |
| ChannelReorder | MUSA | 10 | 0.000086 | 0.000003 |
|  | CUDA | 10 | 0.000049 | 0.000008 |
| ColorTwist | MUSA | 20 | 0.000061 | 0.000018 |
|  | CUDA | 20 | 0.000032 | 0.000004 |
| Composite | MUSA | 10 | 0.000052 | 0.000003 |
|  | CUDA | 10 | 0.000064 | 0.000005 |
| Conv2D | MUSA | 10 | 0.000558 | 0.000009 |
|  | CUDA | 10 | 0.000093 | 0.000000 |
| ConvertTo | MUSA | 10 | 0.000030 | 0.000001 |
|  | CUDA | 10 | 0.000027 | 0.000005 |
| CopyMakeBorder | MUSA | 20 | 0.000130 | 0.000012 |
|  | CUDA | 20 | 0.000045 | 0.000005 |
| CropFlipNormalizeReformat | MUSA | 10 | 0.000133 | 0.000001 |
|  | CUDA | 10 | 0.000044 | 0.000005 |
| CustomCrop | MUSA | 10 | 0.000030 | 0.000002 |
|  | CUDA | 10 | 0.000030 | 0.000006 |
| CvtColor | MUSA | 200 | 0.000649 | 0.001713 |
|  | CUDA | 200 | 0.000383 | 0.000585 |
| Erase | MUSA | 20 | 0.000089 | 0.000011 |
|  | CUDA | 20 | 0.000033 | 0.000011 |
| Flip | MUSA | 20 | 0.000037 | 0.000007 |
|  | CUDA | 20 | 0.000024 | 0.000005 |
| GammaContrast | MUSA | 10 | 0.000204 | 0.000115 |
|  | CUDA | 10 | 0.000030 | 0.000003 |
| Gaussian | MUSA | 20 | 0.002462 | 0.000462 |
|  | CUDA | 20 | 0.000448 | 0.000086 |
| GaussianNoise | MUSA | 20 | 0.005556 | 0.000320 |
|  | CUDA | 20 | 0.001542 | 0.000012 |
| Histogram | MUSA | 5 | 0.000648 | 0.000000 |
|  | CUDA | 5 | 0.000074 | 0.000000 |
| HistogramEq | MUSA | 20 | 0.000615 | 0.000159 |
|  | CUDA | 20 | 0.000086 | 0.000012 |
| HQResize | MUSA | 40 | 0.000112 | 0.000026 |
|  | CUDA | 40 | 0.000037 | 0.000006 |
| Inpaint | MUSA | 20 | 0.560430 | 0.412944 |
|  | CUDA | 20 | 0.072430 | 0.052153 |
| JointBilateralFilter | MUSA | 20 | 0.000716 | 0.000046 |
|  | CUDA | 20 | 0.000134 | 0.000001 |
| Label | MUSA | 10 | 0.000199 | 0.000004 |
|  | CUDA | 10 | 0.000043 | 0.000002 |
| Laplacian | MUSA | 20 | 0.000457 | 0.000065 |
|  | CUDA | 20 | 0.000077 | 0.000017 |
| MinAreaRect | MUSA | 5 | 0.000114 | 0.000001 |
|  | CUDA | 5 | 0.000013 | 0.000000 |
| MinMaxLoc | MUSA | 20 | 0.000137 | 0.000035 |
|  | CUDA | 20 | 0.000032 | 0.000010 |
| Morphology | MUSA | 80 | 0.000247 | 0.000088 |
|  | CUDA | 80 | 0.000050 | 0.000014 |
| NMS | MUSA | 10 | 0.000781 | 0.000010 |
|  | CUDA | 10 | 0.000145 | 0.000007 |
| Normalize | MUSA | 20 | 0.000043 | 0.000007 |
|  | CUDA | 20 | 0.000029 | 0.000004 |
| OSD | MUSA | 10 | 0.003032 | 0.000006 |
|  | CUDA | 10 | 0.000685 | 0.000001 |
| PadAndStack | MUSA | 10 | 0.000074 | 0.000001 |
|  | CUDA | 10 | 0.000024 | 0.000003 |
| PairwiseMatcher | MUSA | 10 | 0.348976 | 0.188136 |
|  | CUDA | 10 | 0.017532 | 0.014871 |
| PillowResize | MUSA | 20 | 0.000259 | 0.000085 |
|  | CUDA | 20 | 0.000047 | 0.000009 |
| RandomResizedCrop | MUSA | 20 | 0.000172 | 0.000009 |
|  | CUDA | 20 | 0.000067 | 0.000009 |
| Reformat | MUSA | 10 | 0.000032 | 0.000001 |
|  | CUDA | 10 | 0.000026 | 0.000005 |
| Remap | MUSA | 20 | 0.000202 | 0.000053 |
|  | CUDA | 20 | 0.000079 | 0.000020 |
| Resize | MUSA | 20 | 0.000118 | 0.000033 |
|  | CUDA | 20 | 0.000065 | 0.000027 |
| ResizeCropConvertReformat | MUSA | 10 | 0.000238 | 0.000012 |
|  | CUDA | 10 | 0.000099 | 0.000008 |
| Rotate | MUSA | 20 | 0.000076 | 0.000005 |
|  | CUDA | 20 | 0.000064 | 0.000001 |
| SIFT | MUSA | 5 | 0.034293 | 0.000017 |
|  | CUDA | 5 | 0.007543 | 0.000924 |
| Stack | MUSA | 20 | 0.000135 | 0.000036 |
|  | CUDA | 20 | 0.000212 | 0.000082 |
| Threshold | MUSA | 20 | 0.000093 | 0.000059 |
|  | CUDA | 20 | 0.000026 | 0.000004 |
| WarpAffine | MUSA | 20 | 0.000346 | 0.000023 |
|  | CUDA | 20 | 0.000089 | 0.000007 |
| WarpPerspective | MUSA | 20 | 0.000267 | 0.000069 |
|  | CUDA | 20 | 0.000053 | 0.000003 |

## Fastest common operators by mean GPU time

### MUSA

| Operator | Backend | Runs | Mean GPU Time (sec) | Std Dev GPU Time (sec) |
| --- | --- | ---: | ---: | ---: |
| CenterCrop | MUSA | 10 | 0.000019 | 0.000000 |
| ConvertTo | MUSA | 10 | 0.000030 | 0.000001 |
| CustomCrop | MUSA | 10 | 0.000030 | 0.000002 |
| Reformat | MUSA | 10 | 0.000032 | 0.000001 |
| AdvCvtColor | MUSA | 5 | 0.000036 | 0.000000 |
| Flip | MUSA | 20 | 0.000037 | 0.000007 |
| Normalize | MUSA | 20 | 0.000043 | 0.000007 |
| Composite | MUSA | 10 | 0.000052 | 0.000003 |
| BrightnessContrast | MUSA | 20 | 0.000058 | 0.000017 |
| ColorTwist | MUSA | 20 | 0.000061 | 0.000018 |

### CUDA

| Operator | Backend | Runs | Mean GPU Time (sec) | Std Dev GPU Time (sec) |
| --- | --- | ---: | ---: | ---: |
| CenterCrop | CUDA | 10 | 0.000008 | 0.000000 |
| MinAreaRect | CUDA | 5 | 0.000013 | 0.000000 |
| Flip | CUDA | 20 | 0.000024 | 0.000005 |
| PadAndStack | CUDA | 10 | 0.000024 | 0.000003 |
| Reformat | CUDA | 10 | 0.000026 | 0.000005 |
| Threshold | CUDA | 20 | 0.000026 | 0.000004 |
| BrightnessContrast | CUDA | 20 | 0.000027 | 0.000004 |
| ConvertTo | CUDA | 10 | 0.000027 | 0.000005 |
| Normalize | CUDA | 20 | 0.000029 | 0.000004 |
| AdvCvtColor | CUDA | 5 | 0.000030 | 0.000000 |

## Slowest common operators by mean GPU time

### MUSA

| Operator | Backend | Runs | Mean GPU Time (sec) | Std Dev GPU Time (sec) |
| --- | --- | ---: | ---: | ---: |
| Inpaint | MUSA | 20 | 0.560430 | 0.412944 |
| PairwiseMatcher | MUSA | 10 | 0.348976 | 0.188136 |
| SIFT | MUSA | 5 | 0.034293 | 0.000017 |
| GaussianNoise | MUSA | 20 | 0.005556 | 0.000320 |
| OSD | MUSA | 10 | 0.003032 | 0.000006 |
| Gaussian | MUSA | 20 | 0.002462 | 0.000462 |
| NMS | MUSA | 10 | 0.000781 | 0.000010 |
| JointBilateralFilter | MUSA | 20 | 0.000716 | 0.000046 |
| CvtColor | MUSA | 200 | 0.000649 | 0.001713 |
| Histogram | MUSA | 5 | 0.000648 | 0.000000 |

### CUDA

| Operator | Backend | Runs | Mean GPU Time (sec) | Std Dev GPU Time (sec) |
| --- | --- | ---: | ---: | ---: |
| Inpaint | CUDA | 20 | 0.072430 | 0.052153 |
| PairwiseMatcher | CUDA | 10 | 0.017532 | 0.014871 |
| SIFT | CUDA | 5 | 0.007543 | 0.000924 |
| GaussianNoise | CUDA | 20 | 0.001542 | 0.000012 |
| OSD | CUDA | 10 | 0.000685 | 0.000001 |
| Gaussian | CUDA | 20 | 0.000448 | 0.000086 |
| CvtColor | CUDA | 200 | 0.000383 | 0.000585 |
| Stack | CUDA | 20 | 0.000212 | 0.000082 |
| NMS | CUDA | 10 | 0.000145 | 0.000007 |
| JointBilateralFilter | CUDA | 20 | 0.000134 | 0.000001 |

## Interpretation

The native set covers the same CV-CUDA operator family on both backends, using the same official benchmark driver.
The CUDA-only `FindHomography` benchmark is kept in the CUDA run but excluded from the comparison because the MUSA port does not include that operator.
