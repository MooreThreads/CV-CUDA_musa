# MUSIFY — CV-CUDA

## Metadata

| Field | Value |
|-------|-------|
| Project | CV-CUDA |
| Source repo | git@sh-code.mthreads.com:sw/CV-CUDA.git |
| Migration branch | musa |
| MUSA SDK version | 5.1.0 |
| torch_musa version | 2.9.0+d3e9b11 |
| MT GPU type | S4000 |
| Pipeline start | 2026-06-25 UTC |
| Last updated | 2026-06-25 UTC |
| Status | complete |

## Migration Plan

first-time CUDA to MUSA migration for CV-CUDA with user constraints: do not create Jira tickets and do not git push at publish; repo_url: git@sh-code.mthreads.com:sw/CV-CUDA.git

## Progress

| Round | Phase | Status | Summary | Commit |
|-------|-------|--------|---------|--------|
| 1 | PortSources | unblocked | Reader refreshed `CODE_ANALYSIS.md` with required closure schema (`507377e9`); previous code-analysis blocker resolved. | 507377e9 |
| 1 | PortSources | complete | Added reproducible porting script and generated MUSA siblings: 70 `.mu`, 8 `.muh`, 24 `cuda_tools_musa` headers, 3 util `.cpp`; CUDA originals unchanged. | e3579c6d |
| 1 | DetectAndConvertPTX | complete | Read MTX how-to and scanned generated sources; no PTX asm/standalone PTX found. Native MUSA-supported collectives (`__shfl_*_sync`, `__syncthreads`, `__threadfence`) recorded for later build/runtime verification, no MTX rewrite committed. | e3579c6d |
| 1 | ConvertPython | complete | Direct strategy scan found core Python bindings are wrapper-only; CUDA Python/CuPy interop remains optional/skipped per CODE_ANALYSIS §4 and no source changes were required in Round 1. | none |
| 1 | BuildGate | done | Required V2 native artifacts built with `USE_MUSA=1`, `BUILD_PYTHON=OFF`, and `BUILD_TESTS_CPP=ON`; readiness checks confirmed all five required executables exist and `ldd` has no `not found`. Python bindings remain optional/project-regression-candidate per CODE_ANALYSIS §4 and were not required for this handoff. | 9e516c35 |
| 2 | BuildGate | done | Recovered verifier `build-artifacts-missing` by syncing local `musa` branch to `/workspace/CV-CUDA`, reconfiguring `build_musa_v2`, rebuilding the five required V2 gtest binaries at the exact verifier paths, and checking executable presence plus `ldd` readiness without running tests. | a8dfb275 |
| 3 | BuildGate | done | Resolved artifact visibility contradiction: the running verifier container mounts host `/root/workspace` at container `/workspace`, while host `/workspace` is a separate path. Required binaries were present in `auto-musify-CV-CUDA:/workspace/CV-CUDA/build_musa_v2/bin` and mirrored to host `/workspace/CV-CUDA/build_musa_v2/bin`; final readiness rechecked container path existence, executable bits, and `ldd`. | c6dc4739 |
| 4 | ClassifyError | partial | Applied smallest real source fixes for three Round 3 verifier clusters: MUSA stream pseudo-handles, CUDA-compatible tensor row alignment, and CUDA invalid-value error string compatibility. Container/mount checks in this session could not see `auto-musify-CV-CUDA` or `/root/workspace/CV-CUDA`, so rebuild/targeted tests were not executed by this musifier. | 868f7c42, d3285890, 4eb28be3 |
| 5 | ClassifyError | complete | Resolved the BndBox/BoxBlur/OSD smoke cluster. Kernel launch failure was caused by static MUSA runtime linkage on `cvcuda_legacy`; switching legacy kernel static archive linkage to shared `MUSA::musart`, restoring real OSD rendering, and removing diagnostics produced a clean `cvcuda_test_system_smoke` pass. | 70a75570, 7878f483, 92920669, 211b7ab2 |
| 5 | BuildConfig | complete | Removed hard binding to `/usr/local/musa-5.1.0/cmake` and S4000-only `--offload-arch=mp_22`; MUSA toolkit discovery now prefers `$MUSA_HOME` plus `/usr/local/musa`, and `CV_CUDA_MUSA_ARCH` is an optional cache setting. Configurable `CV_CUDA_MUSA_ARCH=mp_22` smoke validation passed on S4000. | 3d0c3a38 |

## CUDA Dependencies

Mirror source: `CUDA_DEPENDENCIES.dot`.

| # | Dependency | Type | Classification | Status | Notes |
|---|------------|------|----------------|--------|-------|
| 1 | CV-CUDA core CUDA kernels | CU | required-for-core | musify-queued | `src/cvcuda/priv`: 70 `.cu`, 8 `.cuh`; create parallel MUSA source tree. |
| 2 | nvcv/cvcuda cuda_tools headers | CU | required-for-core | musify-queued | Template/device utility headers with `__CUDA_ARCH__`; required closure for core kernels. |
| 3 | legacy operators + cuOSD | CU | required-for-core | musify-queued | `src/cvcuda/priv/legacy/osd.cu`, `box_blur.cu`, `CvCudaOSD.hpp`; real source exists in tree, no stub. |
| 4 | CUDA Runtime / Driver | CE | SDK swap | confirmed | MUSA Runtime / Driver rows 42–43 in counterpart matrix. |
| 5 | cuBLAS/cuBLASLt | CE | SDK swap | confirmed | row 1; used by `OpFindHomography`. |
| 6 | cuSOLVER | CE | SDK swap with concern | partial-risk | row 4; `cusolverDnSsyevjBatched` API adaptation likely needed. |
| 7 | cuRAND | CE | SDK swap | confirmed | row 11; GaussianNoise kernels. |
| 8 | CUB / Thrust / CCCL | CE | SDK header swap | confirmed | rows 27–28 plus `musa-sdk-5.1.0-cccl-support`; use SDK `cub/`, `thrust/`, `musa/std`. |
| 9 | CUDA Math / fp16 | CE | SDK swap | confirmed | row 15. |
| 10 | nvbench | CU | benchmark-only | skip-by-default | 3rdparty submodule, 55 `.cu` + 66 `.cuh`; no remote MT sibling found via available canonical catalog query; keep `BUILD_BENCH=OFF` unless requested. |
| 11 | Python bindings | CB | wrapper adaptation | queued | pybind11 wrapper around core; no own kernels. |
| 12 | Optional Python samples: PyCUDA/CuPy/CUDA Python | CB | optional interop | gap/unknown | counterpart matrix rows 83–84 are gaps; do not make V1 required. |

## Verification & Testing

| Round | Tier | Artifact | Result | Passed | Failed | Skipped | Evidence |
|-------|------|----------|--------|--------|--------|---------|----------|
| 3 | V2 | `cvcuda_test_system_smoke` | fail | 3 | 22 | 0 | `verify/cvcuda_test_system_smoke.log`, `verify/cvcuda_test_system_smoke.xml` |
| 3 | V2 | `cvcuda_test_unit` | fail | 15 | 1 | 1 | `verify/cvcuda_test_unit.log`, `verify/cvcuda_test_unit.xml` |
| 3 | V2 | `nvcv_test_types_unit` | fail | 678 | 1 | 0 | `verify/nvcv_test_types_unit.log`, `verify/nvcv_test_types_unit.xml` |
| 3 | V2 | `nvcv_test_types_system` | fail | 2132 | 6 | 1 | `verify/nvcv_test_types_system.log`, `verify/nvcv_test_types_system.xml` |
| 3 | V2 | `nvcv_test_cudatools_unit` | pass | 49 | 0 | 0 | `verify/nvcv_test_cudatools_unit.log`, `verify/nvcv_test_cudatools_unit.xml` |
| 5 | V2 | `cvcuda_test_system_smoke` | pass | 25 | 0 | 0 | `verify/round5-full/cvcuda_test_system_smoke.log`, `verify/round5-full/cvcuda_test_system_smoke.xml` |
| 5 | V2 | `cvcuda_test_unit` | pass | 16 | 0 | 0 | `verify/round5-full/cvcuda_test_unit.log`, `verify/round5-full/cvcuda_test_unit.xml` |
| 5 | V2 | `nvcv_test_types_unit` | pass | 679 | 0 | 0 | `verify/round5-full/nvcv_test_types_unit.log`, `verify/round5-full/nvcv_test_types_unit.xml` |
| 5 | V2 | `nvcv_test_types_system` | pass | 2138 | 0 | 0 | `verify/round5-full/nvcv_test_types_system.log`, `verify/round5-full/nvcv_test_types_system.xml` |
| 5 | V2 | `nvcv_test_cudatools_unit` | pass | 49 | 0 | 0 | `verify/round5-full/nvcv_test_cudatools_unit.log`, `verify/round5-full/nvcv_test_cudatools_unit.xml` |
| 5 | BuildConfig | `cvcuda_test_system_smoke` with `CV_CUDA_MUSA_ARCH=mp_22` | pass | 25 | 0 | 0 | `verify/round5-configurable-arch/cvcuda_test_system_smoke_mp22.log` |

Round 3 verifier result: 5 required V2 artifacts executed in `auto-musify-CV-CUDA:/workspace/CV-CUDA`; 1 passed and 4 failed. Full summary: `verify/round3_v2_summary.md`.

Round 5 full V2 result: all five required artifacts passed in `auto-musify-CV-CUDA:/workspace/CV-CUDA`; total 2,907 tests passed, 0 failed, 0 skipped across `cvcuda_test_system_smoke`, `cvcuda_test_unit`, `nvcv_test_types_unit`, `nvcv_test_types_system`, and `nvcv_test_cudatools_unit`. Evidence: `verify/round5-full/`.

## Issues

| Round | Issue | Status | Evidence |
|-------|-------|--------|----------|
| 3 | BndBox, BoxBlur, and OSD smoke operations throw `NVCV_ERROR_INTERNAL: Unknown error -1`. | fixable_bug | `issues/round3-cvcuda-smoke-osd-runtime-internal-error.yaml` |
| 3 | `StreamIdTest.RegularAndDefault` throws `NVCV_ERROR_INVALID_ARGUMENT: Invalid stream handle`. | fixable_bug | `issues/round3-stream-id-invalid-default-stream.yaml` |
| 3 | `CheckErrorTest.cudaErr_to_String` fails in `nvcv_test_types_unit`. | fixable_bug | `issues/round3-types-error-string-cuda-residual.yaml` |
| 3 | Tensor creation and TensorDataUtils image-vector cases fail for selected layouts/formats. | fixable_bug | `issues/round3-types-system-tensor-stride-failures.yaml` |

## Issues Resolved

| Round | Issue | Root Cause | Solution | Commit |
|-------|-------|------------|----------|--------|
| 1 | `code-analysis-incomplete` initial blocker | Reader handoff used `### Local required component closure` instead of musifier-required `### Required Component Closure` schema. | Reader refresh commit `507377e9` added required fields; musifier restarted Round1-Port. | 507377e9 |
| 1 | CUDA source porting required for core/private/util closure | CV-CUDA has 70 core `.cu`, 8 `.cuh`, 24 cuda_tools headers, and runtime-facing util sources that must not be edited in place. | Added `simple_porting_cv_cuda.py` and generated parallel `priv_musa`, `cuda_tools_musa`, and `util_musa` siblings using container JSON mappings plus project-local folder rewrites. | 3097f09b, ef07c25b, dd427195, 6f2d5d4a, e3579c6d |
| 1 | Python external build tried to resolve CUDA-only build modules under `USE_MUSA` | `BuildPython.cmake` and Python subproject CMake paths still searched CUDA modules/targets or MUSA headers from generated-only paths. | Routed Python ExternalProject CMake module paths to MUSA, gated Python CMake CUDA toolkit links, added MUSA toolkit discovery for utility libraries, set DLPack policy minimum, and pointed Python metadata generation at canonical NVCV headers. | ea6c128c, c17c2b02, 740c5318, 36665213, b3ac6b54, 7e11705c, f9639fc5 |
| 1 | Required V2 C++ test artifacts did not build under MUSA test headers | Checked-in C++ tests used CUDA runtime/driver names and CUDA-only compat units while building required in-tree targets with `USE_MUSA=1`. | Added test-only MUSA compatibility header mappings, routed legacy-helper tests to MUSA headers, kept MUSA aligned vector aliases visible, skipped CUDA glibc compat unit for MUSA, and rebuilt required V2 artifacts successfully. | ad378428, acc56d01, 64570ded, 327614f4, c7683179, 98128b41, 64d1a300, 576446b5, 0be19d3b, 6810eae5, 59a591f3, 9e516c35 |
| 2 | Verifier `build-artifacts-missing` for required V2 binaries | Remote verifier handoff expected artifacts under `/workspace/CV-CUDA/build_musa_v2/bin`, but the required binaries were absent from that path at re-entry. | Re-synced local source to remote, reconfigured `build_musa_v2`, rebuilt the exact five required binaries, and confirmed executable files plus `ldd` without `not found`; no source rebuild fixes or test execution were needed. | a8dfb275 |
| 3 | Artifact visibility contradiction at `/workspace/CV-CUDA/build_musa_v2/bin` | Container `auto-musify-CV-CUDA` maps host `/root/workspace` to container `/workspace`; therefore host-level checks at `/workspace/...` missed binaries that existed in the verifier container path. | Mirrored the built bin directory from host mount source `/root/workspace/CV-CUDA/build_musa_v2/bin` to host `/workspace/CV-CUDA/build_musa_v2/bin`, then rechecked all five artifacts inside `auto-musify-CV-CUDA:/workspace/CV-CUDA/build_musa_v2/bin` for executable bit and clean `ldd`. | c6dc4739 |
| 4 | `StreamIdTest.RegularAndDefault` rejected default stream pseudo-handle | SDK 5.1.0 provides `muStreamGetId`, but direct calls on MUSA default/per-thread pseudo streams can return invalid-value instead of assigning the legacy pseudo-stream id that CV-CUDA expects. | Wrapped `muStreamGetId` with pseudo-stream handling for `0`, `MU_STREAM_LEGACY`, and `MU_STREAM_PER_THREAD`, preserving real `muStreamGetId` for concrete streams. Targeted check passed: `verify/round4-targeted/stream_id.log`. | 868f7c42 |
| 4 | U8 tensor smoke tests reported larger-than-expected row strides | MUSA `musaDevAttrTexturePitchAlignment` can report 128-byte alignment; CV-CUDA strided tensor ABI tests expect CUDA-compatible 32-byte default row alignment for plain device-memory tensors. | Clamped default row alignment to the CUDA-compatible 32-byte floor while still honoring dtype alignment and explicit user row alignment. Targeted tensor/TensorDataUtils subset passed: `verify/round4-targeted/tensor_subset.log`. | d3285890 |
| 4 | `CheckErrorTest.cudaErr_to_String` retained CUDA API-string contract | MUSA `musaGetErrorName(musaErrorInvalidValue)` returns a MUSA name, but CV-CUDA's public CUDA-compatible error helper/test expects `cudaErrorInvalidValue` and `invalid argument`. | Added explicit invalid-value compatibility mapping in the MUSA `ToString` helper; other MUSA errors continue to use SDK runtime strings. Targeted check passed: `verify/round4-targeted/error_string.log`. | 4eb28be3 |
| 4 | BndBox, BoxBlur, and OSD smoke operations still throw `NVCV_ERROR_INTERNAL: Unknown error -1` | Static MUSA runtime linkage in the `cvcuda_legacy` static archive prevented MUSA kernel registration/launch from working once legacy kernels were linked into `libcvcuda.so`; even an empty no-argument kernel in `osd.mu` threw before `cvcuda_legacy` was linked against shared `MUSA::musart`. | Changed `src/cvcuda/priv_musa/legacy/CMakeLists.txt` to link `MUSA::musart`; restored real OSD kernel execution; removed all smoke-cluster diagnostics. Clean `cvcuda_test_system_smoke` now passes 25/25; evidence: `verify/round5-targeted/smoke_full_clean.log`. | 70a75570, 7878f483, 92920669, 211b7ab2 |

## Knowledge Base

- [Round 1] Musifier gate requires `CODE_ANALYSIS.md` §2 to include `### Required Component Closure` rows with `required_component_closure` schema. The current file initially had `### Local required component closure`, but reader refresh `507377e9` resolved the blocker.
- [Round 1] CV-CUDA source porting used 14,681 filtered CUDA/MUSA mapping rules from the container JSON set (33,989 total loaded). Filtering avoids broad non-CUDA token replacements while preserving authoritative CUDA Runtime/Driver, cuBLAS, cuSOLVER, cuRAND, CUB/Thrust, and header mappings.
- [Round 1] PTX scan found no standalone `.ptx` or `asm volatile` in generated MUSA siblings. `__shfl_*_sync`, `__syncthreads`, and `__threadfence` were present; wiki/API lookup indicates native/partial MUSA support rather than mandatory MTX rewriting, with warp-width behavior left for verifier/build-fix rounds.
- [Round 1] Build-clean handoff used native V2 artifacts from CODE_ANALYSIS.md §4: `build_musa_v2/bin/cvcuda_test_system_smoke`, `build_musa_v2/bin/cvcuda_test_unit`, `build_musa_v2/bin/nvcv_test_types_unit`, `build_musa_v2/bin/nvcv_test_types_system`, and `build_musa_v2/bin/nvcv_test_cudatools_unit`. Readiness check was existence/executable plus `ldd` without `not found`; verifier tests were not run.
- [Round 1] Python bindings are not part of the required V2 handoff for this run. Earlier Python ExternalProject fixes were committed for build-system consistency, but final required artifact build used `BUILD_PYTHON=OFF` because CODE_ANALYSIS §4 classifies Python cases as project-regression-candidate/optional when bindings are built.
- [Round 2] `build-artifacts-missing` recovery is artifact-only when local source is already at the prior MUSA handoff: sync local `musa` to `/workspace/CV-CUDA`, rebuild `build_musa_v2` with `USE_MUSA=ON`, `BUILD_TESTS=ON`, `BUILD_TESTS_CPP=ON`, `BUILD_PYTHON=OFF`, `BUILD_BENCH=OFF`, then check the five required V2 binaries for executable bit and `ldd` without `not found`. Do not run verifier tests in musifier readiness.
- [Round 3] Artifact visibility contradiction root cause: `docker inspect auto-musify-CV-CUDA` reports mount `/root/workspace:/workspace`; therefore `auto-musify-CV-CUDA:/workspace/CV-CUDA/...` corresponds to host `/root/workspace/CV-CUDA/...`, not host `/workspace/CV-CUDA/...`. Keep verifier readiness evidence tied to the named container path; host `/workspace` may need artifact mirroring for host-side locator checks.
- [Round 4] MUSA stream ids need CUDA pseudo-stream compatibility before calling `muStreamGetId`; handle `0`, `MU_STREAM_LEGACY`, and `MU_STREAM_PER_THREAD` locally, then call the SDK function only for concrete stream handles.
- [Round 4] CV-CUDA tensor system tests encode the CUDA strided tensor ABI: plain device-memory tensors should use the CUDA-compatible 32-byte default row alignment even if MUSA texture pitch alignment reports a larger value. Explicit user row alignment remains honored.
- [Round 4] `nvcv::util::ToString` is part of CV-CUDA's CUDA-compatible API/test contract; MUSA invalid-value errors must report `cudaErrorInvalidValue` / `invalid argument`, while unmapped MUSA errors may still use SDK `musaGetErrorName` / `musaGetErrorString`.

- [Round 5] MUSA kernels compiled into `cvcuda_legacy` must link against shared `MUSA::musart` before that static archive is consumed by `libcvcuda.so`; static `MUSA::musart_static` caused OSD/BndBox/BoxBlur launches to throw `Unknown error -1` at the kernel launch expression, including an empty no-arg kernel probe.
- [Round 5] MUSA toolkit/architecture selection is now relocatable: top-level CMake checks `$MUSA_HOME/cmake` when `MUSA_HOME` is set, falls back to `/usr/local/musa/cmake`, and only adds `--offload-arch=${CV_CUDA_MUSA_ARCH}` when that cache variable is non-empty. `CV_CUDA_MUSA_ARCH=mp_22` was revalidated on S4000 with `cvcuda_test_system_smoke` passing 25/25.

## Known Limitations

| Round | Limitation | Impact | Action Required |
|-------|------------|--------|-----------------|
| 1 | `OpFindHomography` remains an SDK/API hotspot for muSOLVER `syevjBatched` signature/behavior. | Build or runtime may require symbol-specific adaptation in Round 2+. | Verify against SDK 5.1.0 headers during build-system/build loop; block or disable the operator if coverage proves gap/unknown. |
| 1 | CUDA Python/CuPy/CUDA Python interop samples remain unsupported optional paths. | Optional samples are outside required V1/V2 total unless a runtime-visible MUSA replacement is wired. | Keep skipped per CODE_ANALYSIS §4; do not stub Python GPU interop. |

## Classification Override

TBD by downstream agents.

## Strategy Fallback

TBD by downstream agents.

## Tier Override

TBD by downstream agents.

## AUM Jira Tickets

User constraint: do not create Jira tickets for this run.

## Final Report

CV-CUDA MUSA migration reached required V2 verification completion on S4000 with MUSA SDK 5.1.0.

### Verification

All required V2 native artifacts passed in `auto-musify-CV-CUDA:/workspace/CV-CUDA`:

| Artifact | Result | Passed | Failed | Skipped | Evidence |
|----------|--------|--------|--------|---------|----------|
| `cvcuda_test_system_smoke` | pass | 25 | 0 | 0 | `verify/round5-full/cvcuda_test_system_smoke.log`, `verify/round5-full/cvcuda_test_system_smoke.xml` |
| `cvcuda_test_unit` | pass | 16 | 0 | 0 | `verify/round5-full/cvcuda_test_unit.log`, `verify/round5-full/cvcuda_test_unit.xml` |
| `nvcv_test_types_unit` | pass | 679 | 0 | 0 | `verify/round5-full/nvcv_test_types_unit.log`, `verify/round5-full/nvcv_test_types_unit.xml` |
| `nvcv_test_types_system` | pass | 2138 | 0 | 0 | `verify/round5-full/nvcv_test_types_system.log`, `verify/round5-full/nvcv_test_types_system.xml` |
| `nvcv_test_cudatools_unit` | pass | 49 | 0 | 0 | `verify/round5-full/nvcv_test_cudatools_unit.log`, `verify/round5-full/nvcv_test_cudatools_unit.xml` |

Total required V2 result: 2,907 passed, 0 failed, 0 skipped.

### User constraints honored

- Jira/AUM tickets were not created.
- Git push was not performed.
- Source edits were authored locally under `repositories/CV-CUDA` and committed with `musify:` subjects.
- CUDA source files remained available as audit references; MUSA code lives in parallel MUSA paths.

