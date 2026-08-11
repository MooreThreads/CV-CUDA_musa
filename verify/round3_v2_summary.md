# Round 3 V2 Verification Summary

Source: `CODE_ANALYSIS.md §4 Native Verification Discovery`
Container: `auto-musify-CV-CUDA`
Workspace: `/workspace/CV-CUDA`
Source head: `438b24d6`

## Required artifacts

| Artifact | Result | Passed | Failed | Skipped | Evidence |
|---|---:|---:|---:|---:|---|
| `cvcuda_test_system_smoke` | fail | 3 | 22 | 0 | `verify/cvcuda_test_system_smoke.log`, `verify/cvcuda_test_system_smoke.xml` |
| `cvcuda_test_unit` | fail | 15 | 1 | 1 | `verify/cvcuda_test_unit.log`, `verify/cvcuda_test_unit.xml` |
| `nvcv_test_types_unit` | fail | 678 | 1 | 0 | `verify/nvcv_test_types_unit.log`, `verify/nvcv_test_types_unit.xml` |
| `nvcv_test_types_system` | fail | 2132 | 6 | 1 | `verify/nvcv_test_types_system.log`, `verify/nvcv_test_types_system.xml` |
| `nvcv_test_cudatools_unit` | pass | 49 | 0 | 0 | `verify/nvcv_test_cudatools_unit.log`, `verify/nvcv_test_cudatools_unit.xml` |

Overall required artifacts: 5 total, 1 passed, 4 failed.

## Fixable bug clusters

1. `issues/round3-cvcuda-smoke-osd-runtime-internal-error.yaml` — BndBox, BoxBlur, and OSD runtime calls throw `NVCV_ERROR_INTERNAL: Unknown error -1`.
2. `issues/round3-stream-id-invalid-default-stream.yaml` — `StreamIdTest.RegularAndDefault` throws `NVCV_ERROR_INVALID_ARGUMENT: Invalid stream handle`.
3. `issues/round3-types-error-string-cuda-residual.yaml` — `CheckErrorTest.cudaErr_to_String` fails after MUSA port.
4. `issues/round3-types-system-tensor-stride-failures.yaml` — tensor creation and TensorDataUtils image-vector cases fail for selected layouts/formats.

## Known limitations

No new known limitation was classified in this verifier pass. The failures above are routed as fixable bugs because the required artifacts executed and produced concrete runtime/test failures.
