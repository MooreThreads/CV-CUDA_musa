# Round 4 Targeted Verification Summary

Source head: `8e2b127f`
Container: `auto-musify-CV-CUDA`
Workspace: `/workspace/CV-CUDA`

## Targeted checks after Round 4 fixes

| Check | Command scope | Result | Evidence |
|---|---|---|---|
| stream_id | `cvcuda_test_unit --gtest_filter=StreamIdTest.RegularAndDefault` | pass | `verify/round4-targeted/stream_id.log` |
| error_string | `nvcv_test_types_unit --gtest_filter=CheckErrorTest.cudaErr_to_String` | pass | `verify/round4-targeted/error_string.log` |
| tensor_subset | `nvcv_test_types_system --gtest_filter='*/TensorTests.smoke_create/*:*/TensorDataUtils.SetGetTensorFromImageVector/*'` | pass | `verify/round4-targeted/tensor_subset.log` |
| smoke_full | `cvcuda_test_system_smoke` | fail | `verify/round4-targeted/smoke_full.log` |

## Remaining blocker

`cvcuda_test_system_smoke` still fails 22 of 25 tests across BndBox, BoxBlur, and OSD. Operator creation tests pass; runtime invocation throws `NVCV_ERROR_INTERNAL: Unknown error -1` and output buffers remain unchanged.

The three Round 4 local fixes are target-verified, but the build-clean handoff is still blocked until the smoke cluster is fixed or classified with real root-cause evidence.
