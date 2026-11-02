# CCgels acceptance adapter

`ccgels_adapter.cpp` exports the task's fixed solver adapter v2 protocol and calls
the real `aclsolverCCgels` Device column-major API. The protocol operator name is
`cgels`, matching the official package. Transport buffers are compact Host
row-major complex64. Inputs are A and B; output is independent X and actual
device info. Workspace uses bytes, and the API receives a real Host niter.

The unchanged protocol header and shared transport implementation are in
`../../tests/third_party/complex_qr/solver_adapter.h` and `../complex_qr/adapter_impl.h`.
Build with `cmake -S . -B build -DSOC_VERSION=ascend950 -DBUILD_TEST=ON -DTEST_NAMES=ccgels`
and `cmake --build build -j8`. The adapter is `build/test/ccgels/libccgels_adapter.so`.
Selecting `TEST_NAMES=complex_qr` builds all four adapters under `test/complex_qr`.

See [revised official package instructions](../complex_qr/REVISED_PACKAGE.md).

## Independent regression

`test_ccgels.py` exercises only the `ccgels` public API with CPU references, padded layouts and numerical checks.
The default `run_tests.sh BUILD_DIR` runs this test, not the complete QR pipeline.
Use `ctest --test-dir build -R ccgels_independent --output-on-failure`.
`test/complex_qr` contains cross-operator integration, shared contract and boundary regressions.
