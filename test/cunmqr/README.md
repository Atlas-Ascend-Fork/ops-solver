# Cunmqr acceptance adapter

`cunmqr_adapter.cpp` exports the task's fixed solver adapter v2 protocol and calls
the real `aclsolverCunmqr` Device column-major API. Transport buffers are compact
Host row-major complex64. Inputs are packed reflectors, tau and C; output is the
transformed C and actual device info. LEFT/RIGHT and N/C are preserved. Input
QR preparation is outside execute and outside target kernel timing.

The unchanged protocol header and shared transport implementation are in
`../../tests/third_party/complex_qr/solver_adapter.h` and `../complex_qr/adapter_impl.h`.
Build with `cmake -S . -B build -DSOC_VERSION=ascend950 -DBUILD_TEST=ON -DTEST_NAMES=cunmqr`
and `cmake --build build -j8`. The adapter is `build/test/cunmqr/libcunmqr_adapter.so`.
Selecting `TEST_NAMES=complex_qr` builds all four adapters under `test/complex_qr`.

See [revised official package instructions](../complex_qr/REVISED_PACKAGE.md).

## Independent regression

`test_cunmqr.py` exercises only the `cunmqr` public API with CPU references, padded layouts and numerical checks.
The default `run_tests.sh BUILD_DIR` runs this test, not the complete QR pipeline.
Use `ctest --test-dir build -R cunmqr_independent --output-on-failure`.
`test/complex_qr` contains cross-operator integration, shared contract and boundary regressions.
