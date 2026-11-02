# Complex QR tests

Default tests use the in-tree suite and require no external task package.
Use [REVISED_PACKAGE.md](REVISED_PACKAGE.md) for optional official task tests.
The required matrix contains 694 numerical cases, 12 info probes and 694
performance baselines. Keep official generators, cases, references, verifiers
and thresholds unchanged. Supplementary tests do not replace them.

## Environment and build

Use the task-specified Ascend 950PR and CANN 9.0.0 build for development and
testing. The revised package records Python 3.12.11, NumPy 2.3.3, SciPy 1.16.2
and 64 OpenBLAS threads. Record the actual loaded BLAS libraries with
`threadpoolctl.threadpool_info()`; package versions alone are insufficient.
Source the selected CANN installation's `set_env.sh` first.

```bash
export OPENBLAS_NUM_THREADS=64
cmake -S . -B build -DSOC_VERSION=ascend950 -DCMAKE_BUILD_TYPE:STRING= -DBUILD_TEST=ON -DTEST_NAMES=complex_qr
cmake --build build -j8
export LD_LIBRARY_PATH="$PWD/build:$LD_LIBRARY_PATH"
```

The recorded build uses an empty `CMAKE_BUILD_TYPE`, not Release. Executables
and adapters are under `build/test/complex_qr/`. Other architectures build the
eight-symbol Host ABI test, checking ARCH_MISMATCH, null handles and untouched outputs.

## Official tests

Follow the revised-package guide to run `package_accuracy.py` with the original
`package/gen_data.py`, then the original `verify_accuracy.py`. Use `--perf-inputs`
to export identical column-major inputs and their SHA256 identities for
profiling. Include the four acceptance adapters with `--adapters`. Full runs
must omit development case filters.

Performance requires original `msprof op` / `OpBasicInfo.csv` measurements.
For each case and kernel name, calculate the arithmetic mean of
`Task Duration(us)` using the actual invocation count. Preserve all 25 official
iterations. Every case/kernel must satisfy GPU/NPU >= 0.35.

## Supplementary tests

```bash
python test/complex_qr/smoke.py --library build/libops_solver.so --device 0
python test/complex_qr/contracts.py --library build/libops_solver.so --device 0
python test/complex_qr/blocked_apply_smoke.py --library build/libops_solver.so --device 0
```

`contracts.py` checks invalid arguments, workspace overflow, nonfinite inputs,
empty problems and singular-solve behavior. `blocked_apply_smoke.py` has 23
cases covering LEFT/RIGHT, N/C, partial panels, padding, multiple RHS and repeated
output identity. `--task-boundaries` expands it to 51 cases including +8/+32
padding, powers of two and powers minus one, RHS 1/8/32/128 and large problems.

`memory_bounds.py --library build/libops_solver.so --device 0` checks allocation
prefix/suffix guards, padding and read-only inputs in 21 supplementary cases.
These guards do not prove absence of out-of-bounds reads or races. Task section
3.4 has no memory acceptance requirement; report that item accordingly.

The repository's `build.sh --soc=ascend950 --ops=complex_qr --run` entry recreates
`build/`. To keep a frozen build, use
`bash test/complex_qr/run_tests.sh /absolute/path/to/build` instead. `PYTHON`
selects the interpreter and `NPU_DEVICE_ID` selects the device. Use an exclusive
device for profiling; do not run supplementary tests concurrently with it.

The standard runner includes all 51 boundary cases, late-element finite-input
validation and allocation guards, in addition to contracts and smoke tests.
`ctest --test-dir build --output-on-failure -R complex_qr` runs the same suite,
with a shared NPU resource lock and a 30-minute timeout. On unsupported targets
it runs only the Host ABI contract test and never starts an NPU kernel.

`../common/complex_qr/numerical_checks.py` provides the shared numerical checks used by the
supplementary boundary and memory guard suites. These generated inputs do not
replace the unchanged official package and its original verifier.

## Test ownership

Each `test/{cgeqrf,cungqr,cunmqr,ccgels}/test_<operator>.py` is an independent API regression.
The Q and reflector fixtures for Cungqr/Cunmqr come from CPU LAPACK, not another NPU operator.
This directory retains integrated QR reconstruction/application and least-squares checks,
contracts, boundaries and memory guards; each is invoked by the standard test runner.
The shared ACL runtime and numerical helpers live in `test/common/complex_qr`.
The obsolete development JSON-driver CLI has been removed; official package execution
and the unchanged task verifier remain available through `run_package.sh`.
