# Optional official task package tests

The default repository tests do not require an external task package. See
[README.md](README.md) for the in-tree suite and build instructions.

To additionally run the community task's official accuracy suite, obtain its
unchanged package from the task publisher. Set `COMPLEX_QR_CASES_ROOT` to the
absolute directory containing `cgeqrf/package`, `cungqr/package`,
`cunmqr/package` and `cgels/package`. Each package must include its generator,
case index, references and `verify_accuracy.py`. Do not edit those assets.

```bash
export COMPLEX_QR_CASES_ROOT=/absolute/path/to/official-task
export COMPLEX_QR_RESULTS=/absolute/path/to/results
export NPU_DEVICE_ID=0
export PYTHON=python3
bash test/cgeqrf/run_tests.sh /absolute/path/to/build
bash test/cungqr/run_tests.sh /absolute/path/to/build
bash test/cunmqr/run_tests.sh /absolute/path/to/build
bash test/ccgels/run_tests.sh /absolute/path/to/build
```

Build the four adapters or the shared `complex_qr` test target first. Run Cungqr
before Cunmqr. The runner executes the original generator, checks repeatability
and adapter output, then invokes the original verifier. A supplied invalid
package path is an error; it does not silently fall back to supplementary tests.
Without `COMPLEX_QR_CASES_ROOT`, each operator entry runs only its own independent numerical regression. Unsupported architectures run the Host ABI
ARCH_MISMATCH checks instead of starting NPU kernels.

Use the environment, complete case set, iteration count, original profiling
tool and precision/performance thresholds specified by the task package. Keep
the original case and reference files unchanged. Accuracy success does not
establish performance success. Store execution logs, reports and source/tool
identities outside the source tree; do not commit machine-specific acceptance
records to this repository.
