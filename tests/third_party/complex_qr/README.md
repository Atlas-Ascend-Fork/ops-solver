# Official acceptance protocol header

`solver_adapter.h` is the unchanged header distributed with the October
Ascend 950 complex QR and least-squares community task's revised test package.
The official generators, verifiers and cases remain external inputs; see
[the test guide](../../../test/complex_qr/REVISED_PACKAGE.md).

Keep this file byte-for-byte identical to the official package, including CRLF
line endings. It is an external protocol definition, not authored project
implementation. The existing third-party clang-format exclusion applies here;
project-owned adapters and tests remain subject to all normal checks.
