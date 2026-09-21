Vendored ONNX Runtime headers from revision `f2c39fe` (API 30), matching the
bundled Windows runtime `1.30.0.20260909.4.f2c39fe`.

Source: https://github.com/microsoft/onnxruntime/tree/f2c39fe/include/onnxruntime/core/session

All quoted header dependencies and the upstream MIT license are included.
`node/` contains the Node.js v24.11.1 Node-API headers and license; the binding
targets stable N-API 8. `../lib/win32/{x64,arm64}/node.lib` comes from
https://nodejs.org/dist/v24.11.1/ and supplies Windows import symbols only.

The build does not download headers, libraries, or ONNX Runtime source.
