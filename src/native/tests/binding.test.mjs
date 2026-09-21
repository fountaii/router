import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';

const require = createRequire(import.meta.url);
const binding = require(`../../onnx/bin/${process.platform}/${process.arch}/onnxruntime_binding.node`);
class Tensor {
  constructor(type, data, dims) {
    Object.assign(this, { type, data, dims, location: 'cpu' });
  }
}
binding.initOrtOnce(2, Tensor, true);

function session(type) {
  const result = new binding.InferenceSession();
  result.loadModel(fileURLToPath(new URL(`fixtures/${type}.onnx`, import.meta.url)), {
    executionProviders: ['cpu'], intraOpNumThreads: 1,
  });
  return result;
}

test('reports cpu first and only backends the binding can append', () => {
  const backends = binding.listSupportedBackends();
  assert.deepEqual(backends[0], { name: 'cpu', bundled: true });
  // cuda so aparece quando a onnxruntime.dll ao lado e a build de GPU
  for (const backend of backends.slice(1)) {
    assert.ok(['cuda'].includes(backend.name), `backend inesperado: ${backend.name}`);
    assert.strictEqual(backend.bundled, true);
  }
});

for (const [type, data] of [
  ['float32', new Float32Array([99, 1.5, -2.25, 88]).subarray(1, 3)],
  ['int64', new BigInt64Array([99n, 9007199254740993n, -7n, 88n]).subarray(1, 3)],
]) {
  test(`${type}: offset input, metadata, repeated runs, ownership after disposal`, () => {
    const s = session(type);
    assert.deepEqual(s.inputMetadata, [{ name: 'input' }]);
    assert.deepEqual(s.outputMetadata, [{ name: 'output' }]);
    let output;
    try {
      for (let i = 0; i < 3; i++) {
        output = s.run({ input: new Tensor(type, data, [2]) }, { output: null }, {}).output;
        assert.equal(output.type, type);
        assert.deepEqual(output.dims, [2]);
        assert.deepEqual([...output.data], [...data]);
        assert.notEqual(output.data.buffer, data.buffer);
      }
    } finally {
      s.dispose();
    }
    assert.deepEqual([...output.data], [...data]);
    assert.throws(() => s.run({}, {}, {}), /disposed/);
  });
}

test('invalid tensor inputs throw without crashing the process', () => {
  const s = session('float32');
  try {
    const input = new Tensor('float32', new Float32Array([1, 2]), [2]);
    for (const patch of [
      { dims: [3] }, { dims: [-1] }, { dims: [NaN] }, { dims: [Infinity] },
      { type: 'string' }, { data: new BigInt64Array([1n, 2n]) }, { location: 'gpu-buffer' },
    ]) {
      assert.throws(() => s.run({ input: { ...input, ...patch } }, { output: null }, {}));
    }
    assert.deepEqual([...s.run({ input }, { output: null }, {}).output.data], [1, 2]);
  } finally {
    s.dispose();
  }
});
