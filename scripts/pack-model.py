#!/usr/bin/env python
"""Pack an exported model directory for distribution (in place of its weights.bin):

* ternary BitLinear weights ({-1, 0, 1} int8) -> 2 bits each, 4 per byte (lossless; 4x smaller);
* the large fp32 matrices (token embeddings, choice-head weights) -> fp16;
* model-cpu.onnx / model-gpu.onnx unpack both with constant-foldable ops (BitShift, Mod, Cast), so ONNX
  Runtime restores the original tensors once, at session load;
* config.json gets the new tensor index for the native runtime, which unpacks the same way.

    python scripts/pack-model.py models/bitnet-decision models/bitnet-decision-packed
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

FP16 = (".weight.T", ".kv.T", ".q.T")  # choice-head matrices (x @ W layout) ...
FP16_NAMES = {"embed"}                  # ... and the token embeddings


def pack_ternary(t):
    """int8 [K, N] in {-1, 0, 1} -> uint8 [K, N/4]; column 4j+s sits in bits 2s..2s+1 of byte j."""
    v = (t.astype(np.int16) + 1).astype(np.uint8).reshape(t.shape[0], -1, 4)
    return v[..., 0] | v[..., 1] << 2 | v[..., 2] << 4 | v[..., 3] << 6


def transform(model, packed):
    """Replace initializers by their packed form plus the nodes that restore them under the old name."""
    graph = model.graph
    keep, nodes = [], []
    consts = {
        "pack.shifts": numpy_helper.from_array(np.array([0, 2, 4, 6], np.uint8), "pack.shifts"),
        "pack.four": numpy_helper.from_array(np.array(4, np.uint8), "pack.four"),
        "pack.one": numpy_helper.from_array(np.array(1, np.int8), "pack.one"),
        "pack.axis2": numpy_helper.from_array(np.array([2], np.int64), "pack.axis2"),
    }
    used = set()
    for init in graph.initializer:
        name = init.name
        if name not in packed:
            keep.append(init)
            continue
        kind, tensor = packed[name]
        keep.append(tensor)
        if kind == "ternary":
            k, n = init.dims
            shape = f"{name}.shape"
            consts[shape] = numpy_helper.from_array(np.array([k, n], np.int64), shape)
            nodes += [
                helper.make_node("Unsqueeze", [tensor.name, "pack.axis2"], [f"{name}.u"]),
                helper.make_node("BitShift", [f"{name}.u", "pack.shifts"], [f"{name}.s"], direction="RIGHT"),
                helper.make_node("Mod", [f"{name}.s", "pack.four"], [f"{name}.m"]),
                helper.make_node("Reshape", [f"{name}.m", shape], [f"{name}.r"]),
                helper.make_node("Cast", [f"{name}.r"], [f"{name}.c"], to=TensorProto.INT8),
                helper.make_node("Sub", [f"{name}.c", "pack.one"], [name]),
            ]
            used |= {"pack.shifts", "pack.four", "pack.one", "pack.axis2", shape}
        else:
            nodes.append(helper.make_node("Cast", [tensor.name], [name], to=TensorProto.FLOAT))
    keep += [consts[c] for c in sorted(used)]
    del graph.initializer[:]
    graph.initializer.extend(keep)
    old = list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes + old)


def tensor_index(onnx_path):
    index = {}
    for t in onnx.load(onnx_path, load_external_data=False).graph.initializer:
        if t.name.startswith(("const_", "pack.")) or t.name.endswith(".shape"):
            continue
        entry = {"dtype": TensorProto.DataType.Name(t.data_type).lower(), "shape": list(t.dims)}
        external = {e.key: e.value for e in t.external_data}
        if external:
            entry.update(offset=int(external.get("offset", 0)), length=int(external["length"]))
        else:
            entry["data"] = numpy_helper.to_array(t).astype(np.float64).ravel().tolist()
        index[t.name] = entry
    return index


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    if dst.exists():
        sys.exit(f"{dst} exists")
    dst.mkdir(parents=True)
    cpu = onnx.load(src / "model-cpu.onnx")  # with external data
    packed = {}
    for init in cpu.graph.initializer:
        if init.name.endswith(".ternary"):
            t = numpy_helper.to_array(init)
            assert t.dtype == np.int8 and set(np.unique(t)) <= {-1, 0, 1} and t.shape[1] % 4 == 0
            packed[init.name] = ("ternary", numpy_helper.from_array(pack_ternary(t), init.name + ".2bit"))
        elif init.data_type == TensorProto.FLOAT and (init.name in FP16_NAMES or init.name.endswith(FP16)):
            packed[init.name] = ("fp16", numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name + ".f16"))
    transform(cpu, packed)
    onnx.save_model(cpu, dst / "model-cpu.onnx", save_as_external_data=True, location="weights.bin",
                    all_tensors_to_one_file=True, size_threshold=4096)
    stored = {t.name: t for t in onnx.load(dst / "model-cpu.onnx", load_external_data=False).graph.initializer}
    gpu = onnx.load(src / "model-gpu.onnx", load_external_data=False)
    transform(gpu, {name: (kind, stored[t.name]) for name, (kind, t) in packed.items()
                    if any(i.name == name for i in gpu.graph.initializer)})
    for tensor in gpu.graph.initializer:
        tensor.CopyFrom(stored[tensor.name])
    onnx.save_model(gpu, dst / "model-gpu.onnx")
    config = json.loads((src / "config.json").read_text(encoding="utf-8"))
    config["tensors"] = tensor_index(dst / "model-cpu.onnx")
    (dst / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    shutil.copyfile(src / "tokenizer.json", dst / "tokenizer.json")
    print(json.dumps({"weights_mb": round((dst / "weights.bin").stat().st_size / 2**20, 1),
                      "ternary": sum(k == "ternary" for k, _ in packed.values()),
                      "fp16": sum(k == "fp16" for k, _ in packed.values())}))


if __name__ == "__main__":
    main()
