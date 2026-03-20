"""
scripts/convert_for_onnxstream.py
==================================
Convert ONNX models sang format mà OnnxStream cần.

OnnxStream KHÔNG đọc file .onnx trực tiếp.
Nó cần weights được split thành các .bin chunks nhỏ + metadata .yaml.

Tại sao OnnxStream cần format riêng?
  - OnnxStream stream weights từ disk thay vì load toàn bộ vào RAM
  - Cho phép chạy model 5GB trong <500MB RAM (giống paging/virtual memory)
  - Mỗi chunk ~ vài MB, load khi cần, unload ngay sau khi dùng

Format output:
  onnxstream-models/
    unet/
      model.yaml         ← graph structure + operator list
      weights_000.bin    ← chunk 0 (mỗi chunk ~16MB default)
      weights_001.bin
      ...
    controlnet/
      model.yaml
      weights_000.bin
      ...
    text_encoder/  ...
    vae_decoder/   ...

Tham khảo OnnxStream format:
  https://github.com/vitoplantamura/OnnxStream/blob/master/src/README.md

Usage:
    python scripts/convert_for_onnxstream.py \
        --input_dir ./onnx-int8 \
        --output_dir ./onnxstream-models \
        --chunk_size_mb 16

    # Chỉ convert 1 model:
    python scripts/convert_for_onnxstream.py --only unet
"""

import argparse
import os
import struct
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",     default="./onnx-int8")
    p.add_argument("--output_dir",    default="./onnxstream-models")
    p.add_argument("--chunk_size_mb", type=int, default=16,
                   help="Size of each weight chunk in MB")
    p.add_argument("--only",          default=None,
                   choices=["unet", "controlnet", "text_encoder",
                            "text_encoder_2", "vae_decoder", "ip_adapter"])
    return p.parse_args()


# ─── OnnxStream weight serialization ─────────────────────────────────────────

# OnnxStream weight file format (little-endian):
#   Header per tensor:
#     [4 bytes] name_len: uint32
#     [name_len bytes] name: utf-8 string
#     [4 bytes] dtype: uint32  (0=float32, 1=float16, 2=int8, 3=int4)
#     [4 bytes] ndim: uint32
#     [ndim * 4 bytes] shape: uint32 array
#     [4 bytes] data_len: uint32 (bytes)
#     [data_len bytes] data: raw bytes

DTYPE_MAP = {
    "float32": 0,
    "float16": 1,
    "int8":    2,
    "int4":    3,  # packed 2 values per byte
}

DTYPE_ITEMSIZE = {
    "float32": 4,
    "float16": 2,
    "int8":    1,
    "int4":    0.5,  # 4 bits per element
}


def serialize_tensor(name: str, data: np.ndarray) -> bytes:
    """Serialize tensor to OnnxStream binary format."""
    # Map numpy dtype to OnnxStream dtype code
    dtype_str = {
        np.float32: "float32",
        np.float16: "float16",
        np.int8:    "int8",
    }.get(data.dtype.type, "float32")

    if dtype_str not in DTYPE_MAP:
        # Convert to float32 as fallback
        data = data.astype(np.float32)
        dtype_str = "float32"

    dtype_code = DTYPE_MAP[dtype_str]

    name_bytes = name.encode("utf-8")
    name_len = len(name_bytes)
    ndim = data.ndim
    shape = list(data.shape)
    raw_data = data.tobytes()
    data_len = len(raw_data)

    # Pack header
    header = struct.pack("<I", name_len)
    header += name_bytes
    header += struct.pack("<I", dtype_code)
    header += struct.pack("<I", ndim)
    header += struct.pack(f"<{ndim}I", *shape)
    header += struct.pack("<I", data_len)

    return header + raw_data


def extract_weights_from_onnx(onnx_path: str) -> Dict[str, np.ndarray]:
    """
    Extract tất cả initializer (weight) tensors từ ONNX model.
    Returns dict: name → numpy array
    """
    import onnx
    from onnx import numpy_helper

    print(f"  Loading {onnx_path} ({os.path.getsize(onnx_path)/1024**2:.0f} MB)...")
    model = onnx.load(onnx_path)

    weights = {}
    for initializer in model.graph.initializer:
        name = initializer.name
        array = numpy_helper.to_array(initializer)
        weights[name] = array

    print(f"  Found {len(weights)} weight tensors")
    return weights


def extract_graph_ops(onnx_path: str) -> List[dict]:
    """
    Extract operator graph structure (nodes, inputs, outputs).
    Returns list of node descriptors.
    """
    import onnx

    model = onnx.load(onnx_path)
    nodes = []
    for node in model.graph.node:
        nodes.append({
            "op_type": node.op_type,
            "name": node.name,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "attributes": {
                attr.name: _extract_attr(attr)
                for attr in node.attribute
            }
        })
    return nodes


def _extract_attr(attr):
    """Extract ONNX attribute value."""
    import onnx
    if attr.type == onnx.AttributeProto.FLOAT:
        return attr.f
    elif attr.type == onnx.AttributeProto.INT:
        return attr.i
    elif attr.type == onnx.AttributeProto.STRING:
        return attr.s.decode("utf-8")
    elif attr.type == onnx.AttributeProto.FLOATS:
        return list(attr.floats)
    elif attr.type == onnx.AttributeProto.INTS:
        return list(attr.ints)
    return None


def convert_to_onnxstream(onnx_path: str, output_dir: str,
                           chunk_size_bytes: int = 16 * 1024 * 1024):
    """
    Convert single ONNX model sang OnnxStream format.

    Steps:
    1. Extract weights từ ONNX initializers
    2. Extract graph structure (nodes + tensor shapes)
    3. Split weights thành chunks
    4. Write weights_NNN.bin files
    5. Write model.yaml (graph + weight index)
    """
    import yaml

    os.makedirs(output_dir, exist_ok=True)

    # ── Extract weights ───────────────────────────────────────────────────
    weights = extract_weights_from_onnx(onnx_path)

    total_bytes = sum(w.nbytes for w in weights.values())
    print(f"  Total weight data: {total_bytes/1024**2:.0f} MB")
    print(f"  Chunk size: {chunk_size_bytes/1024**2:.0f} MB")
    expected_chunks = max(1, total_bytes // chunk_size_bytes)
    print(f"  Expected chunks: ~{expected_chunks}")

    # ── Write weight chunks ───────────────────────────────────────────────
    chunk_idx = 0
    current_chunk = b""
    weight_index = {}  # name → (chunk_idx, byte_offset)

    for name, array in weights.items():
        tensor_bytes = serialize_tensor(name, array)

        # Nếu current chunk đầy, flush và start chunk mới
        if len(current_chunk) + len(tensor_bytes) > chunk_size_bytes and current_chunk:
            chunk_path = os.path.join(output_dir, f"weights_{chunk_idx:04d}.bin")
            with open(chunk_path, "wb") as f:
                f.write(current_chunk)
            chunk_idx += 1
            current_chunk = b""

        # Record index: tensor ở chunk nào, offset bao nhiêu
        weight_index[name] = {
            "chunk": chunk_idx,
            "offset": len(current_chunk),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
        current_chunk += tensor_bytes

    # Flush last chunk
    if current_chunk:
        chunk_path = os.path.join(output_dir, f"weights_{chunk_idx:04d}.bin")
        with open(chunk_path, "wb") as f:
            f.write(current_chunk)
        chunk_idx += 1

    print(f"  Written {chunk_idx} weight chunks")

    # ── Extract graph structure ────────────────────────────────────────────
    nodes = extract_graph_ops(onnx_path)

    # ── Write model.yaml ──────────────────────────────────────────────────
    import onnx
    model = onnx.load(onnx_path)

    # Graph inputs/outputs
    graph_inputs = []
    for inp in model.graph.input:
        # Skip initializers (weight tensors are inputs too in ONNX)
        if inp.name not in weights:
            type_proto = inp.type.tensor_type
            shape = [
                d.dim_value if d.dim_value > 0 else d.dim_param
                for d in type_proto.shape.dim
            ]
            graph_inputs.append({"name": inp.name, "shape": shape,
                                   "dtype": type_proto.elem_type})

    graph_outputs = []
    for out in model.graph.output:
        type_proto = out.type.tensor_type
        shape = [
            d.dim_value if d.dim_value > 0 else d.dim_param
            for d in type_proto.shape.dim
        ]
        graph_outputs.append({"name": out.name, "shape": shape,
                                "dtype": type_proto.elem_type})

    yaml_data = {
        "version": "1",
        "num_chunks": chunk_idx,
        "chunk_size_mb": chunk_size_bytes // (1024 * 1024),
        "graph": {
            "inputs": graph_inputs,
            "outputs": graph_outputs,
            "num_nodes": len(nodes),
        },
        "weight_index": weight_index,
        # Note: full node list saved separately (may be large)
    }

    yaml_path = os.path.join(output_dir, "model.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

    # Save full node graph as separate JSON (untuk C++ loading)
    import json
    nodes_path = os.path.join(output_dir, "graph_nodes.json")
    with open(nodes_path, "w") as f:
        json.dump(nodes, f, indent=2)

    print(f"  Written model.yaml ({len(graph_inputs)} inputs, {len(graph_outputs)} outputs)")
    print(f"  Written graph_nodes.json ({len(nodes)} nodes)")

    # ── Summary ───────────────────────────────────────────────────────────
    total_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
    )
    print(f"  Total output: {total_size/1024**2:.0f} MB in {output_dir}/")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    try:
        import onnx
        import yaml
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Run: pip install onnx pyyaml")
        return

    chunk_bytes = args.chunk_size_mb * 1024 * 1024

    # Components: (input onnx file, output subdir)
    components = [
        ("unet",         "unet/model_int8.onnx",         "unet"),
        ("controlnet",   "controlnet/model_int8.onnx",   "controlnet"),
        ("text_encoder", "text_encoder/model_int8.onnx", "text_encoder"),
        ("text_encoder_2","text_encoder_2/model_int8.onnx","text_encoder_2"),
        ("ip_adapter",   "ip_adapter/model_int8.onnx",   "ip_adapter"),
        # VAE: try int8, fallback to fp16
        ("vae_decoder",  "vae_decoder/model_fp16.onnx",  "vae_decoder"),
    ]

    if args.only:
        components = [(n, i, o) for n, i, o in components if n == args.only]

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nConverting ONNX → OnnxStream format")
    print(f"  Input dir:  {args.input_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Chunk size: {args.chunk_size_mb} MB")

    for comp_name, input_rel, output_subdir in components:
        input_path = os.path.join(args.input_dir, input_rel)

        # Try int8 first, then fp16, then fp32
        if not os.path.exists(input_path):
            alt = input_path.replace("_int8.onnx", "_fp16.onnx")
            if os.path.exists(alt):
                input_path = alt
            else:
                alt = input_path.replace("_int8.onnx", ".onnx")
                if os.path.exists(alt):
                    input_path = alt
                else:
                    print(f"\n[{comp_name}] SKIP — not found: {input_path}")
                    continue

        output_path = os.path.join(args.output_dir, output_subdir)

        print(f"\n[{comp_name}]")
        try:
            convert_to_onnxstream(input_path, output_path, chunk_bytes)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'═'*50}")
    print(f"✅ OnnxStream conversion complete!")
    print(f"\nModel directory structure:")
    for root, dirs, files in os.walk(args.output_dir):
        depth = root.replace(args.output_dir, "").count(os.sep)
        indent = "  " * depth
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = "  " * (depth + 1)
        for f in sorted(files):
            fpath = os.path.join(root, f)
            mb = os.path.getsize(fpath) / 1024**2
            print(f"{sub_indent}{f}: {mb:.1f} MB")

    print(f"\nCopy {args.output_dir}/ to Android/iOS device storage")
    print(f"Then build mobile app with mobile/cpp/ code")


if __name__ == "__main__":
    main()
