"""
scripts/quantize_all.py
=======================
Quantize tất cả ONNX models sang INT8 để giảm kích thước cho mobile.

Strategy:
  - UNet, ControlNet, Text Encoders → INT8 dynamic quantization
    (không cần calibration data, đủ tốt cho weights-dominated models)
  - VAE Decoder → GIỮ FP16 (nhạy cảm với quantization, chỉ 160MB)
  - IP-Adapter Resampler → INT8 dynamic (nhỏ, fast)

Tại sao dynamic thay vì static?
  - Static quantization cần calibration dataset (diffusion latents phức tạp)
  - Dynamic quantization: weights offline INT8, activations FP32 at runtime
  - Với diffusion models, dynamic cho kết quả ~tương đương static về chất lượng
  - Nhưng latency speedup ít hơn static (vẫn ~1.5-2x vs CPU baseline)

Tại sao KHÔNG dùng INT4?
  - onnxruntime INT4 (matmul nbits) cần onnxruntime >= 1.17 + specific backends
  - Artifacts nặng hơn INT8 cho diffusion models
  - Thêm sau nếu INT8 vẫn quá chậm

Size estimates sau quantize:
  UNet:         5.1GB → ~1.3GB  (75% giảm)
  ControlNet:   2.5GB → ~625MB
  TextEnc 1:    500MB → ~125MB
  TextEnc 2:    900MB → ~225MB
  VAE:          160MB → 160MB   (giữ nguyên)
  IP-Adapter:   ~80MB → ~20MB
  TOTAL:        ~9.3GB → ~2.5GB ✓

Usage:
    python scripts/quantize_all.py \
        --input_dir ./onnx \
        --output_dir ./onnx-int8

    # Chỉ quantize 1 model:
    python scripts/quantize_all.py --only unet
    python scripts/quantize_all.py --only controlnet

    # Skip VAE (mặc định đã skip):
    python scripts/quantize_all.py --quantize_vae
"""

import argparse
import os
import shutil
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",    default="./onnx")
    p.add_argument("--output_dir",   default="./onnx-int8")
    p.add_argument("--only",         default=None,
                   choices=["unet", "controlnet", "text_encoder",
                            "text_encoder_2", "ip_adapter", "vae"])
    p.add_argument("--quantize_vae", action="store_true",
                   help="Quantize VAE (risky — may cause artifacts)")
    p.add_argument("--per_channel",  action="store_true", default=True,
                   help="Per-channel quantization (better quality, default=True)")
    p.add_argument("--reduce_range", action="store_true", default=False,
                   help="Reduce range (for Haswell+ CPUs, less impact on mobile)")
    p.add_argument("--skip_slim",    action="store_true", default=False,
                   help="Skip onnxslim graph optimization (saves RAM for large models)")
    p.add_argument("--slim_max_mb",  type=float, default=2048,
                   help="Auto-skip onnxslim for models larger than this (MB, default=2048)")
    return p.parse_args()


def quantize_model(input_path: str, output_path: str,
                   per_channel: bool = True, reduce_range: bool = False):
    """
    Quantize single ONNX model.

    per_channel=True: Mỗi output channel có scale/zero_point riêng.
                      Chất lượng tốt hơn, phù hợp với Conv/Linear weights.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        print(f"  [skip] {os.path.basename(output_path)} already exists")
        return

    # Tính tổng size (bao gồm external data nếu có)
    input_dir = os.path.dirname(input_path)
    input_size = sum(
        os.path.getsize(os.path.join(input_dir, f))
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
    ) / 1024**2
    print(f"  Input:  {input_size:.0f} MB (total dir)")
    print(f"  Output: {output_path}")

    t0 = time.time()

    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        reduce_range=reduce_range,
        # Quantize tất cả nodes có thể (MatMul, Conv)
        nodes_to_exclude=[],
        # Giữ một số ops ở FP32 để tránh artifacts
        op_types_to_quantize=["MatMul", "Gather", "Transpose"],
    )

    output_size = os.path.getsize(output_path) / 1024**2
    elapsed = time.time() - t0
    ratio = input_size / output_size

    print(f"  Output: {output_size:.0f} MB ({ratio:.1f}x smaller)")
    print(f"  Time:   {elapsed:.0f}s")


def onnx_slim(input_path: str, output_path: str):
    """
    Optimize ONNX graph trước khi quantize dùng onnxslim.
    Removes dead nodes, fuses ops, improves layout.
    """
    try:
        import onnxslim
        print(f"  Running onnxslim optimization...")
        onnxslim.slim(input_path, output_path)
        print(f"  Optimized")
    except ImportError:
        print(f"  [skip] onnxslim not installed — pip install onnxslim")
        shutil.copy2(input_path, output_path)
    except Exception as e:
        print(f"  [warn] onnxslim failed: {e} — using original")
        shutil.copy2(input_path, output_path)


def main():
    args = parse_args()

    try:
        from onnxruntime.quantization import quantize_dynamic
    except ImportError:
        print("ERROR: onnxruntime not installed")
        print("Run: pip install onnxruntime")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Define components to quantize ─────────────────────────────────────
    components = [
        {
            "name": "unet",
            "input": os.path.join(args.input_dir, "unet", "model.onnx"),
            "output": os.path.join(args.output_dir, "unet", "model_int8.onnx"),
            "quantize": True,
            "description": "UNet (largest component, ~5GB → ~1.3GB)",
        },
        {
            "name": "controlnet",
            "input": os.path.join(args.input_dir, "controlnet", "model.onnx"),
            "output": os.path.join(args.output_dir, "controlnet", "model_int8.onnx"),
            "quantize": True,
            "description": "ControlNet IdentityNet (~2.5GB → ~625MB)",
        },
        {
            "name": "text_encoder",
            "input": os.path.join(args.input_dir, "text_encoder", "model.onnx"),
            "output": os.path.join(args.output_dir, "text_encoder", "model_int8.onnx"),
            "quantize": True,
            "description": "Text Encoder 1 CLIP-L (~500MB → ~125MB)",
        },
        {
            "name": "text_encoder_2",
            "input": os.path.join(args.input_dir, "text_encoder_2", "model.onnx"),
            "output": os.path.join(args.output_dir, "text_encoder_2", "model_int8.onnx"),
            "quantize": True,
            "description": "Text Encoder 2 OpenCLIP-G (~900MB → ~225MB)",
        },
        {
            "name": "ip_adapter",
            "input": os.path.join(args.input_dir, "ip_adapter", "model.onnx"),
            "output": os.path.join(args.output_dir, "ip_adapter", "model_int8.onnx"),
            "quantize": True,
            "description": "IP-Adapter Resampler (~80MB → ~20MB)",
        },
        {
            "name": "vae",
            "input": os.path.join(args.input_dir, "vae_decoder", "model.onnx"),
            "output": os.path.join(args.output_dir, "vae_decoder", "model_fp16.onnx"),
            "quantize": args.quantize_vae,
            "description": "VAE Decoder — keeping FP16 (sensitive to quantization)",
        },
    ]

    # ── Filter by --only ───────────────────────────────────────────────────
    if args.only:
        components = [c for c in components if c["name"] == args.only]
        if not components:
            print(f"ERROR: unknown component '{args.only}'")
            return

    print(f"\nQuantization config:")
    print(f"  Input:       {args.input_dir}")
    print(f"  Output:      {args.output_dir}")
    print(f"  Per-channel: {args.per_channel}")
    print(f"  Reduce range:{args.reduce_range}")

    total_input_mb = 0
    total_output_mb = 0

    for comp in components:
        print(f"\n{'─'*50}")
        print(f"[{comp['name']}] {comp['description']}")

        if not os.path.exists(comp["input"]):
            print(f"  [SKIP] Input not found: {comp['input']}")
            print(f"         Run: python scripts/export_all.py --only {comp['name']}")
            continue

        # Tính tổng size bao gồm external data (.pb files)
        input_dir = os.path.dirname(comp["input"])
        input_mb = sum(
            os.path.getsize(os.path.join(input_dir, f))
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ) / 1024**2
        total_input_mb += input_mb

        if comp["quantize"]:
            # Step 1: Slim/optimize graph (skip for large models to avoid OOM)
            skip_slim = args.skip_slim or (input_mb > args.slim_max_mb)
            if skip_slim:
                slim_path = comp["input"]  # quantize trực tiếp từ file gốc
                print(f"  Step 1/2: Graph optimization... [SKIP — {input_mb:.0f}MB > {args.slim_max_mb:.0f}MB threshold]")
            else:
                slim_path = comp["input"].replace(".onnx", "_slim.onnx")
                print(f"  Step 1/2: Graph optimization...")
                onnx_slim(comp["input"], slim_path)

            # Step 2: Quantize
            print(f"  Step 2/2: INT8 quantization...")
            try:
                quantize_model(
                    slim_path,
                    comp["output"],
                    per_channel=args.per_channel,
                    reduce_range=args.reduce_range,
                )
                # Cleanup slim temp file
                if not skip_slim and os.path.exists(slim_path) and slim_path != comp["input"]:
                    os.remove(slim_path)
            except Exception as e:
                print(f"  [ERROR] Quantization failed: {e}")
                print(f"  → Copying original as fallback")
                shutil.copy2(comp["input"], comp["output"])

        else:
            # VAE: copy FP16, chỉ slim
            print(f"  Optimizing graph (no quantization)...")
            onnx_slim(comp["input"], comp["output"])

        if os.path.exists(comp["output"]):
            output_mb = os.path.getsize(comp["output"]) / 1024**2
            total_output_mb += output_mb

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'═'*50}")
    print(f"✅ Quantization complete!")
    print(f"\n  Total input:  {total_input_mb:.0f} MB ({total_input_mb/1024:.1f} GB)")
    print(f"  Total output: {total_output_mb:.0f} MB ({total_output_mb/1024:.1f} GB)")
    if total_input_mb > 0:
        print(f"  Reduction:    {(1 - total_output_mb/total_input_mb)*100:.0f}%")

    print(f"\nOutput files:")
    for root, _, files in os.walk(args.output_dir):
        for f in sorted(files):
            if f.endswith(".onnx"):
                p = os.path.join(root, f)
                mb = os.path.getsize(p) / 1024**2
                rel = os.path.relpath(p, args.output_dir)
                print(f"  {rel}: {mb:.0f} MB")

    print(f"\nNext: python scripts/test_onnx_pipeline.py --onnx_dir {args.output_dir}")


if __name__ == "__main__":
    main()
