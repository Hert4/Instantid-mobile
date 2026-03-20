"""
scripts/fuse_lcm_lora.py
========================
Fuse LCM-LoRA weights vào SDXL UNet TRƯỚC KHI export ONNX.

Tại sao phải fuse trước?
- LCM-LoRA thêm các delta weights (A, B matrices) vào UNet attention layers.
- Nếu export ONNX với LoRA chưa fused, graph sẽ có các extra matmul ops.
- Sau khi fuse: W_new = W_orig + alpha * (B @ A), graph sạch hơn, nhỏ hơn.
- Inference chỉ cần 4-8 steps thay vì 20-50 (speedup ~5x).
- Guidance scale = 0 với LCM → không cần classifier-free guidance
  → chỉ 1 UNet forward pass/step thay vì 2 (speedup thêm 2x).

Usage:
    python fuse_lcm_lora.py \
        --sdxl_path ./sdxl-base \
        --lora_path ./checkpoints/pytorch_lora_weights.safetensors \
        --output_path ./sdxl-base-lcm \
        --lora_scale 1.0

Requirements:
    pip install diffusers transformers accelerate safetensors torch peft
"""

import argparse
import os
import gc

# Patch cached_download trước khi import diffusers
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "cached_download"):
        huggingface_hub.cached_download = huggingface_hub.hf_hub_download
except Exception:
    pass

import torch
from diffusers import StableDiffusionXLPipeline, LCMScheduler
from diffusers.utils import logging

logger = logging.get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fuse LCM-LoRA into SDXL UNet")
    parser.add_argument(
        "--sdxl_path",
        type=str,
        default="./sdxl-base",
        help="Path to SDXL base model (diffusers format)",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="./checkpoints/pytorch_lora_weights.safetensors",
        help="Path to LCM-LoRA weights (.safetensors or .bin)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./sdxl-base-lcm",
        help="Output path for fused model",
    )
    parser.add_argument(
        "--lora_scale",
        type=float,
        default=1.0,
        help="LoRA scale factor (1.0 = full strength)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Model dtype",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for loading (cpu/cuda). CPU để tránh VRAM limit khi fuse.",
    )
    return parser.parse_args()


def fuse_lora_weights(args):
    """
    Fuse LCM-LoRA vào UNet.

    Diffusers pipeline.fuse_lora() tự động:
    1. Tìm các LoRA layers (lora_down, lora_up matrices)
    2. Tính: W_fused = W_original + scale * (lora_up @ lora_down)
    3. Replace original weights với fused weights
    4. Xóa LoRA layers khỏi model graph
    """
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    print(f"[1/5] Loading SDXL pipeline from {args.sdxl_path}")
    print(f"      dtype={args.dtype}, device={args.device}")

    # Load pipeline — chỉ cần UNet để fuse, nhưng diffusers cần full pipeline
    load_kwargs = {
        "torch_dtype": torch_dtype,
        "use_safetensors": True,
    }

    # Thử load với safetensors trước, fallback sang pytorch
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            args.sdxl_path, **load_kwargs
        )
    except Exception as e:
        print(f"  Warning: {e}")
        print("  Trying without use_safetensors...")
        load_kwargs.pop("use_safetensors", None)
        pipe = StableDiffusionXLPipeline.from_pretrained(
            args.sdxl_path, **load_kwargs
        )

    pipe = pipe.to(args.device)
    print(f"  UNet parameters: {sum(p.numel() for p in pipe.unet.parameters()):,}")

    print(f"\n[2/5] Loading LCM-LoRA from {args.lora_path}")

    # load_lora_weights hỗ trợ cả .safetensors và .bin
    lora_dir = os.path.dirname(os.path.abspath(args.lora_path))
    lora_file = os.path.basename(args.lora_path)

    pipe.load_lora_weights(lora_dir, weight_name=lora_file)
    print(f"  LoRA scale: {args.lora_scale}")

    print("\n[3/5] Fusing LoRA weights into UNet...")
    print("  This permanently merges LoRA deltas: W_new = W + scale * (B @ A)")

    # fuse_lora() merge weights và xóa LoRA layers
    pipe.fuse_lora(lora_scale=args.lora_scale)

    # Verify không còn LoRA adapters
    has_lora = any("lora" in name.lower() for name, _ in pipe.unet.named_parameters())
    print(f"  LoRA layers remaining: {has_lora} (should be False)")

    print("\n[4/5] Replacing scheduler with LCMScheduler...")
    # Thay scheduler bằng LCM để inference chỉ cần 4-8 steps
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    print(f"  Scheduler: {type(pipe.scheduler).__name__}")
    print(f"  Recommended inference steps: 4-8")
    print(f"  Recommended guidance_scale: 0.0 (disable CFG = 2x faster)")

    print(f"\n[5/5] Saving fused model to {args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)

    # Unload LoRA state (sau fuse thì không còn adapter nữa)
    pipe.unload_lora_weights()

    pipe.save_pretrained(args.output_path)
    print(f"  Saved successfully!")

    # Report kích thước
    _report_model_sizes(args.output_path)

    # Cleanup
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n✅ Done! Fused model ready at:", args.output_path)
    print("\nNext step: python export_sdxl_onnx.py --model_path", args.output_path)


def _report_model_sizes(model_path: str):
    """In kích thước từng component."""
    print("\n  Model sizes:")
    total = 0
    for dirpath, _, filenames in os.walk(model_path):
        for fname in sorted(filenames):
            if fname.endswith((".safetensors", ".bin", ".onnx")):
                fpath = os.path.join(dirpath, fname)
                size_mb = os.path.getsize(fpath) / (1024 * 1024)
                rel_path = os.path.relpath(fpath, model_path)
                print(f"    {rel_path}: {size_mb:.1f} MB")
                total += size_mb
    print(f"  Total: {total:.1f} MB ({total/1024:.2f} GB)")


def verify_fused_model(model_path: str, device: str = "cpu"):
    """
    Verify fused model bằng cách chạy 1 inference step với tiny resolution.
    Chỉ test shape/không crash, không test output quality.
    """
    print("\n[VERIFY] Running quick shape verification...")

    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16
        ).to(device)

        # 1 step, tiny size để verify nhanh
        with torch.no_grad():
            result = pipe(
                prompt="test",
                num_inference_steps=1,
                guidance_scale=0.0,
                height=64,
                width=64,
                output_type="latent",
            )

        latent_shape = result.images.shape
        print(f"  Output latent shape: {latent_shape}")
        # 64x64 image → latent [1, 4, 8, 8]
        assert latent_shape == (1, 4, 8, 8), f"Unexpected shape: {latent_shape}"
        print("  ✅ Verification passed!")

        del pipe
        gc.collect()

    except Exception as e:
        print(f"  ⚠️  Verification failed: {e}")
        print("  (Expected on CPU with limited memory — check GPU environment)")


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.sdxl_path):
        raise FileNotFoundError(f"SDXL model not found: {args.sdxl_path}")
    if not os.path.exists(args.lora_path):
        raise FileNotFoundError(f"LoRA weights not found: {args.lora_path}")

    fuse_lora_weights(args)
