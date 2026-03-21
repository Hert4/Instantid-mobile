"""
scripts/export_all.py
=====================
Master export script — gọi tuần tự tất cả export scripts:
  1. Export SDXL UNet (đã fuse LCM-LoRA + IP-Adapter weights)
  2. Export Text Encoders (CLIP-L + OpenCLIP-bigG)
  3. Export VAE Decoder
  4. Export ControlNet (IdentityNet)
  5. Export IP-Adapter Resampler

Lý do merge vào 1 script:
  - Dễ chạy 1 lệnh duy nhất
  - Share chung model loading code
  - Tránh load/unload pipeline nhiều lần

Usage:
    python scripts/export_all.py \
        --sdxl_path ./sdxl-base-lcm \
        --controlnet_path ./checkpoints/ControlNetModel \
        --ip_adapter_path ./checkpoints/ip-adapter.bin \
        --output_dir ./onnx \
        --opset 17

    # Chỉ export 1 component (nếu đã export phần còn lại):
    python scripts/export_all.py --only unet
    python scripts/export_all.py --only controlnet
    python scripts/export_all.py --only ip_adapter
"""

import argparse
import gc
import os
import sys
from pathlib import Path

# ── Fix 1: patch cached_download trước khi import diffusers ──────────────────
# diffusers<=0.27 dùng huggingface_hub.cached_download đã bị xóa ở hub>=0.23
# Patch bằng cách alias hf_hub_download thay thế
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "cached_download"):
        huggingface_hub.cached_download = huggingface_hub.hf_hub_download
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────

import torch
import numpy as np

# Thêm root vào path để import ip_adapter
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sdxl_path",        default="./sdxl-base-lcm")
    p.add_argument("--controlnet_path",  default="./checkpoints/ControlNetModel")
    p.add_argument("--ip_adapter_path",  default="./checkpoints/ip-adapter.bin")
    p.add_argument("--output_dir",       default="./onnx")
    p.add_argument("--opset",            type=int, default=18)
    p.add_argument("--dtype",            default="float16",
                   choices=["float16", "float32"])
    p.add_argument("--only",             default=None,
                   choices=["unet", "text_encoders", "vae", "controlnet", "ip_adapter"],
                   help="Export chỉ 1 component")
    p.add_argument("--device",           default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def torch_dtype(dtype_str):
    return torch.float16 if dtype_str == "float16" else torch.float32


def save_onnx(model, dummy_inputs, output_path, input_names, output_names,
              dynamic_axes, opset):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"  → Exporting to {output_path} ...")
    with torch.no_grad():
        # dynamo=False: force legacy ONNX exporter (TorchScript-based).
        # PyTorch 2.5+ mặc định dùng dynamo exporter mới, nhưng nó không
        # tương thích với diffusion models (ControlNet added_cond_kwargs,
        # dynamic tuple outputs, opset version mismatch với onnxscript).
        export_kwargs = dict(
            opset_version=opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
        # Thử force legacy exporter nếu PyTorch hỗ trợ dynamo param
        try:
            torch.onnx.export(
                model, dummy_inputs, output_path,
                dynamo=False,
                **export_kwargs,
            )
        except TypeError:
            # PyTorch cũ không có param dynamo → gọi bình thường
            torch.onnx.export(
                model, dummy_inputs, output_path,
                **export_kwargs,
            )
    size_mb = os.path.getsize(output_path) / 1024**2
    print(f"  ✓ Saved {size_mb:.0f} MB")


def report_size(path):
    if os.path.exists(path):
        mb = os.path.getsize(path) / 1024**2
        print(f"    Size: {mb:.0f} MB")


# ─── 1. UNet + IP-Adapter weights fused ───────────────────────────────────────

def export_unet(args, pipe, dtype):
    """
    Export SDXL UNet với IP-Adapter cross-attention weights đã được inject.

    IP-Adapter thêm 2 extra cross-attention layers vào mỗi transformer block:
      - to_k_ip, to_v_ip: project image_prompt_embeds → K, V
    Các weights này PHẢI được bake vào UNet graph trước khi export.

    Cách inject:
      1. Load IP-Adapter state dict
      2. Set attn_procs với IPAttnProcessor (chứa to_k_ip, to_v_ip weights)
      3. Export UNet — graph sẽ include các IP layers
      4. Input thêm `image_prompt_embeds` bên cạnh `encoder_hidden_states`

    UNet inputs (SDXL + ControlNet residuals + IP-Adapter):
      - sample:                [B, 4, H/8, W/8]      - noisy latent
      - timestep:              [B]
      - encoder_hidden_states: [B, 77, 2048]          - text + image embeds concat
      - added_cond_kwargs:
          - text_embeds:       [B, 1280]              - pooled text embed
          - time_ids:          [B, 6]                 - SDXL time conditioning
      - down_block_additional_residuals: list of tensors  - từ ControlNet
      - mid_block_additional_residual: tensor              - từ ControlNet

    Vấn đề với ONNX export:
      - `added_cond_kwargs` là dict → không support trực tiếp
      - ControlNet residuals là list → cần flatten thành positional args
      → Giải pháp: wrap UNet trong UNetWrapper class
    """
    print("\n[1/5] Exporting UNet...")
    out_path = os.path.join(args.output_dir, "unet", "model.onnx")

    if os.path.exists(out_path):
        print(f"  [skip] {out_path} already exists")
        return

    # Load IP-Adapter weights để inject vào UNet attention processors
    print("  Loading IP-Adapter weights for UNet injection...")
    _inject_ip_adapter_into_unet(pipe.unet, args.ip_adapter_path, dtype)

    unet = pipe.unet.eval()

    # ─── Wrapper để flatten inputs cho ONNX ───────────────────────────────
    class UNetWrapper(torch.nn.Module):
        """
        Flatten dict/list inputs thành positional args cho ONNX export.

        ControlNet returns 13 down_block residuals + 1 mid_block residual.
        SDXL UNet có 12 down blocks (3 levels × 4) = thực tế 9-13 tuỳ config.
        Kiểm tra runtime để biết chính xác số lượng.
        """

        def __init__(self, unet):
            super().__init__()
            self.unet = unet

        def forward(
            self,
            sample,              # [B, 4, H, W]
            timestep,            # [B] hoặc scalar
            encoder_hidden_states,  # [B, 77+16, 2048] (text + image prompts concat)
            text_embeds,         # [B, 1280] — pooled
            time_ids,            # [B, 6]
            # ControlNet residuals — flatten list → positional args
            # SDXL ControlNet: 9 down + 1 mid = 10 tensors
            d0, d1, d2, d3, d4, d5, d6, d7, d8,  # down blocks
            mid,                 # mid block
        ):
            added_cond_kwargs = {
                "text_embeds": text_embeds,
                "time_ids": time_ids,
            }
            down_residuals = (d0, d1, d2, d3, d4, d5, d6, d7, d8)

            out = self.unet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=down_residuals,
                mid_block_additional_residual=mid,
                return_dict=False,
            )
            return out[0]  # noise_pred [B, 4, H, W]

    wrapped = UNetWrapper(unet)

    # ─── Dummy inputs ──────────────────────────────────────────────────────
    # Resolution: 512×512 → latent 64×64
    B, H, W = 1, 64, 64
    dt = torch.float16 if dtype == "float16" else torch.float32

    sample = torch.randn(B, 4, H, W, dtype=dt, device=args.device)
    timestep = torch.tensor([999], dtype=torch.long, device=args.device)
    # encoder_hidden_states: concat text (77 tokens) + image (16 tokens từ Resampler)
    encoder_hidden_states = torch.randn(B, 77 + 16, 2048, dtype=dt, device=args.device)
    text_embeds = torch.randn(B, 1280, dtype=dt, device=args.device)
    time_ids = torch.zeros(B, 6, dtype=dt, device=args.device)

    # ControlNet residuals shapes cho SDXL (resolution 64×64 latent):
    # down_block_res_samples từ IdentityNet ControlNet:
    controlnet_down_shapes = [
        (B, 320, H, W),
        (B, 320, H, W),
        (B, 320, H, W),
        (B, 320, H // 2, W // 2),
        (B, 640, H // 2, W // 2),
        (B, 640, H // 2, W // 2),
        (B, 640, H // 4, W // 4),
        (B, 1280, H // 4, W // 4),
        (B, 1280, H // 4, W // 4),
    ]
    down_residuals = tuple(
        torch.zeros(s, dtype=dt, device=args.device)
        for s in controlnet_down_shapes
    )
    mid_residual = torch.zeros(B, 1280, H // 4, W // 4, dtype=dt, device=args.device)

    dummy_inputs = (
        sample, timestep, encoder_hidden_states,
        text_embeds, time_ids,
        *down_residuals, mid_residual
    )

    # ─── Dynamic axes ──────────────────────────────────────────────────────
    dynamic_axes = {
        "sample": {0: "batch", 2: "height", 3: "width"},
        "encoder_hidden_states": {0: "batch"},
        "text_embeds": {0: "batch"},
        "time_ids": {0: "batch"},
        "noise_pred": {0: "batch", 2: "height", 3: "width"},
    }
    # Thêm dynamic batch cho ControlNet residuals
    for i in range(9):
        dynamic_axes[f"down_residual_{i}"] = {0: "batch"}
    dynamic_axes["mid_residual"] = {0: "batch"}

    input_names = [
        "sample", "timestep", "encoder_hidden_states",
        "text_embeds", "time_ids",
        *[f"down_residual_{i}" for i in range(9)],
        "mid_residual",
    ]

    save_onnx(
        wrapped.to(args.device),
        dummy_inputs,
        out_path,
        input_names=input_names,
        output_names=["noise_pred"],
        dynamic_axes=dynamic_axes,
        opset=args.opset,
    )


def _inject_ip_adapter_into_unet(unet, ip_adapter_path, dtype):
    """
    Inject IP-Adapter cross-attention KV weights vào UNet attn processors.

    State dict structure của ip-adapter.bin:
      image_proj.*     → Resampler weights (export riêng)
      ip_adapter.*     → per-layer KV projection weights

    ip_adapter keys pattern:
      ip_adapter.{layer_idx}.to_k_ip.weight  → [320, 2048]
      ip_adapter.{layer_idx}.to_v_ip.weight  → [320, 2048]
    """
    try:
        from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor
        from ip_adapter.attention_processor import AttnProcessor2_0 as AttnProcessor
    except ImportError:
        try:
            from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor
        except ImportError:
            print("  [warn] ip_adapter module not found — UNet exported WITHOUT IP-Adapter")
            print("         Run setup.sh first to copy ip_adapter/ module")
            return

    print("  Injecting IP-Adapter weights into UNet attention processors...")
    state_dict = torch.load(ip_adapter_path, map_location="cpu")

    # ── Detect nested vs flat format (same fix as export_ip_adapter) ──
    if "ip_adapter" in state_dict and isinstance(state_dict["ip_adapter"], dict):
        # Nested dict format (InstantID official)
        ip_layers = state_dict["ip_adapter"]
    else:
        # Flat keys format
        ip_layers = {
            k.replace("ip_adapter.", ""): v
            for k, v in state_dict.items()
            if k.startswith("ip_adapter.")
        }

    # Set attention processors
    attn_procs = {}
    ip_layer_idx = 0

    for name in unet.attn_processors.keys():
        # cross_attention layers nhận encoder_hidden_states
        if name.endswith("attn2.processor"):
            # Lấy hidden_size từ UNet config
            cross_attention_dim = unet.config.cross_attention_dim
            if isinstance(cross_attention_dim, list):
                # SDXL có nhiều levels
                layer_name = name.split(".")[0]
                hidden_size = _get_hidden_size(unet, name)
            else:
                hidden_size = cross_attention_dim

            proc = IPAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim if not isinstance(cross_attention_dim, list)
                                   else cross_attention_dim[0],
                num_tokens=16,  # Resampler num_queries=16
            )

            # Load weights cho processor này
            key_k = f"{ip_layer_idx}.to_k_ip.weight"
            key_v = f"{ip_layer_idx}.to_v_ip.weight"
            if key_k in ip_layers and key_v in ip_layers:
                proc.to_k_ip.weight.data = ip_layers[key_k].to(
                    torch.float16 if dtype == "float16" else torch.float32
                )
                proc.to_v_ip.weight.data = ip_layers[key_v].to(
                    torch.float16 if dtype == "float16" else torch.float32
                )
            ip_layer_idx += 1

            attn_procs[name] = proc
        else:
            attn_procs[name] = AttnProcessor()

    unet.set_attn_processor(attn_procs)
    print(f"  Injected IP-Adapter into {ip_layer_idx} cross-attention layers")


def _get_hidden_size(unet, attn_name):
    """Lấy hidden_size của một attention layer từ UNet."""
    # SDXL hidden sizes: [320, 640, 1280]
    if "down_blocks.0" in attn_name or "up_blocks.3" in attn_name:
        return 320
    elif "down_blocks.1" in attn_name or "up_blocks.2" in attn_name:
        return 640
    else:
        return 1280


# ─── 2. Text Encoders ─────────────────────────────────────────────────────────

def export_text_encoders(args, pipe, dtype):
    """
    Export 2 text encoders của SDXL:
      - text_encoder:   CLIP-ViT-L/14  → [B, 77, 768]
      - text_encoder_2: OpenCLIP-ViT-G → [B, 77, 1280] + pooled [B, 1280]

    Cả 2 outputs được concat: [B, 77, 768+1280] = [B, 77, 2048]
    """
    print("\n[2/5] Exporting Text Encoders...")
    dt = torch.float16 if dtype == "float16" else torch.float32

    # ── Text Encoder 1 (CLIP-L) ──────────────────────────────────────────
    out_path_1 = os.path.join(args.output_dir, "text_encoder", "model.onnx")
    if not os.path.exists(out_path_1):
        print("  Exporting text_encoder_1 (CLIP-L)...")
        enc1 = pipe.text_encoder.eval().to(args.device)

        class TextEncoder1Wrapper(torch.nn.Module):
            def __init__(self, enc): super().__init__(); self.enc = enc
            def forward(self, input_ids):
                out = self.enc(input_ids, return_dict=True)
                return out.last_hidden_state  # [B, 77, 768]

        dummy = torch.zeros(1, 77, dtype=torch.long, device=args.device)
        save_onnx(
            TextEncoder1Wrapper(enc1),
            (dummy,),
            out_path_1,
            input_names=["input_ids"],
            output_names=["last_hidden_state"],
            dynamic_axes={"input_ids": {0: "batch"}, "last_hidden_state": {0: "batch"}},
            opset=args.opset,
        )
        # Unload sau khi export
        del enc1
        gc.collect()
    else:
        print(f"  [skip] text_encoder_1 exists")

    # ── Text Encoder 2 (OpenCLIP-bigG) ───────────────────────────────────
    out_path_2 = os.path.join(args.output_dir, "text_encoder_2", "model.onnx")
    if not os.path.exists(out_path_2):
        print("  Exporting text_encoder_2 (OpenCLIP-G)...")
        enc2 = pipe.text_encoder_2.eval().to(args.device)

        class TextEncoder2Wrapper(torch.nn.Module):
            def __init__(self, enc): super().__init__(); self.enc = enc
            def forward(self, input_ids):
                out = self.enc(input_ids, return_dict=True, output_hidden_states=True)
                # penultimate hidden state (không phải final layer norm)
                hidden = out.hidden_states[-2]   # [B, 77, 1280]
                pooled = out.text_embeds          # [B, 1280]
                return hidden, pooled

        dummy = torch.zeros(1, 77, dtype=torch.long, device=args.device)
        save_onnx(
            TextEncoder2Wrapper(enc2),
            (dummy,),
            out_path_2,
            input_names=["input_ids"],
            output_names=["hidden_states", "pooled_output"],
            dynamic_axes={
                "input_ids": {0: "batch"},
                "hidden_states": {0: "batch"},
                "pooled_output": {0: "batch"},
            },
            opset=args.opset,
        )
        del enc2
        gc.collect()
    else:
        print(f"  [skip] text_encoder_2 exists")


# ─── 3. VAE Decoder ───────────────────────────────────────────────────────────

def export_vae(args, pipe, dtype):
    """
    Export VAE Decoder: latent [B,4,H,W] → image [B,3,H*8,W*8]

    Note: VAE rất nhạy với quantization.
    Giữ FP16, KHÔNG quantize INT8 trừ khi test kỹ.

    Kích thước VAE decoder: ~160MB FP16 → chấp nhận được.
    """
    print("\n[3/5] Exporting VAE Decoder...")
    out_path = os.path.join(args.output_dir, "vae_decoder", "model.onnx")

    if os.path.exists(out_path):
        print(f"  [skip] {out_path} exists")
        return

    dt = torch.float16 if dtype == "float16" else torch.float32
    vae = pipe.vae.eval().to(args.device)

    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.decoder = vae.decoder
            self.post_quant_conv = vae.post_quant_conv
            # Scale factor của VAE (SDXL = 0.13025)
            self.scale = 1.0 / vae.config.scaling_factor

        def forward(self, latent):
            # Un-scale latent
            z = latent * self.scale
            z = self.post_quant_conv(z)
            image = self.decoder(z)
            # Clamp và normalize [−1,1] → [0,1]
            image = (image / 2 + 0.5).clamp(0, 1)
            return image  # [B, 3, H*8, W*8]

    # 512×512 output → latent 64×64
    dummy = torch.randn(1, 4, 64, 64, dtype=dt, device=args.device)

    save_onnx(
        VAEDecoderWrapper(vae),
        (dummy,),
        out_path,
        input_names=["latent"],
        output_names=["image"],
        dynamic_axes={
            "latent": {0: "batch", 2: "height", 3: "width"},
            "image": {0: "batch", 2: "height", 3: "width"},
        },
        opset=args.opset,
    )

    del vae
    gc.collect()


# ─── 4. ControlNet (IdentityNet) ──────────────────────────────────────────────

def export_controlnet(args, dtype):
    """
    Export ControlNet IdentityNet của InstantID.

    Input:
      - sample:                [B, 4, H, W]         — noisy latent
      - timestep:              [B]
      - encoder_hidden_states: [B, 77, 2048]         — text embeddings
      - controlnet_cond:       [B, 3, H*8, W*8]      — face keypoints image

    Output:
      - down_block_res_samples: tuple of 9 tensors
      - mid_block_res_sample:   [B, 1280, H/4, W/4]

    Note: ControlNet của InstantID là standard ControlNet, KHÔNG có custom
    cross-attention như IP-Adapter. Output là residuals inject vào UNet.
    """
    print("\n[4/5] Exporting ControlNet (IdentityNet)...")
    out_path = os.path.join(args.output_dir, "controlnet", "model.onnx")

    if os.path.exists(out_path):
        print(f"  [skip] {out_path} exists")
        return

    from diffusers.models import ControlNetModel

    print(f"  Loading ControlNet from {args.controlnet_path}...")
    dt = torch.float16 if dtype == "float16" else torch.float32
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_path,
        torch_dtype=dt,
    ).eval().to(args.device)

    class ControlNetWrapper(torch.nn.Module):
        """
        Flatten tuple output thành flat positional outputs cho ONNX.
        ONNX không support dynamic-length tuple outputs.

        SDXL ControlNet cần added_cond_kwargs (text_embeds, time_ids) —
        nếu không truyền sẽ gặp lỗi 'NoneType is not iterable'.
        """
        def __init__(self, controlnet):
            super().__init__()
            self.controlnet = controlnet

        def forward(self, sample, timestep, encoder_hidden_states,
                    controlnet_cond, conditioning_scale,
                    text_embeds, time_ids):
            added_cond_kwargs = {
                "text_embeds": text_embeds,
                "time_ids": time_ids,
            }
            down_samples, mid_sample = self.controlnet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_cond,
                conditioning_scale=conditioning_scale,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )
            # down_samples là tuple 9 tensors, mid_sample là 1 tensor
            # Flatten thành flat outputs
            return (*down_samples, mid_sample)

    B, H, W = 1, 64, 64
    dummy_inputs = (
        torch.randn(B, 4, H, W, dtype=dt, device=args.device),          # sample
        torch.tensor([999], dtype=torch.long, device=args.device),        # timestep
        torch.randn(B, 77, 2048, dtype=dt, device=args.device),           # encoder_hidden_states
        torch.randn(B, 3, H * 8, W * 8, dtype=dt, device=args.device),   # controlnet_cond (face kps)
        torch.tensor(1.0, dtype=dt, device=args.device),                  # conditioning_scale
        torch.randn(B, 1280, dtype=dt, device=args.device),               # text_embeds (pooled)
        torch.randn(B, 6, dtype=dt, device=args.device),                  # time_ids (SDXL: 6 values)
    )

    # Output names: 9 down + 1 mid
    output_names = [f"down_{i}" for i in range(9)] + ["mid"]

    dynamic_axes = {
        "sample": {0: "batch", 2: "height", 3: "width"},
        "encoder_hidden_states": {0: "batch"},
        "controlnet_cond": {0: "batch", 2: "img_height", 3: "img_width"},
        "text_embeds": {0: "batch"},
        "time_ids": {0: "batch"},
    }
    for name in output_names:
        dynamic_axes[name] = {0: "batch"}

    save_onnx(
        ControlNetWrapper(controlnet).eval(),
        dummy_inputs,
        out_path,
        input_names=["sample", "timestep", "encoder_hidden_states",
                     "controlnet_cond", "conditioning_scale",
                     "text_embeds", "time_ids"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset=args.opset,
    )

    del controlnet
    gc.collect()


# ─── 5. IP-Adapter Resampler ──────────────────────────────────────────────────

def export_ip_adapter(args, dtype):
    """
    Export IP-Adapter Resampler (projection network).

    Resampler: face_embedding [B, 512] → image_prompt_embeds [B, 16, 2048]

    Architecture:
      - proj_in: Linear(512, 1024)
      - 8 × (PerceiverAttention + FeedForward)
      - proj_out: Linear(1024, 2048)
      - norm_out: LayerNorm(2048)
      - latents: learnable queries [1, 16, 1024] (broadcast over batch)

    Note: Resampler nhỏ (~100MB) và chạy rất nhanh (~50ms).
    Export riêng để load/unload độc lập với UNet.
    """
    print("\n[5/5] Exporting IP-Adapter Resampler...")
    out_path = os.path.join(args.output_dir, "ip_adapter", "model.onnx")

    if os.path.exists(out_path):
        print(f"  [skip] {out_path} exists")
        return

    from ip_adapter.resampler import Resampler

    print(f"  Loading ip-adapter from {args.ip_adapter_path}...")
    state_dict = torch.load(args.ip_adapter_path, map_location="cpu")

    # ── InstantID ip-adapter.bin có thể có 2 format: ──
    # Format A (flat): {"image_proj.proj_in.weight": ..., "ip_adapter.1.to_k_ip.weight": ...}
    # Format B (nested dict): {"image_proj": {"proj_in.weight": ...}, "ip_adapter": {...}}
    #
    # Detect format tự động:
    image_proj_state = {}

    if "image_proj" in state_dict and isinstance(state_dict["image_proj"], dict):
        # Format B: nested dict — InstantID official format
        print("  Detected nested dict format (InstantID official)")
        image_proj_state = state_dict["image_proj"]
    else:
        # Format A: flat keys
        image_proj_state = {
            k.replace("image_proj.", ""): v
            for k, v in state_dict.items()
            if k.startswith("image_proj.")
        }

    if not image_proj_state:
        # Debug: in top-level keys để biết cấu trúc thực tế
        top_keys = list(state_dict.keys())[:20]
        sample_types = {k: type(v).__name__ for k, v in list(state_dict.items())[:5]}
        raise ValueError(
            f"ip-adapter.bin không có key 'image_proj.*'\n"
            f"Top-level keys ({len(state_dict)}): {top_keys}\n"
            f"Sample value types: {sample_types}"
        )

    # ── Fix 2: tự detect Resampler config từ state_dict thay vì hardcode ──
    # Đọc shape thực tế từ weights để tránh mismatch
    print("  Detecting Resampler config from state_dict...")
    proj_in_w  = image_proj_state.get("proj_in.weight")   # [dim, embedding_dim]
    proj_out_w = image_proj_state.get("proj_out.weight")   # [output_dim, dim]
    latents    = image_proj_state.get("latents")           # [1, num_queries, dim]

    if proj_in_w is None or proj_out_w is None or latents is None:
        print("  Available keys:", list(image_proj_state.keys())[:10])
        raise ValueError("Không tìm thấy proj_in/proj_out/latents trong state_dict")

    dim           = proj_in_w.shape[0]        # e.g. 1024
    embedding_dim = proj_in_w.shape[1]        # e.g. 512
    output_dim    = proj_out_w.shape[0]       # e.g. 2048
    num_queries   = latents.shape[1]          # e.g. 16

    # Đếm số layers từ keys
    depth = sum(1 for k in image_proj_state if k.startswith("layers.") and k.endswith(".to_q.weight"))

    # Detect dim_head và heads từ to_q weight: [inner_dim, dim] where inner_dim = heads * dim_head
    to_q_w    = image_proj_state.get("layers.0.0.to_q.weight")
    inner_dim = to_q_w.shape[0] if to_q_w is not None else dim
    # Thường dim_head=64, heads = inner_dim // 64
    dim_head  = 64
    heads     = max(1, inner_dim // dim_head)

    print(f"  Detected config: dim={dim}, depth={depth}, heads={heads}, "
          f"num_queries={num_queries}, embedding_dim={embedding_dim}, output_dim={output_dim}")

    resampler = Resampler(
        dim=dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        num_queries=num_queries,
        embedding_dim=embedding_dim,
        output_dim=output_dim,
        ff_mult=4,
    )
    resampler.load_state_dict(image_proj_state, strict=True)

    dt = torch.float16 if dtype == "float16" else torch.float32
    resampler = resampler.to(dt).eval()

    # Input: face embedding từ InsightFace glintr100
    # Shape: [B, 512] (glintr100 output là 512-dim normalized vector)
    # Nhưng Resampler.forward() nhận [B, N, dim] → cần unsqueeze
    class ResamplerWrapper(torch.nn.Module):
        def __init__(self, resampler):
            super().__init__()
            self.resampler = resampler

        def forward(self, face_embedding):
            # face_embedding: [B, 512]
            # Resampler expects [B, N, dim] → treat như [B, 1, 512]
            x = face_embedding.unsqueeze(1)  # [B, 1, 512]
            return self.resampler(x)          # [B, 16, 2048]

    dummy = torch.randn(1, 512, dtype=dt)

    save_onnx(
        ResamplerWrapper(resampler).eval(),
        (dummy,),
        out_path,
        input_names=["face_embedding"],
        output_names=["image_prompt_embeds"],
        dynamic_axes={
            "face_embedding": {0: "batch"},
            "image_prompt_embeds": {0: "batch"},
        },
        opset=args.opset,
    )

    del resampler
    gc.collect()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dtype = args.dtype

    print(f"\nExport config:")
    print(f"  SDXL:      {args.sdxl_path}")
    print(f"  ControlNet:{args.controlnet_path}")
    print(f"  IP-Adapter:{args.ip_adapter_path}")
    print(f"  Output:    {args.output_dir}")
    print(f"  Device:    {args.device}")
    print(f"  Dtype:     {dtype}")
    print(f"  Opset:     {args.opset}")

    # ──────────────────────────────────────────────────────────────────────
    # Validate paths — chỉ check những path cần thiết cho component đang export
    needs_sdxl = args.only in (None, "unet", "text_encoders", "vae")
    needs_controlnet = args.only in (None, "controlnet")
    needs_ip_adapter = args.only in (None, "ip_adapter", "unet")

    path_checks = []
    if needs_sdxl:
        path_checks.append((args.sdxl_path, "SDXL model"))
    if needs_controlnet:
        path_checks.append((args.controlnet_path, "ControlNet"))
    if needs_ip_adapter:
        path_checks.append((args.ip_adapter_path, "IP-Adapter"))

    for path, name in path_checks:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} not found: {path}\n"
                f"Run ./setup.sh first to download models"
            )

    dt = torch_dtype(dtype)

    # ──────────────────────────────────────────────────────────────────────
    # ControlNet và IP-Adapter Resampler không cần pipeline → export trước
    if args.only in (None, "controlnet"):
        export_controlnet(args, dtype)

    if args.only in (None, "ip_adapter"):
        export_ip_adapter(args, dtype)

    # ──────────────────────────────────────────────────────────────────────
    # Load SDXL pipeline (cần cho UNet, TextEncoders, VAE)
    if args.only not in ("controlnet", "ip_adapter"):
        print(f"\nLoading SDXL pipeline from {args.sdxl_path}...")
        print("  (This may take a few minutes for large models)")

        from diffusers import StableDiffusionXLPipeline

        pipe = StableDiffusionXLPipeline.from_pretrained(
            args.sdxl_path,
            torch_dtype=dt,
            use_safetensors=True,
        )
        pipe = pipe.to(args.device)

        if args.only in (None, "unet"):
            export_unet(args, pipe, dtype)

        if args.only in (None, "text_encoders"):
            export_text_encoders(args, pipe, dtype)

        if args.only in (None, "vae"):
            export_vae(args, pipe, dtype)

        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "═" * 50)
    print("✅ Export complete!")
    print("\nExported files:")
    for root, _, files in os.walk(args.output_dir):
        for f in sorted(files):
            if f.endswith(".onnx"):
                p = os.path.join(root, f)
                mb = os.path.getsize(p) / 1024**2
                rel = os.path.relpath(p, args.output_dir)
                print(f"  {rel}: {mb:.0f} MB")

    print(f"\nNext: python scripts/quantize_all.py --input_dir {args.output_dir}")


if __name__ == "__main__":
    main()
