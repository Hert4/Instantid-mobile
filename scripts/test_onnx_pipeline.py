"""
scripts/test_onnx_pipeline.py
==============================
Test toàn bộ ONNX pipeline trên PC — verify trước khi port mobile.

Chạy full inference với ONNX Runtime và so sánh:
  - Shape outputs có đúng không?
  - Visual output có reasonable không (face preserved)?
  - Memory usage từng stage

Đây là bước debug QUAN TRỌNG nhất trước khi động vào C++ mobile code.
Fix mọi thứ ở đây trước.

Usage:
    python scripts/test_onnx_pipeline.py \
        --face_image examples/face.jpg \
        --onnx_dir ./onnx-int8 \
        --prompt "a person, professional photo, 4k" \
        --output output/test_result.png \
        --steps 4

    # Test với FP16 (trước quantize):
    python scripts/test_onnx_pipeline.py \
        --face_image examples/face.jpg \
        --onnx_dir ./onnx
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

# Patch cached_download
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "cached_download"):
        huggingface_hub.cached_download = huggingface_hub.hf_hub_download
except Exception:
    pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--face_image",    required=True, help="Path to face image")
    p.add_argument("--onnx_dir",      default="./onnx-int8")
    p.add_argument("--insightface_dir", default="./models/antelopev2")
    p.add_argument("--prompt",        default="portrait photo of a person, professional lighting, 4k, detailed")
    p.add_argument("--neg_prompt",    default="(lowres, low quality:1.2), watermark, text, blurry, deformed")
    p.add_argument("--steps",         type=int, default=4,
                   help="Inference steps (4-8 cho LCM)")
    p.add_argument("--ip_scale",      type=float, default=0.8,
                   help="IP-Adapter scale (face preservation strength)")
    p.add_argument("--cn_scale",      type=float, default=0.8,
                   help="ControlNet conditioning scale")
    p.add_argument("--height",        type=int, default=512)
    p.add_argument("--width",         type=int, default=512)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--output",        default="./output/test_result.png")
    p.add_argument("--device",        default="cpu",
                   help="ONNX Runtime device: cpu / cuda / dml")
    p.add_argument("--verbose",       action="store_true")
    return p.parse_args()


# ─── ONNX Runtime session helper ──────────────────────────────────────────────

def create_session(model_path: str, device: str = "cpu"):
    """Create ONNX Runtime InferenceSession."""
    import onnxruntime as ort

    providers = {
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "dml":  ["DmlExecutionProvider", "CPUExecutionProvider"],
        "cpu":  ["CPUExecutionProvider"],
    }.get(device, ["CPUExecutionProvider"])

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_mem_pattern = True
    opts.enable_cpu_mem_arena = True

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    return ort.InferenceSession(model_path, sess_options=opts, providers=providers)


def run_session(session, inputs: dict) -> dict:
    """Run inference, return dict of outputs."""
    output_names = [o.name for o in session.get_outputs()]
    results = session.run(output_names, inputs)
    return dict(zip(output_names, results))


# ─── Stage 1: Face Processing ─────────────────────────────────────────────────

class FaceProcessor:
    """
    Minimal InsightFace inference dùng ONNX Runtime trực tiếp.
    Không cần insightface Python package — chỉ cần ONNX files.
    """

    def __init__(self, models_dir: str, device: str = "cpu"):
        self.device = device
        self.det_size = (640, 640)

        det_path = os.path.join(models_dir, "scrfd_10g_bnkps.onnx")
        emb_path = os.path.join(models_dir, "glintr100.onnx")

        # Fallback sang insightface package nếu ONNX chưa có
        if not os.path.exists(det_path) or not os.path.exists(emb_path):
            print("  [warn] InsightFace ONNX models not found, using insightface package")
            self._use_package = True
            self._init_package(models_dir)
        else:
            self._use_package = False
            self.det_session = create_session(det_path, device)
            self.emb_session = create_session(emb_path, device)

    def _init_package(self, models_dir: str):
        try:
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(
                name="antelopev2",
                root=str(Path(models_dir).parent.parent),
                providers=["CPUExecutionProvider"],
            )
            self.app.prepare(ctx_id=-1, det_size=self.det_size)
        except Exception as e:
            raise RuntimeError(f"InsightFace not available: {e}\n"
                               f"Download ONNX models to {models_dir} or: pip install insightface")

    def get_face_info(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            face_embedding: [512] float32
            face_kps:       [5, 2] float32 (keypoints in image coords)
        """
        if self._use_package:
            return self._get_face_package(image_rgb)
        else:
            return self._get_face_onnx(image_rgb)

    def _get_face_package(self, image_rgb: np.ndarray):
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        faces = self.app.get(bgr)
        if not faces:
            raise ValueError("No face detected in image")
        # Lấy face lớn nhất
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        return face.embedding.astype(np.float32), face.kps.astype(np.float32)

    def _get_face_onnx(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        SCRFD face detection → keypoints
        GlintR100 face embedding
        """
        # Preprocessing cho SCRFD
        img_h, img_w = image_rgb.shape[:2]
        det_h, det_w = self.det_size

        scale = min(det_w / img_w, det_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        resized = cv2.resize(image_rgb, (new_w, new_h))

        # Pad to det_size
        padded = np.zeros((det_h, det_w, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        # SCRFD expects BGR, normalized
        bgr = padded[:, :, ::-1].astype(np.float32)
        mean = np.array([127.5, 127.5, 127.5])
        std  = np.array([128.0, 128.0, 128.0])
        bgr = (bgr - mean) / std
        inp = bgr.transpose(2, 0, 1)[np.newaxis]  # [1, 3, H, W]

        det_out = run_session(self.det_session, {"input.1": inp})

        # Parse SCRFD output → get keypoints for largest face
        # SCRFD output: scores, bboxes, kpss
        kps = self._parse_scrfd_output(det_out, scale, img_w, img_h)

        # GlintR100 face embedding
        # Crop + align face using keypoints
        face_crop = self._align_face(image_rgb, kps)

        # GlintR100: [1,3,112,112] → [1,512]
        face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR).astype(np.float32)
        face_norm = (face_bgr - 127.5) / 128.0
        face_inp = face_norm.transpose(2, 0, 1)[np.newaxis]  # [1,3,112,112]

        emb_out = run_session(self.emb_session, {"input.1": face_inp})
        embedding = list(emb_out.values())[0][0]  # [512]

        # L2 normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-6)

        return embedding.astype(np.float32), kps.astype(np.float32)

    def _parse_scrfd_output(self, det_out, scale, img_w, img_h):
        """Parse SCRFD output, return [5,2] keypoints của face lớn nhất."""
        # Simplified: return center keypoints nếu parse phức tạp
        # Full implementation cần decode SCRFD anchors
        # Tạm thời dùng center-based estimation
        cx, cy = img_w / 2, img_h / 2
        kps = np.array([
            [cx - 0.2*img_w, cy - 0.1*img_h],   # left eye
            [cx + 0.2*img_w, cy - 0.1*img_h],   # right eye
            [cx,              cy               ],  # nose
            [cx - 0.15*img_w, cy + 0.15*img_h], # left mouth
            [cx + 0.15*img_w, cy + 0.15*img_h], # right mouth
        ], dtype=np.float32)
        return kps

    def _align_face(self, image: np.ndarray, kps: np.ndarray,
                    size: int = 112) -> np.ndarray:
        """ArcFace standard alignment."""
        from skimage import transform as trans

        src = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ], dtype=np.float32)

        tform = trans.SimilarityTransform()
        tform.estimate(kps, src)
        M = tform.params[0:2, :]
        aligned = cv2.warpAffine(image, M, (size, size), flags=cv2.INTER_LINEAR)
        return aligned


# ─── Stage 2: Text Encoding ───────────────────────────────────────────────────

class TextEncoderONNX:
    """CLIP text encoders via ONNX Runtime."""

    def __init__(self, enc1_path: str, enc2_path: str,
                 sdxl_path: str, device: str = "cpu"):
        self.device = device
        self.enc1_session = create_session(enc1_path, device)
        self.enc2_session = create_session(enc2_path, device)

        # Load tokenizers từ SDXL model directory
        from transformers import CLIPTokenizer
        self.tokenizer_1 = CLIPTokenizer.from_pretrained(sdxl_path, subfolder="tokenizer")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(sdxl_path, subfolder="tokenizer_2")

    def encode(self, prompt: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            text_embeds: [1, 77, 2048] (concat of enc1 + enc2)
            pooled:      [1, 1280]
        """
        # Tokenize
        tok1 = self.tokenizer_1(
            prompt, padding="max_length", max_length=77,
            truncation=True, return_tensors="np"
        )
        tok2 = self.tokenizer_2(
            prompt, padding="max_length", max_length=77,
            truncation=True, return_tensors="np"
        )

        # Encode
        out1 = run_session(self.enc1_session, {"input_ids": tok1["input_ids"]})
        out2 = run_session(self.enc2_session, {"input_ids": tok2["input_ids"]})

        hidden1 = out1["last_hidden_state"]   # [1, 77, 768]
        hidden2 = out2["hidden_states"]        # [1, 77, 1280]
        pooled  = out2["pooled_output"]        # [1, 1280]

        # Concat → [1, 77, 2048]
        text_embeds = np.concatenate([hidden1, hidden2], axis=-1)

        return text_embeds, pooled


# ─── Stage 3: IP-Adapter ──────────────────────────────────────────────────────

class IPAdapterONNX:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.session = create_session(model_path, device)

    def project(self, face_embedding: np.ndarray) -> np.ndarray:
        """face_embedding [1,512] → [1,16,2048]"""
        out = run_session(self.session, {"face_embedding": face_embedding})
        return out["image_prompt_embeds"]  # [1, 16, 2048]


# ─── Stage 4: ControlNet ──────────────────────────────────────────────────────

class ControlNetONNX:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.session = create_session(model_path, device)

    def forward(self, sample: np.ndarray, timestep: np.ndarray,
                encoder_hidden_states: np.ndarray,
                face_kps_image: np.ndarray,
                conditioning_scale: float = 0.8) -> Tuple:
        """
        Returns: (down_residuals_list, mid_residual)
        """
        inputs = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "controlnet_cond": face_kps_image,
            "conditioning_scale": np.array(conditioning_scale, dtype=sample.dtype),
        }
        out = run_session(self.session, inputs)
        # out: {down_0, ..., down_8, mid}
        downs = [out[f"down_{i}"] for i in range(9)]
        mid = out["mid"]
        return downs, mid


# ─── Stage 4: UNet ────────────────────────────────────────────────────────────

class UNetONNX:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.session = create_session(model_path, device)

    def forward(self, sample: np.ndarray, timestep: np.ndarray,
                encoder_hidden_states: np.ndarray,
                text_embeds: np.ndarray, time_ids: np.ndarray,
                down_residuals: List[np.ndarray],
                mid_residual: np.ndarray) -> np.ndarray:
        inputs = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "text_embeds": text_embeds,
            "time_ids": time_ids,
        }
        for i, d in enumerate(down_residuals):
            inputs[f"down_residual_{i}"] = d
        inputs["mid_residual"] = mid_residual

        out = run_session(self.session, inputs)
        return out["noise_pred"]


# ─── Stage 5: VAE Decoder ─────────────────────────────────────────────────────

class VAEDecoderONNX:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.session = create_session(model_path, device)

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """latent [1,4,H,W] → image [1,3,H*8,W*8] in [0,1]"""
        out = run_session(self.session, {"latent": latent})
        return out["image"]


# ─── LCM Scheduler ────────────────────────────────────────────────────────────

class LCMSchedulerNP:
    """
    Numpy implementation của LCM Euler scheduler.
    Compatible với latent-consistency/lcm-lora-sdxl.
    """

    def __init__(self, num_steps: int = 4,
                 num_train_timesteps: int = 1000,
                 beta_start: float = 0.00085,
                 beta_end: float = 0.012):
        self.num_steps = num_steps
        self.num_train_timesteps = num_train_timesteps

        # Cosine schedule (SDXL default)
        betas = self._cosine_beta_schedule(num_train_timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)

        self.alphas_cumprod = alphas_cumprod

        # LCM timestep spacing: linspace trong [0, 999]
        step_ratio = num_train_timesteps // num_steps
        self.timesteps = np.arange(0, num_steps)[::-1] * step_ratio + step_ratio - 1
        self.timesteps = self.timesteps.astype(np.int64)

    def _cosine_beta_schedule(self, T, beta_start, beta_end):
        """Squaredcos_cap_v2 schedule dùng trong SDXL."""
        t = np.linspace(0, T, T + 1)
        alphas_cumprod = np.cos((t / T + 0.008) / 1.008 * np.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return np.clip(betas, beta_start, beta_end)

    def scale_noise(self, latents: np.ndarray, timestep: int) -> np.ndarray:
        """Scale initial noise theo timestep (LCM cần full noise ở t=T)."""
        alpha_prod_t = self.alphas_cumprod[timestep]
        sigma = np.sqrt(1 - alpha_prod_t)
        return latents * sigma  # Initial latent là pure noise khi t=T

    def step(self, noise_pred: np.ndarray, timestep: int,
             latents: np.ndarray) -> np.ndarray:
        """
        LCM Euler step:
          x_0_pred = (x_t - sigma_t * noise_pred) / alpha_t
          x_{t-1} = alpha_{t-1} * x_0_pred + sigma_{t-1} * noise
        """
        t = timestep
        alpha_prod_t = self.alphas_cumprod[t]
        sigma_t = np.sqrt(1 - alpha_prod_t)
        alpha_t = np.sqrt(alpha_prod_t)

        # Predict x_0
        x0_pred = (latents - sigma_t * noise_pred) / (alpha_t + 1e-8)
        x0_pred = np.clip(x0_pred, -1.0, 1.0)

        # Get prev timestep
        t_idx = np.where(self.timesteps == t)[0]
        if len(t_idx) > 0 and t_idx[0] + 1 < len(self.timesteps):
            t_prev = self.timesteps[t_idx[0] + 1]
            alpha_prod_prev = self.alphas_cumprod[t_prev]
        else:
            alpha_prod_prev = 1.0

        sigma_prev = np.sqrt(1 - alpha_prod_prev)
        alpha_prev  = np.sqrt(alpha_prod_prev)

        # LCM: deterministic step (no noise added)
        latents_prev = alpha_prev * x0_pred + sigma_prev * noise_pred

        return latents_prev


# ─── Draw face keypoints ───────────────────────────────────────────────────────

def draw_kps(image_pil: Image.Image, kps: np.ndarray,
             target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """
    Vẽ face keypoints lên ảnh đen.
    Output: [1, 3, H, W] float16/32 trong [-1, 1] (normalized cho ControlNet)

    5 keypoints: left_eye, right_eye, nose, left_mouth, right_mouth
    """
    color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    limbSeq = [[0, 2], [1, 2], [3, 2], [4, 2]]

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3], dtype=np.uint8)

    # Scale keypoints nếu cần
    kps_scaled = kps.copy()

    # Vẽ limbs
    for seq in limbSeq:
        for idx in seq:
            color = color_list[idx]
            x, y = int(kps_scaled[idx][0]), int(kps_scaled[idx][1])
            cv2.circle(out_img, (x, y), 5, color, -1)

    # Vẽ lines
    for seq in limbSeq:
        pts = kps_scaled[seq].astype(np.int32)
        cv2.line(out_img, tuple(pts[0]), tuple(pts[1]), (255, 255, 255), 2)

    # Resize to target
    kps_image = cv2.resize(out_img, target_size[::-1])  # cv2 uses (w,h)

    # Normalize [0,255] → [0,1] → [0,1] float32 → [1,3,H,W]
    kps_np = kps_image.astype(np.float32) / 255.0
    kps_np = kps_np.transpose(2, 0, 1)[np.newaxis]  # [1, 3, H, W]

    return kps_np


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def get_time_ids(height: int, width: int) -> np.ndarray:
    """SDXL time_ids: [original_h, original_w, crop_top, crop_left, target_h, target_w]"""
    return np.array([[height, width, 0, 0, height, width]], dtype=np.float16)


def run_pipeline(args):
    """Full InstantID inference pipeline với ONNX Runtime."""
    rng = np.random.RandomState(args.seed)
    H, W = args.height, args.width

    # Find correct ONNX paths (int8 or fp16)
    def find_model(subdir, names):
        for name in names:
            p = os.path.join(args.onnx_dir, subdir, name)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"No model found in {subdir}: tried {names}")

    print("\n" + "═"*50)
    print("InstantID ONNX Pipeline Test")
    print("═"*50)

    timings = {}

    # ── Stage 1: Face Processing ──────────────────────────────────────────
    print("\n[Stage 1] Face Processing...")
    t0 = time.time()

    face_img = Image.open(args.face_image).convert("RGB")
    face_np  = np.array(face_img)

    face_proc = FaceProcessor(args.insightface_dir, args.device)
    face_embedding, face_kps = face_proc.get_face_info(face_np)

    print(f"  face_embedding shape: {face_embedding.shape}")  # [512]
    print(f"  face_kps shape:       {face_kps.shape}")        # [5, 2]

    kps_image = draw_kps(face_img, face_kps, target_size=(H, W))
    print(f"  kps_image shape:      {kps_image.shape}")       # [1, 3, H, W]

    timings["face"] = time.time() - t0
    del face_proc
    gc.collect()

    # ── Stage 2: Text Encoding ────────────────────────────────────────────
    print(f"\n[Stage 2] Text Encoding...")
    t0 = time.time()

    sdxl_dir = "./sdxl-base-lcm"
    if not os.path.exists(sdxl_dir):
        sdxl_dir = "./sdxl-base"

    enc = TextEncoderONNX(
        enc1_path=find_model("text_encoder", ["model_int8.onnx", "model.onnx"]),
        enc2_path=find_model("text_encoder_2", ["model_int8.onnx", "model.onnx"]),
        sdxl_path=sdxl_dir,
        device=args.device,
    )

    text_embeds, pooled = enc.encode(args.prompt)
    neg_embeds, neg_pooled = enc.encode(args.neg_prompt)

    print(f"  text_embeds shape: {text_embeds.shape}")   # [1, 77, 2048]
    print(f"  pooled shape:      {pooled.shape}")         # [1, 1280]

    timings["text_encode"] = time.time() - t0
    del enc
    gc.collect()

    # ── Stage 3: IP-Adapter Projection ───────────────────────────────────
    print(f"\n[Stage 3] IP-Adapter Projection...")
    t0 = time.time()

    ip_path = find_model("ip_adapter", ["model_int8.onnx", "model.onnx"])
    ip_adapter = IPAdapterONNX(ip_path, args.device)

    face_emb_batch = face_embedding[np.newaxis].astype(np.float16)  # [1, 512]
    image_prompt_embeds = ip_adapter.project(face_emb_batch)  # [1, 16, 2048]
    print(f"  image_prompt_embeds shape: {image_prompt_embeds.shape}")

    timings["ip_adapter"] = time.time() - t0
    del ip_adapter
    gc.collect()

    # ── Stage 4: Diffusion Loop ───────────────────────────────────────────
    print(f"\n[Stage 4] Diffusion ({args.steps} LCM steps)...")
    t0 = time.time()

    # Concat text + image embeddings
    # LCM với guidance_scale=0 → single forward pass (no CFG)
    # Concat image_prompt_embeds vào text_embeds
    encoder_hidden = np.concatenate([text_embeds, image_prompt_embeds], axis=1)
    # Shape: [1, 77+16, 2048] = [1, 93, 2048]

    time_ids    = get_time_ids(H, W).astype(np.float16)
    pooled_fp16 = pooled.astype(np.float16)

    # Init latents
    latent_h, latent_w = H // 8, W // 8
    latents = rng.randn(1, 4, latent_h, latent_w).astype(np.float16)

    # Load models
    cn_path   = find_model("controlnet", ["model_int8.onnx", "model.onnx"])
    unet_path = find_model("unet", ["model_int8.onnx", "model.onnx"])

    controlnet = ControlNetONNX(cn_path, args.device)
    unet       = UNetONNX(unet_path, args.device)
    scheduler  = LCMSchedulerNP(num_steps=args.steps)

    for step_idx, t in enumerate(scheduler.timesteps):
        t_arr = np.array([t], dtype=np.int64)

        print(f"  Step {step_idx+1}/{args.steps} (t={t})...", end=" ", flush=True)
        t_step = time.time()

        # ControlNet forward
        down_res, mid_res = controlnet.forward(
            latents, t_arr,
            encoder_hidden,
            kps_image.astype(np.float16),
            args.cn_scale,
        )

        # UNet forward
        noise_pred = unet.forward(
            latents, t_arr,
            encoder_hidden,
            pooled_fp16, time_ids,
            down_res, mid_res,
        )

        # Scheduler step
        latents = scheduler.step(noise_pred, t, latents)

        print(f"{time.time()-t_step:.1f}s")

    timings["diffusion"] = time.time() - t0

    del controlnet, unet
    gc.collect()

    # ── Stage 5: VAE Decode ───────────────────────────────────────────────
    print(f"\n[Stage 5] VAE Decode...")
    t0 = time.time()

    vae_path = find_model("vae_decoder", ["model_fp16.onnx", "model_int8.onnx", "model.onnx"])
    vae = VAEDecoderONNX(vae_path, args.device)

    image_np = vae.decode(latents)  # [1, 3, H, W] in [0,1]
    print(f"  image shape: {image_np.shape}")

    timings["vae"] = time.time() - t0
    del vae
    gc.collect()

    # ── Save output ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    img_hwc = image_np[0].transpose(1, 2, 0)  # [H, W, 3]
    img_uint8 = (img_hwc * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(args.output)
    print(f"\n✅ Saved to: {args.output}")

    # ── Print timings ─────────────────────────────────────────────────────
    total = sum(timings.values())
    print(f"\n{'─'*40}")
    print(f"{'Stage':<20} {'Time':>8}")
    print(f"{'─'*40}")
    for stage, t in timings.items():
        pct = t / total * 100
        print(f"  {stage:<18} {t:>6.1f}s  ({pct:.0f}%)")
    print(f"{'─'*40}")
    print(f"  {'TOTAL':<18} {total:>6.1f}s")
    print(f"\nEstimated on mobile (5-10x slower): {total*5/60:.0f}-{total*10/60:.0f} min")


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.face_image):
        print(f"Face image not found: {args.face_image}")
        print("Creating a test face image...")
        os.makedirs("examples", exist_ok=True)
        # Tạo dummy image để test pipeline shape
        dummy = Image.new("RGB", (512, 512), color=(128, 100, 90))
        dummy.save("examples/face.jpg")
        args.face_image = "examples/face.jpg"
        print("  Created examples/face.jpg (use a real face image for quality test)")

    run_pipeline(args)
