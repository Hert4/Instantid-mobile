#!/usr/bin/env bash
# =============================================================================
# setup.sh — Bootstrap toàn bộ môi trường cho InstantID Mobile
#
# Chạy script này 1 lần trên PC/server có GPU để:
#   1. Tạo cấu trúc thư mục
#   2. Clone các repos cần thiết (OnnxStream, InsightFace, v.v.)
#   3. Cài Python dependencies
#   4. Tải model weights từ HuggingFace
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh [--skip-models] [--skip-deps] [--gpu]
#
# Flags:
#   --skip-models   Bỏ qua bước tải weights (nếu đã tải)
#   --skip-deps     Bỏ qua bước pip install
#   --gpu           Cài torch với CUDA support
# =============================================================================

set -euo pipefail

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; NC='\033[0m'; BOLD='\033[1m'

log()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*" >&2; exit 1; }
step() { echo -e "\n${BOLD}${BLUE}══ $* ══${NC}"; }

# ── Parse args ────────────────────────────────────────────────────────────────
SKIP_MODELS=false
SKIP_DEPS=false
USE_GPU=false

for arg in "$@"; do
  case $arg in
    --skip-models) SKIP_MODELS=true ;;
    --skip-deps)   SKIP_DEPS=true ;;
    --gpu)         USE_GPU=true ;;
    *) warn "Unknown arg: $arg" ;;
  esac
done

# ── Repo root = thư mục chứa script này ───────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
log "Repo root: $REPO_ROOT"

# =============================================================================
# STEP 1: Tạo cấu trúc thư mục
# =============================================================================
step "1/5 — Tạo cấu trúc thư mục"

mkdir -p \
  checkpoints/ControlNetModel \
  models/antelopev2 \
  sdxl-base \
  sdxl-base-lcm \
  onnx/{unet,vae_decoder,text_encoder,text_encoder_2,ip_adapter,controlnet} \
  onnx-int8/{unet,vae_decoder,text_encoder,text_encoder_2,ip_adapter,controlnet} \
  onnxstream-models \
  third_party \
  mobile/cpp \
  mobile/android \
  mobile/ios \
  output

log "Directories created"

# =============================================================================
# STEP 2: Clone third-party repos
# =============================================================================
step "2/5 — Clone third-party repos"

# ── OnnxStream ────────────────────────────────────────────────────────────────
# Engine để chạy SDXL trong <300MB RAM trên mobile (ARM)
if [ ! -d "third_party/OnnxStream/.git" ]; then
  log "Cloning OnnxStream..."
  git clone https://github.com/vitoplantamura/OnnxStream.git third_party/OnnxStream
else
  log "OnnxStream already cloned — pulling latest"
  git -C third_party/OnnxStream pull --ff-only || warn "Pull failed, using existing"
fi

# ── InsightFace source (cho reference implementation) ────────────────────────
if [ ! -d "third_party/insightface/.git" ]; then
  log "Cloning insightface..."
  git clone --depth=1 https://github.com/deepinsight/insightface.git third_party/insightface
else
  log "insightface already cloned"
fi

# ── InstantID original (reference pipeline) ───────────────────────────────────
if [ ! -d "third_party/InstantID/.git" ]; then
  log "Cloning InstantID..."
  git clone --depth=1 https://github.com/instantX-research/InstantID.git third_party/InstantID
else
  log "InstantID already cloned"
fi

# Copy ip_adapter module vào root để import dễ
if [ ! -d "ip_adapter" ]; then
  cp -r third_party/InstantID/ip_adapter ./ip_adapter
  log "Copied ip_adapter module"
fi

log "All repos cloned"

# =============================================================================
# STEP 3: Python dependencies
# =============================================================================
step "3/5 — Cài Python dependencies"

if $SKIP_DEPS; then
  warn "Skipping pip install (--skip-deps)"
else
  # Torch — CPU hoặc CUDA
  if $USE_GPU; then
    log "Installing torch with CUDA 12.1..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  else
    log "Installing torch CPU-only..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  fi

  log "Installing core dependencies..."
  # Pin versions cho đồng bộ — huggingface_hub>=0.23 bỏ cached_download
  # diffusers>=0.28 đã dùng hf_hub_download thay thế
  pip install \
    "diffusers>=0.28.0" \
    "transformers>=4.40.0" \
    "huggingface_hub>=0.23.0" \
    "accelerate>=0.30.0" \
    safetensors \
    peft \
    insightface \
    onnx \
    onnxscript \
    onnxruntime \
    onnxslim \
    "optimum[exporters]" \
    opencv-python \
    Pillow \
    numpy \
    tqdm \
    scipy

  # onnxruntime-gpu nếu có GPU
  if $USE_GPU; then
    pip install onnxruntime-gpu || warn "onnxruntime-gpu failed, using CPU version"
  fi

  log "Dependencies installed"
fi

# =============================================================================
# STEP 4: Tải model weights
# =============================================================================
step "4/5 — Tải model weights"

if $SKIP_MODELS; then
  warn "Skipping model download (--skip-models)"
else
  python3 - <<'PYEOF'
import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

REPO_ROOT = Path(__file__).parent if '__file__' in dir() else Path('.')

def download(repo_id, filename, local_dir, repo_type="model"):
    dest = Path(local_dir) / filename
    if dest.exists():
        print(f"  [skip] {filename} already exists")
        return str(dest)
    print(f"  [↓] Downloading {filename} from {repo_id}...")
    os.makedirs(local_dir, exist_ok=True)
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        repo_type=repo_type,
    )

print("\n── InstantID weights ──")
# ip-adapter (~1.6GB)
download("InstantX/InstantID", "ip-adapter.bin", "checkpoints")
# ControlNet IdentityNet (~2.5GB, safetensors)
download("InstantX/InstantID", "ControlNetModel/config.json", "checkpoints")
download(
    "InstantX/InstantID",
    "ControlNetModel/diffusion_pytorch_model.safetensors",
    "checkpoints",
)

print("\n── SDXL Base (FP16 variant) ──")
# Dùng snapshot_download để tải nguyên thư mục diffusers
if not Path("sdxl-base/unet").exists():
    print("  [↓] Downloading SDXL base model (may take a while ~6GB)...")
    snapshot_download(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        local_dir="sdxl-base",
        ignore_patterns=["*.bin", "flax_model*", "tf_model*"],  # Chỉ lấy safetensors
    )
else:
    print("  [skip] sdxl-base already exists")

print("\n── LCM-LoRA ──")
download(
    "latent-consistency/lcm-lora-sdxl",
    "pytorch_lora_weights.safetensors",
    "checkpoints",
)

print("\n── InsightFace antelopev2 ──")
# Chỉ tải 2 files cần thiết cho InstantID core:
# - scrfd_10g_bnkps.onnx (face detection, 16.9MB)
# - glintr100.onnx (face embedding, 261MB)
for fname in ["scrfd_10g_bnkps.onnx", "glintr100.onnx"]:
    try:
        download(
            "LPDoctor/insightface",
            f"models/antelopev2/{fname}",
            "models",
        )
    except Exception as e:
        print(f"  [warn] Could not download {fname}: {e}")
        print(f"  → Manual download: https://huggingface.co/LPDoctor/insightface/tree/main/models/antelopev2")

print("\n✅ All downloads complete!")
print("\nModel sizes:")
for root, dirs, files in os.walk("."):
    # Skip hidden, venv, third_party
    dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('third_party', '__pycache__', 'venv')]
    for f in files:
        if f.endswith(('.safetensors', '.bin', '.onnx')):
            p = Path(root) / f
            mb = p.stat().st_size / 1024**2
            print(f"  {p}: {mb:.0f} MB")
PYEOF

fi

# =============================================================================
# STEP 5: Build OnnxStream (desktop test build)
# =============================================================================
step "5/5 — Build OnnxStream (desktop)"

ONNXSTREAM_DIR="third_party/OnnxStream"

if [ ! -f "$ONNXSTREAM_DIR/build/src/sd" ]; then
  if command -v cmake &>/dev/null; then
    log "Building OnnxStream (desktop)..."
    mkdir -p "$ONNXSTREAM_DIR/build"
    cmake -S "$ONNXSTREAM_DIR/src" -B "$ONNXSTREAM_DIR/build" \
      -DMAX_SPEED=ON \
      -DCMAKE_BUILD_TYPE=Release
    cmake --build "$ONNXSTREAM_DIR/build" --parallel "$(nproc 2>/dev/null || echo 4)"
    log "OnnxStream built: $ONNXSTREAM_DIR/build/src/sd"
  else
    warn "cmake not found — skip OnnxStream build"
    warn "Install: sudo apt install cmake"
  fi
else
  log "OnnxStream already built"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${BOLD}${GREEN}══════════════════════════════════════════${NC}"
echo -e "${BOLD}  Setup complete! Next steps:${NC}"
echo -e "${BOLD}${GREEN}══════════════════════════════════════════${NC}"
echo ""
echo "  1. Fuse LCM-LoRA vào SDXL:"
echo "     python scripts/fuse_lcm_lora.py"
echo ""
echo "  2. Export tất cả sang ONNX:"
echo "     python scripts/export_all.py"
echo ""
echo "  3. Quantize INT8:"
echo "     python scripts/quantize_all.py"
echo ""
echo "  4. Test ONNX pipeline trên PC:"
echo "     python scripts/test_onnx_pipeline.py --face_image examples/face.jpg"
echo ""
echo "  5. Convert sang OnnxStream format:"
echo "     python scripts/convert_for_onnxstream.py"
echo ""
