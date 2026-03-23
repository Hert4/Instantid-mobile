"""
InstantID Mobile — Full Pipeline (Google Colab TPU v5e-1)
=========================================================
File này là SOURCE để convert sang .ipynb. KHÔNG chạy trực tiếp bằng python.

Cách convert sang notebook:
    cd notebooks/
    python convert_to_ipynb.py instantid_tpu_v5e.py
    # → tạo ra instantid_tpu_v5e.ipynb

Hoặc dùng jupytext:
    pip install jupytext
    jupytext --to notebook instantid_tpu_v5e.py

─── Cấu trúc marker ───────────────────────────────────────────────────────────
  # %% [markdown]    → bắt đầu markdown cell
  # %%               → bắt đầu code cell
  (nội dung markdown viết sau dấu # , code cell viết thẳng)
────────────────────────────────────────────────────────────────────────────────

Pipeline:
    TPU v5e-1 (8 cores, 16 GB HBM/core × 8 = 128 GB total)
    Colab TPU runtime: JAX/XLA backend, PyTorch/XLA (torch_xla)

  Bước 0 : Mount Drive + Install deps (JAX TPU + PyTorch/XLA + ONNX tools)
  Bước 1 : Clone repo + download models
  Bước 2 : Fuse LCM-LoRA  (chạy trên TPU với torch_xla)
  Bước 3 : Export ONNX     (clone về CPU trước khi export)
  Bước 4 : Quantize INT8   (ONNX Runtime CPU — TPU không cần GPU)
  Bước 5 : Test pipeline
  Bước 6 : Save Drive

Lưu ý TPU v5e trên Colab:
  - torch_xla  : pip install torch~=2.3 torch_xla[tpu]~=2.3
  - JAX         : pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
  - ONNX export : torch.onnx.export không hỗ trợ XLA tensor → clone về CPU trước
  - HBM         : 16 GB/core × 8 cores; UNet cần shard qua nhiều core nếu batch > 1
  - Disk Colab  : ~100 GB (Colab Pro+); dùng Drive cache giống notebook T4
"""

# Tránh execute nhầm khi chạy python trực tiếp
import sys as _sys
if _sys.argv[0].endswith('instantid_tpu_v5e.py'):
    print("=" * 60)
    print("  File này là notebook SOURCE, không chạy trực tiếp.")
    print()
    print("  Để convert sang .ipynb:")
    print("    python convert_to_ipynb.py instantid_tpu_v5e.py")
    print()
    print("  Hoặc dùng jupytext:")
    print("    pip install jupytext")
    print("    jupytext --to notebook instantid_tpu_v5e.py")
    print("=" * 60)
    _sys.exit(0)

# =============================================================================
# NOTEBOOK SOURCE BẮT ĐẦU TỪ ĐÂY
# Các marker # %% và # %% [markdown] được convert_to_ipynb.py đọc để tạo cells
# =============================================================================

# %% [markdown]
# # InstantID Mobile — Full Export Pipeline (TPU v5e-1 · Google Colab)
#
# | Bước | Cell | Thời gian | Disk cần |
# |------|------|-----------|----------|
# | 0. Setup + Mount Drive | 0 | ~10 phút | — |
# | 1. Clone + Download models | 1 | ~10 phút | ~15 GB |
# | 2. Fuse LCM-LoRA (XLA) | 2 | ~3 phút | ~13 GB peak |
# | 3. Export ONNX (XLA→CPU) | 3 | ~20-40 phút | ~20 GB peak |
# | 4. Quantize INT8 | 4 | ~15-25 phút | ~10 GB peak |
# | 5. Test pipeline | 5 | ~5 phút | ~3 GB |
# | 6. Save Drive | 6 | ~5 phút | — |
#
# **Runtime:** Colab TPU v5e-1 (8 chips × 16 GB HBM = 128 GB total)
#
# **QUAN TRỌNG:** Chạy cells THEO THỨ TỰ. Không skip.

# %% [markdown]
# ---
# ## Cell 0 — Mount Drive + Install Dependencies

# %%
# ─── Cell 0 ──────────────────────────────────────────────────────────────────
from google.colab import drive  # type: ignore[import-untyped]
drive.mount('/content/drive')

DRIVE_CACHE = '/content/drive/MyDrive/instantid-mobile-cache'
import os
os.makedirs(f'{DRIVE_CACHE}/checkpoints', exist_ok=True)
os.makedirs(f'{DRIVE_CACHE}/sdxl-base-lcm', exist_ok=True)
os.makedirs(f'{DRIVE_CACHE}/onnx-int8', exist_ok=True)

import subprocess, sys

def pip(*args):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *args])

# ── JAX + TPU libtpu ──────────────────────────────────────────────────────────
print('Installing JAX + TPU support...')
pip('jax[tpu]', '-f',
    'https://storage.googleapis.com/jax-releases/libtpu_releases.html')

# ── PyTorch/XLA — phiên bản khớp Colab TPU v5e (2.3.x) ─────────────────────
print('Installing PyTorch + torch_xla for TPU v5e...')
pip('torch~=2.3.0', 'torchvision~=0.18.0',
    '--index-url', 'https://download.pytorch.org/whl/cpu')
pip('torch_xla[tpu]~=2.3.0',
    '-f', 'https://storage.googleapis.com/libtpu-releases/index.html')

# ── Diffusion stack ───────────────────────────────────────────────────────────
print('Installing diffusion stack...')
pip('diffusers>=0.28.0', 'transformers>=4.40.0', 'huggingface_hub>=0.23.0',
    'accelerate>=0.30.0', 'safetensors', 'peft')

# ── ONNX tools ────────────────────────────────────────────────────────────────
print('Installing ONNX tools...')
pip('onnx', 'onnxscript', 'onnxruntime', 'onnxslim',
    'insightface', 'opencv-python', 'Pillow', 'scipy', 'psutil')

# ── Verify ────────────────────────────────────────────────────────────────────
import jax                              # type: ignore[import-untyped]
import torch
import torch_xla.core.xla_model as xm  # type: ignore[import-untyped]

print(f'\nJAX  {jax.__version__}')
print(f'JAX devices : {jax.devices()}')      # phải thấy TpuDevice
print(f'PyTorch     : {torch.__version__}')
xla_dev = xm.xla_device()
print(f'XLA device  : {xla_dev}')            # xla:0  (TPU)
print(f'TPU cores   : {xm.xrt_world_size()}')

import onnxruntime, diffusers
print(f'onnxruntime : {onnxruntime.__version__}')
print(f'diffusers   : {diffusers.__version__}')

print('\nDisk:')
os.system('df -h /content | tail -1')

# %% [markdown]
# ---
# ## Cell 1 — Clone Repo + Download Models

# %%
# ─── Cell 1a — Clone repo ────────────────────────────────────────────────────
import os
os.chdir('/content')

if not os.path.exists('Instantid-mobile/.git'):
    os.system('git clone https://github.com/Hert4/Instantid-mobile')
else:
    os.system('cd Instantid-mobile && git pull origin main')

os.chdir('/content/Instantid-mobile')
PROJ = os.getcwd()
print(f'Working dir: {PROJ}')

# %%
# ─── Cell 1b — Drive cache helpers ───────────────────────────────────────────
import os, shutil

DRIVE_CACHE = '/content/drive/MyDrive/instantid-mobile-cache'

def dir_size_mb(path):
    if not os.path.exists(path):
        return 0
    total = 0
    for dp, _, fns in os.walk(path):
        for f in fns:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total / 1024 ** 2

def restore_from_drive(name, min_mb=10):
    drive_path = f'{DRIVE_CACHE}/{name}'
    local_path = f'/content/Instantid-mobile/{name}'
    drive_mb = dir_size_mb(drive_path)
    local_mb = dir_size_mb(local_path)
    if drive_mb > min_mb and local_mb < min_mb:
        print(f'  Restoring {name} from Drive ({drive_mb:.0f} MB)...')
        os.system(f'cp -r "{drive_path}" "{local_path}"')
        print(f'  ✅ Restored {name}')
        return True
    elif local_mb >= min_mb:
        print(f'  ✅ {name} already local ({local_mb:.0f} MB)')
        return True
    return False

def save_to_drive(name, min_mb=10):
    src = f'/content/Instantid-mobile/{name}'
    dst = f'{DRIVE_CACHE}/{name}'
    src_mb = dir_size_mb(src)
    dst_mb = dir_size_mb(dst)
    if src_mb < min_mb:
        print(f'  [skip] {name} too small ({src_mb:.0f} MB)')
        return
    if abs(src_mb - dst_mb) < min_mb:
        print(f'  [skip] {name} already on Drive ({dst_mb:.0f} MB)')
        return
    print(f'  Saving {name} → Drive ({src_mb:.0f} MB)...')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.system(f'cp -r "{src}" "{dst}"')
    print(f'  ✅ Cached {name}')

print('Drive helpers loaded ✅')

# %%
# ─── Cell 1c — Download model checkpoints ────────────────────────────────────
import os, glob
os.chdir('/content/Instantid-mobile')

os.makedirs('checkpoints/ControlNetModel', exist_ok=True)
os.makedirs('models/antelopev2', exist_ok=True)

restore_from_drive('checkpoints', min_mb=100)

from huggingface_hub import hf_hub_download, snapshot_download

# IP-Adapter
print('=== IP-Adapter ===')
if not os.path.exists('checkpoints/ip-adapter.bin'):
    hf_hub_download('InstantX/InstantID', 'ip-adapter.bin',
                    local_dir='./checkpoints')
print('  ✅ ip-adapter.bin')

# ControlNet
print('=== ControlNet ===')
if not os.path.exists('checkpoints/ControlNetModel/config.json'):
    snapshot_download('InstantX/InstantID',
                      allow_patterns=['ControlNetModel/*'],
                      local_dir='./checkpoints')
print('  ✅ ControlNetModel/')

# LCM-LoRA
print('=== LCM-LoRA ===')
if not os.path.exists('checkpoints/pytorch_lora_weights.safetensors'):
    hf_hub_download('latent-consistency/lcm-lora-sdxl',
                    'pytorch_lora_weights.safetensors',
                    local_dir='./checkpoints')
print('  ✅ pytorch_lora_weights.safetensors')

save_to_drive('checkpoints')

# InsightFace antelopev2
print('\n=== InsightFace ===')
scrfd_files = glob.glob('models/**/scrfd_10g_bnkps.onnx', recursive=True)
if not scrfd_files:
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name='antelopev2', root='./models',
                           providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        print('  ✅ InsightFace (via insightface package)')
    except Exception as e1:
        print(f'  auto-download failed: {e1}')
        try:
            for f in ['scrfd_10g_bnkps.onnx', 'glintr100.onnx']:
                hf_hub_download('DIAMONIK7777/antelopev2', f,
                                local_dir='./models/antelopev2')
            print('  ✅ InsightFace (HF fallback)')
        except Exception as e2:
            print(f'  ⚠️  {e2}')
else:
    print(f'  ✅ Already exists: {scrfd_files[0]}')

# SDXL base (chỉ download nếu chưa có sdxl-base-lcm trên Drive)
print('\n=== SDXL base ===')
if not os.path.exists('sdxl-base-lcm/model_index.json') and \
        not restore_from_drive('sdxl-base-lcm', min_mb=1000):
    if not os.path.exists('sdxl-base/model_index.json'):
        print('  Downloading SDXL base (~6.5 GB)...')
        snapshot_download(
            'stabilityai/stable-diffusion-xl-base-1.0',
            local_dir='./sdxl-base',
            ignore_patterns=['*.bin', '*.ckpt', 'flax_model*', 'tf_model*',
                             '*.onnx*', '*.msgpack', '*.xml', '*.pb'],
        )
    print('  ✅ sdxl-base downloaded')

os.system('df -h /content | tail -1')

# %% [markdown]
# ---
# ## Cell 2 — Fuse LCM-LoRA trên TPU (torch_xla)
#
# Merge LCM-LoRA vào SDXL UNet.
# Dùng XLA device để tận dụng HBM của TPU v5e.
# TPU v5e native **bfloat16** — không dùng float16 khi load lên TPU.

# %%
# ─── Cell 2 — Fuse LCM-LoRA (torch_xla) ─────────────────────────────────────
import os, shutil, gc
import torch
import torch_xla.core.xla_model as xm  # type: ignore[import-untyped]

os.chdir('/content/Instantid-mobile')

if os.path.exists('sdxl-base-lcm/model_index.json'):
    print('✅ sdxl-base-lcm đã tồn tại — skip fuse')
elif restore_from_drive('sdxl-base-lcm', min_mb=1000):
    assert os.path.exists('sdxl-base-lcm/model_index.json'), 'Restore failed!'
else:
    assert os.path.exists('sdxl-base/model_index.json'), \
        'sdxl-base missing! Chạy lại Cell 1.'

    xla_dev = xm.xla_device()
    print(f'Fusing on device: {xla_dev}')
    print(f'TPU cores available: {xm.xrt_world_size()}')

    from diffusers import StableDiffusionXLPipeline

    print('  Loading SDXL pipeline...')
    pipe = StableDiffusionXLPipeline.from_pretrained(
        './sdxl-base',
        torch_dtype=torch.bfloat16,   # native trên TPU v5e
        use_safetensors=True,
    )
    # Chuyển UNet lên TPU HBM
    pipe.unet = pipe.unet.to(xla_dev)
    xm.mark_step()                    # flush XLA computation graph

    print('  Loading + fusing LCM-LoRA...')
    pipe.load_lora_weights('./checkpoints/pytorch_lora_weights.safetensors',
                           adapter_name='lcm')
    pipe.fuse_lora(lora_scale=1.0)
    pipe.unload_lora_weights()
    xm.mark_step()

    # Clone về CPU trước khi save (torch.save không hỗ trợ XLA tensor trực tiếp)
    print('  Moving UNet back to CPU for save...')
    pipe.unet = pipe.unet.to('cpu').to(torch.float16)
    pipe.to('cpu')
    xm.mark_step()

    print('  Saving sdxl-base-lcm...')
    pipe.save_pretrained('./sdxl-base-lcm')

    del pipe
    gc.collect()
    xm.mark_step()

    assert os.path.exists('sdxl-base-lcm/model_index.json'), 'Fuse FAILED!'
    save_to_drive('sdxl-base-lcm', min_mb=1000)

# Cleanup sdxl-base gốc
if os.path.exists('sdxl-base'):
    shutil.rmtree('sdxl-base', ignore_errors=True)
    print('Freed sdxl-base')

print(f'\n✅ Fuse OK!  sdxl-base-lcm = {dir_size_mb("sdxl-base-lcm"):.0f} MB')
os.system('df -h /content | tail -1')

# %% [markdown]
# ---
# ## Cell 3 — Export ONNX (XLA → CPU clone)
#
# TPU không hỗ trợ `torch.onnx.export` trực tiếp trên XLA tensor.
#
# **Giải pháp:** Truyền `--device cpu` vào `export_all.py` →
# script tự clone model về CPU trước khi gọi `torch.onnx.export`.
#
# TPU v5e-1 có 128 GB HBM tổng nên toàn bộ pipeline load được cùng lúc.

# %%
# ─── Cell 3a — Setup helpers ─────────────────────────────────────────────────
import os, gc, torch
import torch_xla.core.xla_model as xm  # type: ignore[import-untyped]

os.chdir('/content/Instantid-mobile')
os.makedirs('onnx', exist_ok=True)

def dir_total_mb(dirpath):
    if not os.path.isdir(dirpath):
        return 0
    return sum(
        os.path.getsize(os.path.join(dirpath, f))
        for f in os.listdir(dirpath)
        if os.path.isfile(os.path.join(dirpath, f))
    ) / 1024 ** 2

EXPECTED_SIZES = {
    'ip_adapter':     ('onnx/ip_adapter',    50),
    'controlnet':     ('onnx/controlnet',    2000),
    'text_encoder':   ('onnx/text_encoder',  300),
    'text_encoder_2': ('onnx/text_encoder_2',700),
    'vae':            ('onnx/vae_decoder',   80),
    'unet':           ('onnx/unet',          4000),
}

def verify_onnx(name):
    dirpath, min_mb = EXPECTED_SIZES[name]
    onnx_file = os.path.join(dirpath, 'model.onnx')
    if not os.path.exists(onnx_file):
        return False
    actual_mb = dir_total_mb(dirpath)
    if actual_mb < min_mb:
        print(f'  ⚠️  {name}: {actual_mb:.0f} MB < {min_mb} MB — will re-export')
        import shutil; shutil.rmtree(dirpath, ignore_errors=True)
        return False
    print(f'  ✅ {name}: {actual_mb:.0f} MB')
    return True

print('=== Current ONNX status ===')
for name in EXPECTED_SIZES:
    verify_onnx(name)

# %%
# ─── Cell 3b — Export tất cả components ──────────────────────────────────────
import os, gc
import torch_xla.core.xla_model as xm  # type: ignore[import-untyped]

os.chdir('/content/Instantid-mobile')
os.environ['INSTANTID_DEVICE'] = 'tpu'   # hint cho export_all.py

EXPORT_ORDER = {
    'ip_adapter':    ['ip_adapter'],
    'controlnet':    ['controlnet'],
    'text_encoders': ['text_encoder', 'text_encoder_2'],
    'vae':           ['vae'],
    'unet':          ['unet'],
}

for comp, verify_keys in EXPORT_ORDER.items():
    if all(verify_onnx(k) for k in verify_keys):
        continue

    print(f'\n{"="*60}')
    print(f'EXPORTING: {comp}')
    print(f'{"="*60}')

    # --device cpu : clone về CPU trước khi export (XLA workaround)
    ret = os.system(f'python scripts/export_all.py --only {comp} --device cpu')
    if ret != 0:
        print(f'  ⚠️  export_all.py exited with {ret}')

    for k in verify_keys:
        verify_onnx(k)

    xm.mark_step()   # flush XLA graph
    gc.collect()

# Summary
print(f'\n{"="*60}')
print('EXPORT SUMMARY')
print(f'{"="*60}')
total_mb = 0
for name, (dirpath, _) in EXPECTED_SIZES.items():
    mb = dir_total_mb(dirpath)
    if mb > 0:
        total_mb += mb
        print(f'  {name:20s}  {mb:>8.0f} MB')
print(f'  {"TOTAL":20s}  {total_mb:>8.0f} MB ({total_mb/1024:.1f} GB)')

os.system('df -h /content | tail -1')

# %% [markdown]
# ---
# ## Cell 4 — Quantize INT8
#
# Dynamic INT8 quantization chạy trên **CPU** (ONNX Runtime).
# TPU không tham gia bước này.
#
# TPU v5e có đủ RAM (200+ GB) để quantize UNet không bị OOM
# — khác Colab T4 (12 GB RAM) thường fail ở bước này.

# %%
# ─── Cell 4a — Cleanup trước quantize ────────────────────────────────────────
import os, shutil, gc
import torch_xla.core.xla_model as xm  # type: ignore[import-untyped]

os.chdir('/content/Instantid-mobile')

# Giải phóng disk — sdxl-base-lcm không cần sau khi export xong
for path in ['./sdxl-base-lcm',
             os.path.expanduser('~/.cache/huggingface/hub')]:
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
        print(f'Freed {path}')

xm.mark_step()
gc.collect()
os.system('df -h /content | tail -1')

# %%
# ─── Cell 4b — Quantize INT8 ─────────────────────────────────────────────────
import os, gc, glob

os.chdir('/content/Instantid-mobile')
os.makedirs('onnx-int8', exist_ok=True)

# Nhỏ trước → UNet cuối cùng
components = ['text_encoder', 'ip_adapter', 'vae',
              'text_encoder_2', 'controlnet', 'unet']

for comp in components:
    print(f'\n{"="*50}')
    print(f'Quantizing: {comp}')
    print(f'{"="*50}')
    gc.collect()
    ret = os.system(f'python scripts/quantize_all.py --skip_slim --only {comp}')
    if ret != 0:
        print(f'  ⚠️  quantize_all.py exited with {ret}')

    outs = (glob.glob(f'onnx-int8/{comp}*/*.onnx') +
            glob.glob(f'onnx-int8/{comp}*/*.pb') +
            glob.glob(f'onnx-int8/vae_decoder/*.onnx'))
    if outs:
        total = sum(os.path.getsize(f) for f in outs) / 1024 ** 2
        print(f'  → {comp}: {total:.0f} MB')
    else:
        print(f'  ⚠️  {comp}: no output found!')

print('\n' + '=' * 50)
print('Quantization complete!')
os.system('df -h /content | tail -1')

# %%
# ─── Cell 4c — Verify + cleanup float16 originals ────────────────────────────
import os, shutil

os.chdir('/content/Instantid-mobile')

print('=== Quantized ONNX files ===')
total_mb = 0
for root, _, files in os.walk('onnx-int8'):
    for f in sorted(files):
        if f.endswith(('.onnx', '.pb', '.onnx_data')):
            p = os.path.join(root, f)
            mb = os.path.getsize(p) / 1024 ** 2
            total_mb += mb
            rel = os.path.relpath(p, 'onnx-int8')
            print(f'  {rel:50s}  {mb:>8.0f} MB')

print(f'  {"TOTAL":50s}  {total_mb:>8.0f} MB ({total_mb/1024:.1f} GB)')

if total_mb > 500:
    if os.path.exists('onnx'):
        shutil.rmtree('onnx', ignore_errors=True)
        print('\nFreed onnx/ (float16 originals)')
else:
    print('\n⚠️  Output quá nhỏ — GIỮ LẠI onnx/. Chạy lại cell quantize!')

os.system('df -h /content | tail -1')

# %% [markdown]
# ---
# ## Cell 5 — Test Pipeline

# %%
# ─── Cell 5 — Test ONNX pipeline ─────────────────────────────────────────────
import os
os.chdir('/content/Instantid-mobile')
os.makedirs('test_images', exist_ok=True)

FACE_IMAGE = 'test_images/face.jpg'

if not os.path.exists(FACE_IMAGE):
    print('⚠️  Chưa có face image.')
    print('Upload bằng:')
    print('  from google.colab import files')
    print('  uploaded = files.upload()   # chọn face.jpg')
    print('  import shutil')
    print('  shutil.copy(list(uploaded.keys())[0], "test_images/face.jpg")')
    print('Sau đó chạy lại cell này.')
else:
    ret = os.system(
        f'python scripts/test_onnx_pipeline.py '
        f'--face_image "{FACE_IMAGE}" '
        f'--onnx_dir ./onnx-int8 '
        f'--insightface_dir ./models/antelopev2 '
        f'--prompt "portrait photo of a person, professional lighting, 4k, detailed" '
        f'--steps 4 --seed 42 '
        f'--output ./output/test_result.png '
        f'--verbose'
    )
    if ret == 0 and os.path.exists('./output/test_result.png'):
        from IPython.display import Image, display
        display(Image('./output/test_result.png', width=512))
        print('✅ Pipeline test PASSED!')
    else:
        print('❌ Test failed — kiểm tra errors trên')

# %% [markdown]
# ---
# ## Cell 6 — Save lên Google Drive

# %%
# ─── Cell 6 — Save Drive ─────────────────────────────────────────────────────
import os
os.chdir('/content/Instantid-mobile')
DRIVE_CACHE = '/content/drive/MyDrive/instantid-mobile-cache'

save_to_drive('onnx-int8', min_mb=50)

if os.path.exists('onnxstream-models'):
    save_to_drive('onnxstream-models', min_mb=50)

if os.path.exists('output/test_result.png'):
    os.makedirs(f'{DRIVE_CACHE}/output', exist_ok=True)
    os.system(f'cp output/test_result.png "{DRIVE_CACHE}/output/"')
    print('  ✅ Saved test_result.png')

print(f'\n✅ Tất cả đã lưu tại: {DRIVE_CACHE}/')
os.system(f'du -sh "{DRIVE_CACHE}"/* 2>/dev/null')

# %% [markdown]
# ---
# ## TPU v5e — Lưu ý khi chạy trên Colab
#
# | Vấn đề | Giải pháp |
# |--------|-----------|
# | `torch.onnx.export` lỗi với XLA tensor | Dùng `--device cpu` trong export_all.py — script clone về CPU trước |
# | `xm.mark_step()` sau mỗi bước lớn | Đã có trong notebook — flush XLA computation graph |
# | bfloat16 thay vì float16 | TPU v5e native bfloat16; script fuse dùng `torch.bfloat16` rồi cast `float16` khi save |
# | `xrt_world_size()` = 8 (8 chips) | Pipeline chạy trên 1 chip (xla:0); sharding multi-chip cần xla_fsdp (không cần cho export) |
# | Disk Colab TPU ~100 GB | Giống Colab Pro+ — Drive cache được dùng để tránh tải lại |
# | JAX + PyTorch/XLA cùng lúc | Cả 2 có thể dùng TPU; trong notebook này chỉ dùng torch_xla |

# %%
# ─── (Optional) Pack output để download về máy ───────────────────────────────

# import shutil
# shutil.make_archive('/content/instantid-int8', 'zip',
#                     '/content/Instantid-mobile', 'onnx-int8')
# from google.colab import files
# files.download('/content/instantid-int8.zip')