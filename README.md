# InstantID Mobile

Chạy InstantID (giữ nguyên khuôn mặt khi generate ảnh) hoàn toàn **on-device**, không cần server.

- Target: Android/iOS flagship (8GB+ RAM, Snapdragon 8 Gen 2+ / A16+)
- Size sau optimize: ~2.5GB (từ 11GB gốc)
- Tốc độ ước tính: 2–10 phút/ảnh (512×512, 4 steps LCM)

---

## Cách hoạt động

```
Ảnh mặt người  →  [InsightFace]  →  face embedding (512d)
                                  →  keypoints image
Prompt text    →  [CLIP]         →  text embeddings
                                          ↓
                              [UNet + ControlNet] × 4 steps
                                          ↓
                                   [VAE Decoder]
                                          ↓
                                    Ảnh output
```

---

## Cấu trúc repo

```
instantid-mobile/
├── setup.sh                      # Tải deps + models (chạy 1 lần)
├── scripts/
│   ├── fuse_lcm_lora.py          # Bước 1: gộp LCM-LoRA vào UNet
│   ├── export_all.py             # Bước 2: export sang ONNX
│   ├── quantize_all.py           # Bước 3: nén INT8 (~75% nhỏ hơn)
│   ├── test_onnx_pipeline.py     # Bước 4: test trên PC
│   └── convert_for_onnxstream.py # Bước 5: convert cho mobile
└── mobile/cpp/
    ├── face_processor.h/.cpp     # InsightFace inference
    ├── instantid_pipeline.h/.cpp # Pipeline chính
    └── CMakeLists.txt            # Build Android / iOS
```

---

## Setup (chạy trên PC/server có GPU)

**Yêu cầu:** Python 3.10+, CUDA GPU (≥8GB VRAM), ~50GB disk

> RTX 3070 / 4070 (8–12GB VRAM) đủ dùng — export từng phần và dùng `--device cpu` nếu VRAM không đủ load toàn bộ model cùng lúc.

```bash
git clone https://github.com/Hert4/Instantid-mobile
cd Instantid-mobile

# Tải deps + clone OnnxStream + tải tất cả model weights (~30GB)
./setup.sh --gpu

# Nếu đã có models rồi, bỏ qua bước tải:
./setup.sh --skip-models
```

`setup.sh` tự động tải: SDXL base 1.0, InstantID weights, LCM-LoRA, InsightFace antelopev2.

---

## Chạy trên Google Colab

Colab free (T4 16GB) hoặc Pro (A100 40GB) đều đủ dùng.

**1. Tạo notebook mới, chọn Runtime → T4 GPU, rồi chạy:**

```python
# Cell 0 — Fix version conflict trước (Colab cài sẵn diffusers cũ)
!pip install -q "diffusers>=0.28.0" "huggingface_hub>=0.23.0" "transformers>=4.40.0"
import importlib, diffusers
importlib.reload(diffusers)
```

```python
# Cell 1 — Clone repo + setup
!git clone https://github.com/Hert4/Instantid-mobile
%cd Instantid-mobile
!./setup.sh --gpu
```

```python
# Cell 2 — Fuse LCM-LoRA
!python scripts/fuse_lcm_lora.py
```

```python
# Cell 3 — Export ONNX (làm từng phần để tránh OOM)
!python scripts/export_all.py --only text_encoders
!python scripts/export_all.py --only ip_adapter
!python scripts/export_all.py --only controlnet
!python scripts/export_all.py --only vae
!python scripts/export_all.py --only unet   # cái này nặng nhất, ~8GB VRAM
```

```python
# Cell 4 — Quantize
!python scripts/quantize_all.py
```

```python
# Cell 5 — Convert cho mobile
!python scripts/convert_for_onnxstream.py
```

```python
# Cell 6 — Tải về máy (zip folder onnxstream-models)
import shutil
shutil.make_archive("onnxstream-models", "zip", "onnxstream-models")

from google.colab import files
files.download("onnxstream-models.zip")
```

**Lưu ý khi dùng Colab:**

- Session tự disconnect sau ~90 phút idle — dùng Google Drive để lưu checkpoint giữa chừng:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  # Sau mỗi bước nặng, copy kết quả sang Drive:
  !cp -r onnx/ /content/drive/MyDrive/instantid-mobile/
  ```
- Export ONNX mất ~1–2 giờ — bật "Keep awake" trên trình duyệt hoặc dùng Colab Pro
- Colab free giới hạn ~12GB RAM + 16GB VRAM — nếu OOM thì restart runtime và chạy lại từ bước bị lỗi (các bước trước đã skip tự động)

---

## Chạy từng bước

### Bước 1 — Fuse LCM-LoRA
Gộp LoRA vào UNet để inference chỉ cần 4 steps thay vì 50.
```bash
python scripts/fuse_lcm_lora.py
```

### Bước 2 — Export ONNX
```bash
python scripts/export_all.py

# Chỉ export 1 phần nếu cần:
python scripts/export_all.py --only unet
python scripts/export_all.py --only controlnet
```

### Bước 3 — Quantize INT8
```bash
python scripts/quantize_all.py
```

| Model | Gốc | Sau INT8 |
|---|---|---|
| UNet | 5.1 GB | 1.3 GB |
| ControlNet | 2.5 GB | 625 MB |
| Text Encoders | 1.4 GB | 350 MB |
| VAE | 160 MB | 160 MB (giữ nguyên) |
| IP-Adapter | 80 MB | 20 MB |
| **Tổng** | **~9.3 GB** | **~2.5 GB** |

### Bước 4 — Test trên PC
Verify output trước khi port mobile.
```bash
python scripts/test_onnx_pipeline.py --face_image photo.jpg
```

### Bước 5 — Convert cho mobile
```bash
python scripts/convert_for_onnxstream.py
# Output: onnxstream-models/  ← copy folder này vào device
```

---

## Build mobile

### Android
```bash
cd mobile/cpp && mkdir build && cd build

cmake .. \
  -DBUILD_ANDROID=ON -DUSE_ONNXSTREAM=ON \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=26 \
  -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)
```

### iOS (chạy trên macOS)
```bash
cmake .. \
  -DBUILD_IOS=ON -DUSE_ONNXSTREAM=ON \
  -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_ARCHITECTURES=arm64

make -j$(nproc)
```

---

## RAM usage trên device

| Giai đoạn | RAM peak |
|---|---|
| Face detection | ~70 MB |
| Text encoding | ~350 MB |
| IP-Adapter | ~25 MB |
| **Diffusion loop** | **~2 GB** ← bottleneck |
| VAE decode | ~200 MB |

OnnxStream stream weights từ disk → UNet chạy trong ~300–500 MB thay vì 1.3 GB load toàn bộ.

---

## Tham khảo

- [InstantID](https://github.com/instantX-research/InstantID)
- [OnnxStream](https://github.com/vitoplantamura/OnnxStream)
- [InsightFace](https://github.com/deepinsight/insightface)
- [LCM-LoRA](https://huggingface.co/latent-consistency/lcm-lora-sdxl)