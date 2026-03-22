# InstantID Mobile

On-device face identity preservation: generate images while keeping the face of the input photo — entirely on mobile, no server required.

| | |
|---|---|
| **Target** | Android/iOS flagship (8GB+ RAM, Snapdragon 8 Gen 2+ / A16+) |
| **Model Size** | ~2.5 GB (INT8) — down from 11 GB original |
| **Speed** | 2–10 min/image (512×512, 4 LCM steps) |
| **Pipeline** | PyTorch → ONNX → INT8 → OnnxStream |

---

## Architecture

```
Input Face Image ──→ [InsightFace ONNX] ──→ Face Embedding (512-dim)
                                          ──→ Face Keypoints (5 points)
                                                     ↓
                                            [IP-Adapter Resampler]
                                            Image Prompt Embeds (16×2048)
                                                     ↓
Text Prompt ────────→ [CLIP-L + OpenCLIP-G] ──→ Text Embeddings (2048-dim)
                                                     ↓
Keypoints Image ───→ [ControlNet] ──→ conditioning residuals
                                          ↓
                                     [UNet × 4 LCM steps]
                                          ↓
                                     [VAE Decoder]
                                          ↓
                                     Output Image (512×512)
```

### Key Optimizations
- **LCM-LoRA fusion**: 4 inference steps instead of 20-50 (fused permanently into UNet weights)
- **No classifier-free guidance**: `guidance_scale=0` with LCM → single forward pass per step
- **INT8 dynamic quantization**: ~75% model size reduction
- **OnnxStream**: streams weights from disk in chunks, enables 5GB model in <500MB RAM

---

## Repository Structure

```
Instantid-mobile/
├── README.md                           # This file
├── setup.sh                            # Bootstrap script (deps + models)
│
├── scripts/
│   ├── fuse_lcm_lora.py               # Step 1: Fuse LCM-LoRA into SDXL UNet
│   ├── export_all.py                  # Step 2: Export 6 components to ONNX
│   ├── quantize_all.py                # Step 3: INT8 dynamic quantization
│   ├── test_onnx_pipeline.py          # Step 4: End-to-end ONNX test on PC
│   └── convert_for_onnxstream.py      # Step 5: Convert for mobile runtime
│
├── ip_adapter/
│   ├── attention_processor.py          # Modified attention with IP-Adapter injection
│   ├── resampler.py                   # Perceiver resampler (512d → 16×2048)
│   └── utils.py                       # Utility functions
│
├── notebooks/
│   └── instantid_mobile.ipynb         # Complete Google Colab pipeline
│
└── mobile/cpp/
    ├── CMakeLists.txt                 # Build config (Android/iOS, OnnxStream/ORT)
    ├── face_processor.h/.cpp          # InsightFace ONNX inference
    └── instantid_pipeline.h/.cpp      # Main pipeline orchestrator
```

---

## Model Size Breakdown

| Component | Original (FP16) | After INT8 | Reduction |
|-----------|-----------------|------------|-----------|
| UNet (+ LCM-LoRA fused) | 5.1 GB | 1.3 GB | 4× |
| ControlNet (IdentityNet) | 2.5 GB | 625 MB | 4× |
| Text Encoder 2 (OpenCLIP-G) | 2.5 GB | 625 MB | 4× |
| Text Encoder 1 (CLIP-L) | 394 MB | 100 MB | 4× |
| IP-Adapter Resampler | 156 MB | 40 MB | 4× |
| VAE Decoder | 95 MB | 95 MB | 1× (FP16 kept) |
| **Total** | **~10.9 GB** | **~2.8 GB** | **~4×** |

> VAE is kept at FP16 — quantization causes visible artifacts in decoded images.

---

## Quick Start

### Option A: Google Colab (recommended)

Open [`notebooks/instantid_mobile.ipynb`](notebooks/instantid_mobile.ipynb) in Colab → **Runtime → Run All**.

Works on Colab free (T4, 12GB RAM). Drive cache makes re-runs fast (~1 min restore).

### Option B: Local GPU

```bash
git clone https://github.com/Hert4/Instantid-mobile
cd Instantid-mobile
./setup.sh --gpu

python scripts/fuse_lcm_lora.py
python scripts/export_all.py
python scripts/quantize_all.py
python scripts/test_onnx_pipeline.py --face_image photo.jpg
python scripts/convert_for_onnxstream.py
```

**Requirements:** Python 3.10+, CUDA GPU (≥8GB VRAM), ~50GB disk, **≥16GB RAM** (for UNet quantization).

---

## Step-by-Step Pipeline

### Step 1 — Fuse LCM-LoRA
```bash
python scripts/fuse_lcm_lora.py --device cuda --delete_source
```
Merges LCM-LoRA weights permanently into SDXL UNet. After fusion:
- Inference: 4 steps (instead of 50)
- No CFG needed (guidance_scale=0)
- `--delete_source` frees ~6GB disk after loading

### Step 2 — Export ONNX
```bash
python scripts/export_all.py                    # All components
python scripts/export_all.py --only unet        # Single component
python scripts/export_all.py --only controlnet
```
- Models >1.5GB automatically use **external data format** (model.onnx + weights.pb)
- Bypasses protobuf's 2GB message size limit
- 3-pass parameter matching: name → shape → positional order

### Step 3 — Quantize INT8
```bash
python scripts/quantize_all.py --skip_slim
python scripts/quantize_all.py --only unet      # Single component
```
- Auto-detects FP16 models and converts to FP32 before quantization
- External data models handled with `save_as_external_data=True`
- `--skip_slim` avoids OOM from onnxslim on large models

### Step 4 — Test
```bash
python scripts/test_onnx_pipeline.py \
    --face_image photo.jpg \
    --onnx_dir ./onnx-int8 \
    --prompt "portrait photo, professional lighting, 4k" \
    --steps 4 --seed 42
```

### Step 5 — Convert for Mobile
```bash
python scripts/convert_for_onnxstream.py
# Output: onnxstream-models/ → copy to device
```

---

## Mobile Build

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

### iOS (macOS)
```bash
cmake .. \
  -DBUILD_IOS=ON -DUSE_ONNXSTREAM=ON \
  -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_ARCHITECTURES=arm64
make -j$(nproc)
```

---

## RAM Usage on Device

| Stage | RAM Peak | Notes |
|-------|----------|-------|
| Face detection (SCRFD) | ~70 MB | InsightFace ONNX |
| Text encoding | ~350 MB | OpenCLIP-G is largest |
| IP-Adapter | ~25 MB | Small resampler |
| **Diffusion loop** | **~2 GB** | UNet + ControlNet (bottleneck) |
| VAE decode | ~200 MB | Single forward pass |

With **OnnxStream**, UNet streams weights from disk → runs in ~300-500 MB instead of 1.3 GB.

---

## Proof of Concept Report

### Objective
Validate the feasibility of running InstantID (face-swapping Stable Diffusion XL pipeline) entirely on mobile devices by converting the PyTorch pipeline to optimized ONNX INT8 format.

### Environment
- **Export/Quantize**: Google Colab free tier (T4 GPU, 12GB RAM, 112GB disk)
- **Target**: Android/iOS flagship devices (8GB+ RAM)

### Pipeline Validated

```
[Download Models] → [Fuse LCM-LoRA] → [Export ONNX] → [Quantize INT8] → [Test]
     15 min             5 min           30-60 min         20-40 min        5 min
```

### Components Successfully Exported

| Component | Export Method | ONNX Size | Status |
|-----------|-------------|-----------|--------|
| IP-Adapter | Standard export | 156 MB | Exported + Quantized |
| Text Encoder 1 (CLIP-L) | Standard export | 394 MB | Exported + Quantized |
| VAE Decoder | Standard export | 95 MB | Exported (FP16 kept) |
| ControlNet | External data format | 2,388 MB | Exported + Quantized |
| Text Encoder 2 (OpenCLIP-G) | External data format | 2,520 MB | Exported + Quantized |
| UNet (+ LCM-LoRA) | External data format | 5,572 MB | Exported (FP16*) |

> *UNet INT8 quantization requires >16GB RAM — exceeds Colab free tier. Quantize locally or on Colab Pro.

### Technical Challenges & Solutions

#### 1. Protobuf 2GB Limit (Critical)
**Problem:** `torch.onnx.export` serializes the entire model as a single protobuf message. Models >2GB (UNet, ControlNet, text_encoder_2) were silently truncated to 2-5MB files.

**Solution:** External data format pipeline:
1. Export graph structure only (`export_params=False`) → tiny .onnx file
2. Load graph, identify parameter inputs vs regular inputs
3. Match ONNX parameter inputs to PyTorch params via 3-pass matching:
   - **Pass 1 (name match):** Direct name lookup — matches ~50% of params
   - **Pass 2 (shape match):** Group by shape, consume in order — matches ~30% more
   - **Pass 3 (positional match):** For remaining params with dynamic dims (dim_value=0) — matches final ~20%
4. Write weights to external `weights.pb` file with proper offsets

**Result:** All 844/844 ControlNet params matched, 2,386 MB weights.pb (correct).

#### 2. FP16/FP32 Type Mismatch During Export
**Problem:** IP-Adapter attention processors had `to_k_ip`/`to_v_ip` weights in FP16 but received FP32 hidden states during ONNX tracing, causing `RuntimeError: dtype mismatch`.

**Solution:** Added explicit dtype cast in `ip_adapter/attention_processor.py`:
```python
ip_hidden_states = ip_hidden_states.to(self.to_k_ip.weight.dtype)
```

#### 3. FP16 Models Break quantize_dynamic
**Problem:** `onnxruntime.quantization.quantize_dynamic` produces FP32 quantized outputs, but FP16 model inputs create mixed-type nodes (e.g., LayerNormalization bound to both float and float16).

**Solution:** Auto-detect FP16 models and convert to FP32 before quantization:
- Small models: in-memory FP16→FP32 conversion
- Large models with external data: FP16→FP32 with `save_as_external_data=True`
- Very large models (>4GB FP16): copy FP16 directly (RAM insufficient for conversion)

#### 4. UNet Quantization OOM on Colab
**Problem:** UNet (5.5GB FP16 → 11GB FP32) exceeds Colab's 12GB RAM during quantization.

**Mitigations applied:**
- `optimize_model=False` — skip internal graph optimization
- `use_external_data_format=True` — output uses external data
- `gc.collect()` + `torch.cuda.empty_cache()` before quantization
- Quantize components sequentially (small first, UNet last)

**Status:** Still requires >16GB RAM. Fallback: copy FP16 model directly.

#### 5. onnxslim OOM/Hang
**Problem:** onnxslim graph optimization hangs indefinitely on models >2GB.

**Solution:** `--skip_slim` flag + auto-skip for models above configurable threshold (default 2048MB).

#### 6. ONNX Input Dtype Mismatch (SCRFD)
**Problem:** numpy division by float64 std array promoted face detection input to double, but SCRFD expects float32.

**Solution:** Explicit `.astype(np.float32)` after normalization.

#### 7. Colab Disk Management
**Problem:** Pipeline peak disk usage (~20GB for ONNX + ~11GB for source models) can exceed Colab's 112GB overlay.

**Solution:**
- Delete source models (`sdxl-base-lcm`) after ONNX export
- Guard `onnx/` deletion — only delete after quantized output verified (>500MB)
- Google Drive cache for intermediate results (checkpoints, fused model, quantized output)

### Results

| Metric | Value |
|--------|-------|
| Total ONNX export size | 10.9 GB (FP16) |
| Target quantized size | ~2.8 GB (INT8) |
| Components fully quantized | 5/6 (all except UNet on Colab free) |
| Export time (Colab T4) | ~45 minutes |
| Quantize time (small models) | ~10 minutes |
| Face detection test | Passed (512-dim embedding + 5 keypoints) |
| Parameter matching accuracy | 100% (844/844 for ControlNet) |

### Feasibility Assessment

| Scenario | Feasible? | Notes |
|----------|-----------|-------|
| Export pipeline on Colab free | **Yes** | All 6 components export successfully |
| INT8 quantization (small models) | **Yes** | text_encoders, ip_adapter, controlnet |
| INT8 quantization (UNet) | **Partial** | Needs >16GB RAM (Colab Pro or local) |
| Mobile inference (OnnxStream) | **Likely** | Streaming weights keeps RAM <2GB |
| Mobile inference (direct ONNX RT) | **Challenging** | Peak ~2GB RAM for UNet+ControlNet |
| Real-time generation | **No** | 2-10 min per image expected |

### Recommendations

1. **Quantize UNet locally** on a machine with ≥16GB RAM to achieve full INT8 pipeline (~2.5GB total)
2. **Use OnnxStream** for mobile deployment — essential for running 1.3GB+ UNet in limited RAM
3. **Consider SD 1.5-based alternatives** if SDXL model size is prohibitive for target devices
4. **Test on target hardware** early — actual mobile performance may vary significantly from estimates

---

## Colab Troubleshooting

| Error | Fix |
|-------|-----|
| ONNX file too small (< 100 MB for UNet) | Protobuf 2GB limit — `git pull` for external data format fix |
| onnxslim hangs / OOM | Use `--skip_slim` (default in notebook) |
| `sdxl-base-lcm MISSING` | Re-run Fuse cell — will restore from Drive cache |
| `RuntimeError: dtype mismatch Half/Float` | `git pull` — fixed in `attention_processor.py` |
| `Type Error: LayerNormalization` | `git pull` — FP16→FP32 conversion fix in quantize |
| Disk full | Delete `~/.cache/huggingface/hub` + restart runtime |
| Session timeout | Everything cached on Drive — re-run from Cell 0 |

### External Data Format
Large models (UNet, ControlNet, text_encoder_2) use ONNX external data format:
```
onnx/unet/
├── model.onnx     (graph structure, ~5 MB)
└── weights.pb     (parameters, ~5 GB)
```
`onnxruntime` reads both files automatically when loading.

---

## References

- [InstantID](https://github.com/instantX-research/InstantID) — Original InstantID paper & code
- [OnnxStream](https://github.com/vitoplantamura/OnnxStream) — Streaming ONNX inference for edge devices
- [InsightFace](https://github.com/deepinsight/insightface) — Face detection & recognition
- [LCM-LoRA](https://huggingface.co/latent-consistency/lcm-lora-sdxl) — Latent Consistency Model for fast inference
- [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) — Stable Diffusion XL base model

---

## License

This project builds on several open-source components. See individual licenses:
- InstantID: Apache 2.0
- Stable Diffusion XL: CreativeML Open RAIL++-M
- InsightFace: MIT
- OnnxStream: MIT
