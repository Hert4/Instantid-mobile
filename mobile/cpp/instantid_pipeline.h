#pragma once
/**
 * instantid_pipeline.h
 * =====================
 * On-device InstantID inference pipeline orchestrator.
 *
 * Manages the full pipeline:
 *   Stage 1: FaceProcessor  → face embedding + keypoints image
 *   Stage 2: TextEncoder    → text embeddings (CLIP-L + OpenCLIP-G)
 *   Stage 3: IPAdapter      → project face embedding → image prompt embeds
 *   Stage 4: DiffusionLoop  → UNet + ControlNet × N steps (LCM scheduler)
 *   Stage 5: VAEDecoder     → latent → RGB image
 *
 * KEY DESIGN: Load/Unload each model individually.
 * NEVER load all models at once — mobile RAM cannot fit ~2.5GB simultaneously.
 *
 * RAM timeline:
 *   Stage 1:  ~70MB  (InsightFace models)
 *   Stage 2:  ~350MB peak (text encoder 2 is largest)
 *   Stage 3:  ~25MB  (IP-Adapter resampler)
 *   Stage 4:  ~2GB   peak (UNet 1.3GB + ControlNet 625MB + intermediate tensors)
 *             ← BOTTLENECK: use OnnxStream streaming mode here
 *   Stage 5:  ~200MB (VAE decoder)
 *
 * Dependencies: OnnxStream or ONNX Runtime Mobile
 */

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "face_processor.h"

namespace instantid {

// ─── Config ───────────────────────────────────────────────────────────────────

struct PipelineConfig {
    // Model paths (all relative to models_dir)
    std::string models_dir = "models";

    // InsightFace
    std::string scrfd_model    = "antelopev2/scrfd_10g_bnkps_int8.onnx";
    std::string glintr_model   = "antelopev2/glintr100_int8.onnx";

    // OnnxStream model dirs (each contains model.yaml + weights_*.bin)
    std::string unet_dir          = "onnxstream/unet";
    std::string controlnet_dir    = "onnxstream/controlnet";
    std::string text_enc1_dir     = "onnxstream/text_encoder";
    std::string text_enc2_dir     = "onnxstream/text_encoder_2";
    std::string vae_dir           = "onnxstream/vae_decoder";
    std::string ip_adapter_dir    = "onnxstream/ip_adapter";

    // Tokenizer vocab files (from SDXL)
    std::string tokenizer1_vocab  = "tokenizer/vocab.json";
    std::string tokenizer1_merges = "tokenizer/merges.txt";
    std::string tokenizer2_vocab  = "tokenizer_2/vocab.json";
    std::string tokenizer2_merges = "tokenizer_2/merges.txt";

    // Generation params
    int   image_height    = 512;       // 512 for mobile, 1024 for high quality
    int   image_width     = 512;
    int   num_steps       = 4;         // LCM steps (4-8 recommended)
    float guidance_scale  = 0.f;       // 0 with LCM = no CFG = 2x faster
    float ip_adapter_scale   = 0.8f;  // Face preservation strength
    float controlnet_scale   = 0.8f;  // Keypoints conditioning strength
    int64_t seed          = 42;

    // Hardware
    bool use_nnapi    = false;   // Android NNAPI
    bool use_coreml   = false;   // iOS CoreML
    bool use_gpu      = false;   // GPU execution (if available)

    // Memory
    int onnxstream_cache_mb = 256;  // OnnxStream weight cache size
};

// ─── Progress callback ────────────────────────────────────────────────────────

struct ProgressInfo {
    enum class Stage {
        FaceDetection,
        TextEncoding,
        IPAdapter,
        Diffusion,
        VAEDecode,
        Done,
    };

    Stage stage;
    int   step        = 0;   // Current diffusion step (Stage::Diffusion only)
    int   total_steps = 0;
    float progress    = 0.f; // 0.0 → 1.0
    std::string message;
};

using ProgressCallback = std::function<void(const ProgressInfo&)>;

// ─── Result ───────────────────────────────────────────────────────────────────

struct GenerationResult {
    bool success = false;
    std::string error_message;

    // Output image: RGB uint8 [height × width × 3]
    std::vector<uint8_t> image_rgb;
    int image_height = 0;
    int image_width  = 0;

    // Timings (milliseconds)
    struct Timings {
        float face_detection  = 0;
        float text_encoding   = 0;
        float ip_adapter      = 0;
        float diffusion       = 0;
        float vae_decode      = 0;
        float total           = 0;
    } timings;
};

// ─── InstantIDPipeline ────────────────────────────────────────────────────────

class InstantIDPipeline {
public:
    explicit InstantIDPipeline(const PipelineConfig& config);
    ~InstantIDPipeline();

    // Non-copyable
    InstantIDPipeline(const InstantIDPipeline&) = delete;
    InstantIDPipeline& operator=(const InstantIDPipeline&) = delete;

    /**
     * Generate image preserving face identity.
     *
     * @param face_image_rgb  Input face photo [H × W × 3] uint8 RGB
     * @param face_img_h      Face image height
     * @param face_img_w      Face image width
     * @param prompt          Text prompt describing desired output
     * @param neg_prompt      Negative prompt (can be empty with LCM)
     * @param progress_cb     Optional progress callback (called from worker thread)
     */
    GenerationResult generate(
        const uint8_t* face_image_rgb,
        int face_img_h,
        int face_img_w,
        const std::string& prompt,
        const std::string& neg_prompt = "",
        ProgressCallback progress_cb  = nullptr
    );

    /**
     * Cancel ongoing generation.
     * Thread-safe. Sets cancel flag checked at each diffusion step.
     */
    void cancel();

    /**
     * Check if models are available (files exist).
     * Call before generate() to give user feedback.
     */
    bool models_available() const;

    /**
     * Get required disk space for all models.
     */
    size_t required_disk_bytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace instantid
