/**
 * instantid_pipeline.cpp
 * =======================
 * Implementation of InstantIDPipeline.
 *
 * Build requirements:
 *   - C++17
 *   - OnnxStream (third_party/OnnxStream/src/) or onnxruntime-android/ios
 *   - OpenCV (optional, for image I/O)
 *
 * Android CMakeLists.txt excerpt:
 *   add_library(instantid SHARED
 *       instantid_pipeline.cpp
 *       face_processor.cpp)
 *   target_include_directories(instantid PRIVATE
 *       ${CMAKE_SOURCE_DIR}/../../third_party/OnnxStream/src)
 *   target_link_libraries(instantid onnxstream log)
 */

#include "instantid_pipeline.h"
#include "face_processor.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>

// ── OnnxStream include ────────────────────────────────────────────────────────
// Prefer OnnxStream for low RAM usage on mobile.
// Fallback to ONNX Runtime if OnnxStream not available.
#ifdef USE_ONNXSTREAM
    #include "onnxstream.h"  // OnnxStream header
    using InferenceBackend = OnnxStream;
#else
    #include <onnxruntime_cxx_api.h>
    using InferenceBackend = Ort::Session;
#endif

namespace instantid {

// ─── Timing helper ────────────────────────────────────────────────────────────
using Clock    = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

static float elapsed_ms(TimePoint start) {
    auto now = Clock::now();
    return std::chrono::duration<float, std::milli>(now - start).count();
}

// ─── Half-precision utilities ─────────────────────────────────────────────────
// Mobile models use FP16. We need basic FP16 ↔ FP32 conversion.

static uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint16_t h = (x >> 16) & 0x8000;  // sign
    int32_t  e = ((x >> 23) & 0xff) - 127 + 15;
    if (e >= 31)  return h | 0x7c00;   // inf/nan
    if (e <= 0)   return h;             // underflow → 0
    h |= (uint16_t)(e << 10);
    h |= (uint16_t)((x >> 13) & 0x3ff);
    return h;
}

static float half_to_float(uint16_t h) {
    uint32_t x = (h & 0x8000u) << 16;
    int32_t  e = (h >> 10) & 0x1f;
    uint32_t m = h & 0x3ff;
    if (e == 0) {
        if (m == 0) { float f; std::memcpy(&f, &x, 4); return f; }
        // subnormal
        while (!(m & 0x400)) { m <<= 1; e--; }
        e++; m &= 0x3ff;
    } else if (e == 31) {
        x |= 0x7f800000 | (m << 13);
        float f; std::memcpy(&f, &x, 4); return f;
    }
    x |= ((e + 112) << 23) | (m << 13);
    float f; std::memcpy(&f, &x, 4); return f;
}

// ─── LCM Scheduler ────────────────────────────────────────────────────────────

class LCMScheduler {
public:
    explicit LCMScheduler(int num_steps = 4,
                          int num_train_timesteps = 1000) {
        num_steps_ = num_steps;
        T_ = num_train_timesteps;

        // Precompute alpha_cumprod using cosine schedule (SDXL default)
        alphas_cumprod_.resize(T_);
        for (int t = 0; t < T_; t++) {
            float s = 0.008f;
            float alpha = std::cos((float(t) / T_ + s) / (1.f + s) * M_PI / 2.f);
            alphas_cumprod_[t] = alpha * alpha;
        }
        // Normalize
        float alpha0 = alphas_cumprod_[0];
        for (auto& a : alphas_cumprod_) a /= alpha0;

        // LCM timesteps: evenly spaced in [step_ratio-1, T-1]
        int step_ratio = T_ / num_steps;
        timesteps_.resize(num_steps);
        for (int i = 0; i < num_steps; i++) {
            // Reversed: start from high noise
            timesteps_[i] = (num_steps - 1 - i) * step_ratio + step_ratio - 1;
        }
    }

    const std::vector<int>& timesteps() const { return timesteps_; }

    /**
     * LCM Euler step.
     *
     * Given:
     *   noise_pred: UNet prediction of noise
     *   t:          current timestep
     *   latents:    current noisy latents
     *
     * Computes predicted x_0, then steps to x_{t-1}.
     *
     * @param latents  In-place update [B, 4, H, W] as float16 array
     * @param noise_pred [B, 4, H, W] float16
     * @param t_idx    Index into timesteps_ (0-based)
     * @param n        Number of elements (B * 4 * H * W)
     */
    void step(uint16_t* latents, const uint16_t* noise_pred,
              int t_idx, int n) const {
        int t = timesteps_[t_idx];
        float alpha_t = alphas_cumprod_[t];
        float sigma_t = std::sqrt(1.f - alpha_t);
        float sqrt_alpha_t = std::sqrt(alpha_t);

        // Get prev timestep
        float alpha_prev;
        if (t_idx + 1 < (int)timesteps_.size()) {
            int t_prev = timesteps_[t_idx + 1];
            alpha_prev = alphas_cumprod_[t_prev];
        } else {
            alpha_prev = 1.f;  // Final step → pure x_0
        }
        float sigma_prev    = std::sqrt(1.f - alpha_prev);
        float sqrt_alpha_prev = std::sqrt(alpha_prev);

        for (int i = 0; i < n; i++) {
            float z = half_to_float(latents[i]);
            float e = half_to_float(noise_pred[i]);

            // Predict x_0: z = alpha_t * x0 + sigma_t * noise
            float x0 = (z - sigma_t * e) / (sqrt_alpha_t + 1e-8f);
            // Clamp for stability
            x0 = std::max(-1.f, std::min(1.f, x0));

            // LCM deterministic step (no stochastic noise):
            // x_{t-1} = sqrt(alpha_prev) * x0 + sigma_prev * noise_pred
            float z_prev = sqrt_alpha_prev * x0 + sigma_prev * e;

            latents[i] = float_to_half(z_prev);
        }
    }

private:
    int num_steps_, T_;
    std::vector<float> alphas_cumprod_;
    std::vector<int>   timesteps_;
};

// ─── Tokenizer (BPE) ──────────────────────────────────────────────────────────

/**
 * Minimal BPE tokenizer for CLIP.
 * Loads vocab.json + merges.txt in the same format as HuggingFace CLIPTokenizer.
 */
class CLIPTokenizer {
public:
    CLIPTokenizer(const std::string& vocab_path,
                  const std::string& merges_path) {
        load_vocab(vocab_path);
        load_merges(merges_path);
    }

    /**
     * Tokenize text to token IDs.
     * Pads/truncates to max_length=77 (CLIP standard).
     *
     * @return vector of token IDs, length = max_length
     */
    std::vector<int32_t> encode(const std::string& text,
                                int max_length = 77) {
        // Simple whitespace + BPE tokenization
        std::vector<int32_t> tokens;
        tokens.push_back(bos_token_id_);  // <|startoftext|>

        // Lowercase + basic cleanup
        std::string clean = to_lower(text);

        // Split into words
        auto words = split_words(clean);
        for (const auto& word : words) {
            auto word_tokens = bpe_encode(word + "</w>");
            for (auto t : word_tokens) {
                tokens.push_back(t);
                if ((int)tokens.size() >= max_length - 1) break;
            }
            if ((int)tokens.size() >= max_length - 1) break;
        }

        tokens.push_back(eos_token_id_);  // <|endoftext|>

        // Pad to max_length
        while ((int)tokens.size() < max_length) {
            tokens.push_back(pad_token_id_);
        }
        tokens.resize(max_length);
        return tokens;
    }

private:
    std::unordered_map<std::string, int32_t> vocab_;
    std::vector<std::pair<std::string, std::string>> merges_;
    int32_t bos_token_id_ = 49406;
    int32_t eos_token_id_ = 49407;
    int32_t pad_token_id_ = 0;

    void load_vocab(const std::string& path) {
        // Parse JSON vocab file: {"token": id, ...}
        std::ifstream f(path);
        if (!f) throw std::runtime_error("Cannot open vocab: " + path);
        // Minimal JSON parser for flat string→int map
        std::string line;
        while (std::getline(f, line)) {
            size_t colon = line.rfind(':');
            size_t q2    = line.rfind('"', colon - 1);
            size_t q1    = line.rfind('"', q2 - 1);
            if (colon == std::string::npos || q1 == std::string::npos) continue;
            std::string token = line.substr(q1 + 1, q2 - q1 - 1);
            int32_t id = std::stoi(line.substr(colon + 1));
            vocab_[token] = id;
        }
    }

    void load_merges(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("Cannot open merges: " + path);
        std::string line;
        std::getline(f, line);  // skip header "#version: ..."
        while (std::getline(f, line)) {
            size_t space = line.find(' ');
            if (space == std::string::npos) continue;
            merges_.push_back({line.substr(0, space), line.substr(space + 1)});
        }
    }

    std::string to_lower(const std::string& s) {
        std::string r = s;
        for (auto& c : r) c = std::tolower(c);
        return r;
    }

    std::vector<std::string> split_words(const std::string& text) {
        std::vector<std::string> words;
        std::istringstream iss(text);
        std::string word;
        while (iss >> word) words.push_back(word);
        return words;
    }

    std::vector<int32_t> bpe_encode(const std::string& word) {
        // Initialize: each character is a symbol
        std::vector<std::string> symbols;
        for (char c : word) symbols.push_back(std::string(1, c));

        // Apply BPE merges
        bool changed = true;
        while (changed && symbols.size() > 1) {
            changed = false;
            // Find highest-priority merge
            int best_rank = INT_MAX, best_idx = -1;
            for (int i = 0; i + 1 < (int)symbols.size(); i++) {
                for (int r = 0; r < (int)merges_.size(); r++) {
                    if (merges_[r].first == symbols[i] &&
                        merges_[r].second == symbols[i+1] && r < best_rank) {
                        best_rank = r;
                        best_idx  = i;
                    }
                }
            }
            if (best_idx < 0) break;
            // Apply merge
            std::string merged = symbols[best_idx] + symbols[best_idx+1];
            symbols.erase(symbols.begin() + best_idx + 1);
            symbols[best_idx] = merged;
            changed = true;
        }

        // Convert to IDs
        std::vector<int32_t> ids;
        for (const auto& sym : symbols) {
            auto it = vocab_.find(sym);
            ids.push_back(it != vocab_.end() ? it->second : 0);
        }
        return ids;
    }
};

// ─── Impl ─────────────────────────────────────────────────────────────────────

struct InstantIDPipeline::Impl {
    PipelineConfig config;
    std::atomic<bool> cancelled{false};

    // Lazily initialized models
    std::unique_ptr<FaceProcessor>  face_proc;
    std::unique_ptr<CLIPTokenizer>  tokenizer1;
    std::unique_ptr<CLIPTokenizer>  tokenizer2;

    // Construct full model path
    std::string model_path(const std::string& rel) const {
        return config.models_dir + "/" + rel;
    }

    // Report progress
    void report(ProgressCallback& cb, ProgressInfo::Stage stage,
                int step, int total, float pct, const std::string& msg) {
        if (!cb) return;
        ProgressInfo info;
        info.stage       = stage;
        info.step        = step;
        info.total_steps = total;
        info.progress    = pct;
        info.message     = msg;
        cb(info);
    }
};

// ─── Constructor / Destructor ─────────────────────────────────────────────────

InstantIDPipeline::InstantIDPipeline(const PipelineConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;
}

InstantIDPipeline::~InstantIDPipeline() = default;

void InstantIDPipeline::cancel() {
    impl_->cancelled.store(true);
}

bool InstantIDPipeline::models_available() const {
    // Check for key model files
    auto check = [&](const std::string& rel) {
        std::ifstream f(impl_->model_path(rel));
        return f.good();
    };
    return check(impl_->config.scrfd_model)  &&
           check(impl_->config.glintr_model) &&
           check(impl_->config.unet_dir + "/model.yaml") &&
           check(impl_->config.controlnet_dir + "/model.yaml");
}

// ─── Main generate() ──────────────────────────────────────────────────────────

GenerationResult InstantIDPipeline::generate(
    const uint8_t* face_image_rgb,
    int face_img_h, int face_img_w,
    const std::string& prompt,
    const std::string& neg_prompt,
    ProgressCallback progress_cb)
{
    impl_->cancelled.store(false);

    GenerationResult result;
    auto t_total_start = Clock::now();

    const auto& cfg = impl_->config;
    const int H     = cfg.image_height;
    const int W     = cfg.image_width;
    const int latH  = H / 8;
    const int latW  = W / 8;
    const int latN  = 1 * 4 * latH * latW;  // batch=1, channels=4

    try {

    // ══════════════════════════════════════════════════════════════════════
    // STAGE 1: Face Detection + Embedding
    // RAM peak: ~70MB
    // ══════════════════════════════════════════════════════════════════════
    {
        impl_->report(progress_cb, ProgressInfo::Stage::FaceDetection,
                      0, 1, 0.05f, "Detecting face...");
        auto t0 = Clock::now();

        // Load InsightFace models
        impl_->face_proc = std::make_unique<FaceProcessor>(
            impl_->model_path(cfg.scrfd_model),
            impl_->model_path(cfg.glintr_model),
            640,
            cfg.use_nnapi
        );

        FaceInfo face;
        if (!impl_->face_proc->process(face_image_rgb, face_img_h, face_img_w, face)) {
            result.error_message = "No face detected in input image";
            return result;
        }

        // face.embedding: [512] float32
        // face.kps: [5] keypoints

        // Draw keypoints image for ControlNet: [1, 3, H, W] float32
        auto kps_image = FaceProcessor::draw_kps(face.kps, W, H, face_img_w, face_img_h);

        // ── UNLOAD face models — free 70MB ────────────────────────────────
        impl_->face_proc.reset();

        result.timings.face_detection = elapsed_ms(t0);
    }

    if (impl_->cancelled) { result.error_message = "Cancelled"; return result; }

    // ══════════════════════════════════════════════════════════════════════
    // STAGE 2: Text Encoding
    // Load enc1 → encode → unload → load enc2 → encode → unload
    // RAM peak: ~225MB (enc2)
    // ══════════════════════════════════════════════════════════════════════
    {
        impl_->report(progress_cb, ProgressInfo::Stage::TextEncoding,
                      0, 1, 0.15f, "Encoding text...");
        auto t0 = Clock::now();

        // Lazy init tokenizers (tiny, ~5MB each)
        if (!impl_->tokenizer1) {
            impl_->tokenizer1 = std::make_unique<CLIPTokenizer>(
                impl_->model_path(cfg.tokenizer1_vocab),
                impl_->model_path(cfg.tokenizer1_merges)
            );
        }
        if (!impl_->tokenizer2) {
            impl_->tokenizer2 = std::make_unique<CLIPTokenizer>(
                impl_->model_path(cfg.tokenizer2_vocab),
                impl_->model_path(cfg.tokenizer2_merges)
            );
        }

        // Tokenize
        auto ids1 = impl_->tokenizer1->encode(prompt, 77);
        auto ids2 = impl_->tokenizer2->encode(prompt, 77);
        auto neg_ids1 = impl_->tokenizer1->encode(neg_prompt, 77);
        auto neg_ids2 = impl_->tokenizer2->encode(neg_prompt, 77);

        // Text Encoder 1 outputs: [1, 77, 768]
        // Text Encoder 2 outputs: [1, 77, 1280] + pooled [1, 1280]
        // We allocate these on heap to avoid 2MB+ stack allocation
        const int SEQ = 77;
        std::vector<float> hidden1(1 * SEQ * 768);
        std::vector<float> hidden2(1 * SEQ * 1280);
        std::vector<float> pooled(1 * 1280);
        std::vector<float> neg_hidden1(1 * SEQ * 768);
        std::vector<float> neg_hidden2(1 * SEQ * 1280);
        std::vector<float> neg_pooled(1 * 1280);

        // ── Run text_encoder_1 ────────────────────────────────────────────
        // [pseudo-code — replace with actual OnnxStream/ORT calls]
        {
            // auto sess = load_onnxstream(impl_->model_path(cfg.text_enc1_dir));
            // sess.set_input("input_ids", ids1.data(), {1, SEQ});
            // sess.run();
            // sess.get_output("last_hidden_state", hidden1.data());
            // → unload automatically (OnnxStream streams weights)
        }

        // ── Run text_encoder_2 ────────────────────────────────────────────
        {
            // auto sess = load_onnxstream(impl_->model_path(cfg.text_enc2_dir));
            // sess.set_input("input_ids", ids2.data(), {1, SEQ});
            // sess.run();
            // sess.get_output("hidden_states", hidden2.data());
            // sess.get_output("pooled_output", pooled.data());
        }

        // Concat hidden1 [1,77,768] + hidden2 [1,77,1280] → [1,77,2048]
        // text_embeds: [1, 77, 2048]
        // (stored for diffusion loop)

        result.timings.text_encoding = elapsed_ms(t0);
    }

    if (impl_->cancelled) { result.error_message = "Cancelled"; return result; }

    // ══════════════════════════════════════════════════════════════════════
    // STAGE 3: IP-Adapter Projection
    // face_embedding [1,512] → image_prompt_embeds [1,16,2048]
    // RAM: ~25MB
    // ══════════════════════════════════════════════════════════════════════
    {
        impl_->report(progress_cb, ProgressInfo::Stage::IPAdapter,
                      0, 1, 0.25f, "Processing face embedding...");
        auto t0 = Clock::now();

        // image_prompt_embeds: [1, 16, 2048]
        std::vector<float> image_prompt_embeds(1 * 16 * 2048);
        {
            // auto sess = load_onnxstream(impl_->model_path(cfg.ip_adapter_dir));
            // sess.set_input("face_embedding", face.embedding.data(), {1, 512});
            // sess.run();
            // sess.get_output("image_prompt_embeds", image_prompt_embeds.data());
        }

        // Append image_prompt_embeds to text_embeds:
        // encoder_hidden_states = [1, 77+16, 2048] = [1, 93, 2048]
        // (ip_scale controls face preservation)

        result.timings.ip_adapter = elapsed_ms(t0);
    }

    if (impl_->cancelled) { result.error_message = "Cancelled"; return result; }

    // ══════════════════════════════════════════════════════════════════════
    // STAGE 4: Diffusion Loop
    // UNet + ControlNet × num_steps
    // RAM peak: ~2GB (streaming with OnnxStream)
    // ══════════════════════════════════════════════════════════════════════
    {
        auto t0 = Clock::now();
        LCMScheduler scheduler(cfg.num_steps);

        // Initialize random latents: [1, 4, latH, latW]
        std::vector<uint16_t> latents(latN);
        {
            std::mt19937 rng(cfg.seed);
            std::normal_distribution<float> normal(0.f, 1.f);
            for (auto& v : latents) v = float_to_half(normal(rng));
        }

        // Allocate buffers for UNet/ControlNet outputs
        std::vector<uint16_t> noise_pred(latN);
        // ControlNet residuals (9 down + 1 mid)
        // Sizes determined by UNet architecture — allocate based on latH/latW
        struct ResidualBuffer {
            std::vector<uint16_t> data;
            std::vector<int64_t>  shape;
        };
        // SDXL ControlNet residual shapes for latent 64×64:
        const std::vector<std::tuple<int,int,int,int>> down_shapes = {
            {1, 320, latH,   latW  },
            {1, 320, latH,   latW  },
            {1, 320, latH,   latW  },
            {1, 320, latH/2, latW/2},
            {1, 640, latH/2, latW/2},
            {1, 640, latH/2, latW/2},
            {1, 640, latH/4, latW/4},
            {1,1280, latH/4, latW/4},
            {1,1280, latH/4, latW/4},
        };
        const auto mid_shape = std::make_tuple(1, 1280, latH/4, latW/4);

        std::vector<std::vector<uint16_t>> down_residuals(9);
        for (int i = 0; i < 9; i++) {
            auto [b,c,h,w] = down_shapes[i];
            down_residuals[i].resize(b * c * h * w, 0);
        }
        auto [mb,mc,mh,mw] = mid_shape;
        std::vector<uint16_t> mid_residual(mb * mc * mh * mw, 0);

        // Diffusion steps
        const auto& timesteps = scheduler.timesteps();
        for (int step_idx = 0; step_idx < (int)timesteps.size(); step_idx++) {
            if (impl_->cancelled) { result.error_message = "Cancelled"; return result; }

            int t = timesteps[step_idx];
            float pct = 0.30f + 0.60f * float(step_idx) / timesteps.size();
            std::string msg = "Generating... step " +
                              std::to_string(step_idx + 1) + "/" +
                              std::to_string(timesteps.size());

            impl_->report(progress_cb, ProgressInfo::Stage::Diffusion,
                          step_idx + 1, (int)timesteps.size(), pct, msg);

            // ── ControlNet forward ────────────────────────────────────────
            // Inputs:
            //   sample:                latents [1, 4, latH, latW]
            //   timestep:              t
            //   encoder_hidden_states: [1, 77, 2048]
            //   controlnet_cond:       kps_image [1, 3, H, W]
            //   conditioning_scale:    cfg.controlnet_scale
            // Outputs: down_residuals + mid_residual
            {
                // auto cn = load_onnxstream(impl_->model_path(cfg.controlnet_dir));
                // cn.set_input("sample", latents.data(), {1,4,latH,latW});
                // cn.set_input("timestep", &t, {1});
                // ... set encoder_hidden_states, controlnet_cond
                // cn.run();
                // cn.get_output("down_0", down_residuals[0].data());
                // ... etc
                // cn.get_output("mid", mid_residual.data());
                // → OnnxStream auto-streams + frees weight chunks
            }

            // ── UNet forward ──────────────────────────────────────────────
            // Inputs:
            //   sample, timestep, encoder_hidden_states
            //   text_embeds (pooled), time_ids
            //   down_residual_0..8, mid_residual
            // Output: noise_pred [1, 4, latH, latW]
            {
                // auto unet = load_onnxstream(impl_->model_path(cfg.unet_dir));
                // unet.set_input("sample", latents.data(), ...);
                // ... set all inputs
                // unet.run();
                // unet.get_output("noise_pred", noise_pred.data());
            }

            // ── Scheduler step ────────────────────────────────────────────
            scheduler.step(latents.data(), noise_pred.data(), step_idx, latN);
        }

        result.timings.diffusion = elapsed_ms(t0);
    }

    if (impl_->cancelled) { result.error_message = "Cancelled"; return result; }

    // ══════════════════════════════════════════════════════════════════════
    // STAGE 5: VAE Decode
    // latents [1,4,latH,latW] → image [1,3,H,W]
    // RAM: ~200MB
    // ══════════════════════════════════════════════════════════════════════
    {
        impl_->report(progress_cb, ProgressInfo::Stage::VAEDecode,
                      0, 1, 0.92f, "Decoding image...");
        auto t0 = Clock::now();

        const int imgN = 1 * 3 * H * W;
        std::vector<float> image_float(imgN);

        {
            // auto vae = load_onnxstream(impl_->model_path(cfg.vae_dir));
            // vae.set_input("latent", latents.data(), {1, 4, latH, latW});
            // vae.run();
            // vae.get_output("image", image_float.data());
        }

        // Convert float [0,1] CHW → uint8 HWC
        result.image_rgb.resize(H * W * 3);
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    float val = image_float[c * H * W + h * W + w];
                    val = std::max(0.f, std::min(1.f, val));
                    result.image_rgb[(h * W + w) * 3 + c] =
                        static_cast<uint8_t>(val * 255.f + 0.5f);
                }
            }
        }
        result.image_height = H;
        result.image_width  = W;

        result.timings.vae_decode = elapsed_ms(t0);
    }

    // Done
    result.timings.total = elapsed_ms(t_total_start);
    result.success = true;

    impl_->report(progress_cb, ProgressInfo::Stage::Done,
                  0, 1, 1.0f, "Done!");

    } catch (const std::exception& e) {
        result.error_message = e.what();
        result.success = false;
    }

    return result;
}

}  // namespace instantid
