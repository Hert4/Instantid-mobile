#pragma once
/**
 * face_processor.h
 * =================
 * On-device face detection + embedding extraction using InsightFace ONNX models.
 *
 * Models required (place in models/antelopev2/):
 *   scrfd_10g_bnkps_int8.onnx  — Face detection + keypoints (SCRFD-10G)
 *   glintr100_int8.onnx         — Face embedding (GlintR100, ArcFace, 512-dim)
 *
 * Pipeline:
 *   1. SCRFD detect faces in image → bounding boxes + 5 keypoints
 *   2. Align/crop face using keypoints (ArcFace standard alignment)
 *   3. GlintR100 extract 512-dim L2-normalized embedding
 *   4. draw_kps() → keypoints visualization image for ControlNet conditioning
 *
 * Memory: ~70MB peak (scrfd=4MB + glintr100=65MB INT8)
 * Time:   ~150-300ms on Snapdragon 8 Gen 2
 *
 * Dependencies: onnxruntime-android (or OnnxStream for scrfd/glintr)
 */

#include <array>
#include <memory>
#include <string>
#include <vector>

// Forward declare ORT types to avoid including full onnxruntime headers in .h
namespace Ort {
    struct Env;
    struct Session;
    struct AllocatorWithDefaultOptions;
}

namespace instantid {

// ─── Data structures ──────────────────────────────────────────────────────────

struct BoundingBox {
    float x1, y1, x2, y2;
    float score;

    float width()  const { return x2 - x1; }
    float height() const { return y2 - y1; }
    float area()   const { return width() * height(); }
    float cx()     const { return (x1 + x2) / 2.f; }
    float cy()     const { return (y1 + y2) / 2.f; }
};

struct Keypoint {
    float x, y;
};

struct FaceInfo {
    BoundingBox bbox;
    std::array<Keypoint, 5> kps;   // 5 face keypoints
    std::array<float, 512> embedding; // L2-normalized GlintR100 embedding
    float det_score;
};

// Image data (CHW float32, values in [0,1] unless noted)
struct ImageTensor {
    std::vector<float> data;
    int channels, height, width;

    int size() const { return channels * height * width; }
    float* ptr()       { return data.data(); }
    const float* ptr() const { return data.data(); }
};

// ─── FaceProcessor ────────────────────────────────────────────────────────────

class FaceProcessor {
public:
    /**
     * @param det_model_path   Path to scrfd_10g_bnkps_int8.onnx
     * @param emb_model_path   Path to glintr100_int8.onnx
     * @param det_size         SCRFD input size (default 640x640)
     * @param use_nnapi        Enable Android NNAPI acceleration (Android only)
     */
    FaceProcessor(
        const std::string& det_model_path,
        const std::string& emb_model_path,
        int det_size = 640,
        bool use_nnapi = false
    );

    ~FaceProcessor();

    // Non-copyable (owns ORT sessions)
    FaceProcessor(const FaceProcessor&) = delete;
    FaceProcessor& operator=(const FaceProcessor&) = delete;

    /**
     * Detect all faces in image.
     *
     * @param image  RGB image [H, W, 3] uint8
     * @param min_score  Minimum detection score threshold
     * @return  Vector of detected faces, sorted by area (largest first)
     */
    std::vector<FaceInfo> detect_faces(
        const uint8_t* image_rgb,
        int img_h, int img_w,
        float min_score = 0.5f
    );

    /**
     * Extract face embedding for a single detected face.
     * Aligns face using ArcFace standard alignment before embedding.
     *
     * @param image_rgb  Original image
     * @param face       FaceInfo with bbox + keypoints (from detect_faces)
     * @return  L2-normalized 512-dim embedding (modifies face.embedding in-place)
     */
    void extract_embedding(
        const uint8_t* image_rgb,
        int img_h, int img_w,
        FaceInfo& face
    );

    /**
     * Full pipeline: detect + embed largest face.
     * Returns false if no face detected.
     *
     * @param image_rgb  RGB image
     * @param out_face   Output face info (bbox + kps + embedding)
     */
    bool process(
        const uint8_t* image_rgb,
        int img_h, int img_w,
        FaceInfo& out_face
    );

    /**
     * Draw face keypoints visualization image.
     * Used as ControlNet conditioning input (IdentityNet).
     *
     * Draws colored dots + connecting lines for 5 keypoints:
     *   0: left eye  (red)
     *   1: right eye (green)
     *   2: nose      (blue)
     *   3: left mouth (yellow)
     *   4: right mouth (magenta)
     *
     * @param kps        5 keypoints in original image coordinates
     * @param out_w      Output image width (should match generation resolution)
     * @param out_h      Output image height
     * @param src_w      Source image width (for coordinate scaling)
     * @param src_h      Source image height
     * @return  RGB image [3, out_h, out_w] float32 normalized to [0,1]
     *          (CHW format for direct use as ONNX input)
     */
    static ImageTensor draw_kps(
        const std::array<Keypoint, 5>& kps,
        int out_w, int out_h,
        int src_w, int src_h
    );

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    // ── SCRFD inference ───────────────────────────────────────────────────────
    /**
     * Preprocess image for SCRFD:
     *   1. Resize to det_size x det_size (letterbox padding)
     *   2. BGR conversion
     *   3. Normalize: (pixel - 127.5) / 128.0
     *   4. HWC → CHW, add batch dim
     */
    static void preprocess_detection(
        const uint8_t* image_rgb, int img_h, int img_w,
        int det_size,
        std::vector<float>& out_tensor,  // [1, 3, det_size, det_size]
        float& scale, int& pad_left, int& pad_top
    );

    /**
     * Decode SCRFD output:
     * SCRFD outputs 3 stride levels (8, 16, 32), each with:
     *   - cls_scores: [H*W*2, 1] (2 anchors per location)
     *   - bbox_preds: [H*W*2, 4] (ltrb format)
     *   - kps_preds:  [H*W*2, 10] (5 keypoints × 2 coords)
     *
     * Applies NMS to deduplicate overlapping detections.
     */
    std::vector<FaceInfo> decode_scrfd_output(
        const std::vector<float*>& outputs,
        const std::vector<std::vector<int64_t>>& output_shapes,
        int det_size,
        int orig_w, int orig_h,
        float scale, int pad_left, int pad_top,
        float score_threshold = 0.5f,
        float nms_threshold = 0.4f
    );

    // ── GlintR100 inference ───────────────────────────────────────────────────
    /**
     * ArcFace standard face alignment.
     * Transforms face using similarity transform from detected keypoints
     * to canonical 112x112 face coordinates.
     *
     * Reference keypoints (ArcFace standard):
     *   left_eye:  [38.29, 51.70]
     *   right_eye: [73.53, 51.50]
     *   nose:      [56.02, 71.74]
     *   left_mouth:[41.55, 92.37]
     *   right_mouth:[70.73, 92.20]
     */
    static void align_face(
        const uint8_t* image_rgb, int img_h, int img_w,
        const std::array<Keypoint, 5>& kps,
        uint8_t* out_aligned,  // [112, 112, 3] output
        int target_size = 112
    );

    /**
     * Preprocess aligned face for GlintR100:
     *   BGR, normalize: (pixel - 127.5) / 128.0
     *   HWC → CHW → [1, 3, 112, 112]
     */
    static void preprocess_embedding(
        const uint8_t* aligned_rgb,  // [112, 112, 3]
        std::vector<float>& out      // [1, 3, 112, 112]
    );

    // ── NMS ───────────────────────────────────────────────────────────────────
    static std::vector<int> nms(
        const std::vector<BoundingBox>& boxes,
        float threshold
    );

    // ── 2D similarity transform (for face alignment) ──────────────────────────
    /**
     * Estimate 2D similarity transform (rotation, scale, translation)
     * from source keypoints to destination keypoints.
     * Returns 2x3 affine matrix.
     */
    static std::array<float, 6> estimate_similarity_transform(
        const std::array<Keypoint, 5>& src_pts,
        const std::array<Keypoint, 5>& dst_pts
    );

    static void apply_affine_transform(
        const uint8_t* src, int src_h, int src_w,
        uint8_t* dst, int dst_h, int dst_w,
        const std::array<float, 6>& M  // 2x3 matrix
    );
};

}  // namespace instantid
