/**
 * face_processor.cpp
 * ===================
 * InsightFace SCRFD + GlintR100 on-device inference.
 */

#include "face_processor.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

// ── ONNX Runtime ─────────────────────────────────────────────────────────────
#include <onnxruntime_cxx_api.h>

namespace instantid {

// ─── ArcFace canonical keypoints (112×112) ────────────────────────────────────
// Standard alignment target for ArcFace/GlintR100
static const std::array<Keypoint, 5> ARCFACE_DST = {{
    {38.2946f, 51.6963f},  // left eye
    {73.5318f, 51.5014f},  // right eye
    {56.0252f, 71.7366f},  // nose tip
    {41.5493f, 92.3655f},  // left mouth corner
    {70.7299f, 92.2041f},  // right mouth corner
}};

// ─── SCRFD anchor config ───────────────────────────────────────────────────────
// SCRFD-10G-BNKPS: 3 strides, 2 anchors per location
static const int STRIDES[]  = {8, 16, 32};
static const int NUM_ANCHORS = 2;

// ─── Impl struct ──────────────────────────────────────────────────────────────

struct FaceProcessor::Impl {
    int det_size;

    // ORT environment (shared)
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "FaceProcessor"};
    Ort::SessionOptions session_opts;

    std::unique_ptr<Ort::Session> det_session;
    std::unique_ptr<Ort::Session> emb_session;

    Ort::AllocatorWithDefaultOptions allocator;
};

// ─── Constructor ──────────────────────────────────────────────────────────────

FaceProcessor::FaceProcessor(
    const std::string& det_model_path,
    const std::string& emb_model_path,
    int det_size,
    bool use_nnapi)
    : impl_(std::make_unique<Impl>())
{
    impl_->det_size = det_size;

    // Configure session options
    impl_->session_opts.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    impl_->session_opts.SetIntraOpNumThreads(2);  // 2 threads on mobile

    // Android NNAPI acceleration
    if (use_nnapi) {
#ifdef __ANDROID__
        OrtNnapiProviderOptions nnapi_opts{};
        nnapi_opts.flags = NNAPI_FLAG_USE_FP16;  // FP16 on NPU
        impl_->session_opts.AppendExecutionProvider_Nnapi(nnapi_opts);
#endif
    }

    // Load detection model
    impl_->det_session = std::make_unique<Ort::Session>(
        impl_->env,
        det_model_path.c_str(),
        impl_->session_opts
    );

    // Load embedding model
    impl_->emb_session = std::make_unique<Ort::Session>(
        impl_->env,
        emb_model_path.c_str(),
        impl_->session_opts
    );
}

FaceProcessor::~FaceProcessor() = default;

// ─── detect_faces() ───────────────────────────────────────────────────────────

std::vector<FaceInfo> FaceProcessor::detect_faces(
    const uint8_t* image_rgb,
    int img_h, int img_w,
    float min_score)
{
    const int det_size = impl_->det_size;

    // Preprocess
    std::vector<float> input_tensor;
    float scale;
    int pad_left, pad_top;
    preprocess_detection(image_rgb, img_h, img_w, det_size,
                         input_tensor, scale, pad_left, pad_top);

    // Run SCRFD
    std::array<int64_t, 4> input_shape = {1, 3, det_size, det_size};
    Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_val = Ort::Value::CreateTensor<float>(
        mem_info,
        input_tensor.data(),
        input_tensor.size(),
        input_shape.data(),
        input_shape.size()
    );

    const char* input_names[]  = {"input.1"};
    auto output_names_owned = impl_->det_session->GetOutputNames();
    std::vector<const char*> output_names;
    for (auto& n : output_names_owned)
        output_names.push_back(n.get());

    auto outputs = impl_->det_session->Run(
        Ort::RunOptions{nullptr},
        input_names, &input_val, 1,
        output_names.data(), output_names.size()
    );

    // Collect raw output pointers and shapes
    std::vector<float*> out_ptrs;
    std::vector<std::vector<int64_t>> out_shapes;
    for (auto& out : outputs) {
        auto shape = out.GetTensorTypeAndShapeInfo().GetShape();
        out_ptrs.push_back(out.GetTensorMutableData<float>());
        out_shapes.push_back(shape);
    }

    return decode_scrfd_output(out_ptrs, out_shapes,
                               det_size, img_w, img_h,
                               scale, pad_left, pad_top,
                               min_score, 0.4f);
}

// ─── Preprocessing ────────────────────────────────────────────────────────────

void FaceProcessor::preprocess_detection(
    const uint8_t* image_rgb, int img_h, int img_w,
    int det_size,
    std::vector<float>& out, float& scale, int& pad_left, int& pad_top)
{
    // Letterbox resize: maintain aspect ratio, pad with zeros
    float scale_w = float(det_size) / img_w;
    float scale_h = float(det_size) / img_h;
    scale = std::min(scale_w, scale_h);

    int new_w = int(img_w * scale);
    int new_h = int(img_h * scale);
    pad_left  = (det_size - new_w) / 2;
    pad_top   = (det_size - new_h) / 2;

    // Allocate padded CHW tensor [1, 3, det_size, det_size]
    out.assign(3 * det_size * det_size, 0.f);

    // Bilinear resize + normalize + BGR + CHW
    const float mean = 127.5f;
    const float std  = 128.0f;

    for (int dst_y = 0; dst_y < new_h; dst_y++) {
        float src_y = (dst_y + 0.5f) / scale - 0.5f;
        int y0 = std::max(0, std::min(img_h - 1, int(src_y)));
        int y1 = std::max(0, std::min(img_h - 1, y0 + 1));
        float wy = src_y - y0;

        for (int dst_x = 0; dst_x < new_w; dst_x++) {
            float src_x = (dst_x + 0.5f) / scale - 0.5f;
            int x0 = std::max(0, std::min(img_w - 1, int(src_x)));
            int x1 = std::max(0, std::min(img_w - 1, x0 + 1));
            float wx = src_x - x0;

            int actual_y = dst_y + pad_top;
            int actual_x = dst_x + pad_left;

            // Bilinear interp for each channel
            // SCRFD expects BGR input
            for (int c = 0; c < 3; c++) {
                int bgr_c = 2 - c;  // RGB→BGR: R→B, G→G, B→R
                float v00 = image_rgb[(y0 * img_w + x0) * 3 + bgr_c];
                float v01 = image_rgb[(y0 * img_w + x1) * 3 + bgr_c];
                float v10 = image_rgb[(y1 * img_w + x0) * 3 + bgr_c];
                float v11 = image_rgb[(y1 * img_w + x1) * 3 + bgr_c];

                float v = v00 * (1-wx) * (1-wy)
                        + v01 * wx     * (1-wy)
                        + v10 * (1-wx) * wy
                        + v11 * wx     * wy;

                float norm = (v - mean) / std;
                out[c * det_size * det_size + actual_y * det_size + actual_x] = norm;
            }
        }
    }
}

// ─── Decode SCRFD output ──────────────────────────────────────────────────────

std::vector<FaceInfo> FaceProcessor::decode_scrfd_output(
    const std::vector<float*>& outputs,
    const std::vector<std::vector<int64_t>>& output_shapes,
    int det_size, int orig_w, int orig_h,
    float scale, int pad_left, int pad_top,
    float score_threshold, float nms_threshold)
{
    std::vector<FaceInfo> faces;
    std::vector<BoundingBox> all_boxes;
    std::vector<std::array<Keypoint, 5>> all_kps;

    // SCRFD outputs per stride: [cls, bbox, kps]
    // Typical SCRFD-10G output layout (varies — check model):
    //   stride 8:  cls [1, H8*W8*2, 1], bbox [1, H8*W8*2, 4], kps [1, H8*W8*2, 10]
    //   stride 16: cls [1, H16*W16*2, 1], ...
    //   stride 32: cls [1, H32*W32*2, 1], ...

    int out_idx = 0;
    for (int stride_idx = 0; stride_idx < 3; stride_idx++) {
        int stride = STRIDES[stride_idx];
        int feat_h = det_size / stride;
        int feat_w = det_size / stride;
        int num_loc = feat_h * feat_w * NUM_ANCHORS;

        if (out_idx + 3 > (int)outputs.size()) break;

        float* cls_data  = outputs[out_idx];   // [num_loc, 1]
        float* bbox_data = outputs[out_idx+1]; // [num_loc, 4] ltrb in stride units
        float* kps_data  = outputs[out_idx+2]; // [num_loc, 10]
        out_idx += 3;

        for (int i = 0; i < num_loc; i++) {
            float score = cls_data[i];
            // Apply sigmoid if model didn't apply it
            score = 1.f / (1.f + std::exp(-score));
            if (score < score_threshold) continue;

            // Anchor center
            int anchor_i = i / NUM_ANCHORS;
            int ax = (anchor_i % feat_w) * stride + stride / 2;
            int ay = (anchor_i / feat_w) * stride + stride / 2;

            // Decode bbox (ltrb in stride units → pixel coords)
            float l = bbox_data[i * 4 + 0] * stride;
            float t = bbox_data[i * 4 + 1] * stride;
            float r = bbox_data[i * 4 + 2] * stride;
            float b = bbox_data[i * 4 + 3] * stride;

            float x1 = ax - l, y1 = ay - t;
            float x2 = ax + r, y2 = ay + b;

            // Un-letterbox: remove padding, un-scale
            x1 = (x1 - pad_left) / scale;
            y1 = (y1 - pad_top) / scale;
            x2 = (x2 - pad_left) / scale;
            y2 = (y2 - pad_top) / scale;

            // Clamp to image bounds
            x1 = std::max(0.f, std::min(float(orig_w - 1), x1));
            y1 = std::max(0.f, std::min(float(orig_h - 1), y1));
            x2 = std::max(0.f, std::min(float(orig_w - 1), x2));
            y2 = std::max(0.f, std::min(float(orig_h - 1), y2));

            BoundingBox box{x1, y1, x2, y2, score};
            all_boxes.push_back(box);

            // Decode keypoints
            std::array<Keypoint, 5> kps;
            for (int k = 0; k < 5; k++) {
                float kx = kps_data[i * 10 + k * 2 + 0] * stride + ax;
                float ky = kps_data[i * 10 + k * 2 + 1] * stride + ay;
                kx = (kx - pad_left) / scale;
                ky = (ky - pad_top)  / scale;
                kps[k] = {
                    std::max(0.f, std::min(float(orig_w - 1), kx)),
                    std::max(0.f, std::min(float(orig_h - 1), ky)),
                };
            }
            all_kps.push_back(kps);
        }
    }

    // NMS
    auto keep = nms(all_boxes, nms_threshold);

    for (int idx : keep) {
        FaceInfo face;
        face.bbox     = all_boxes[idx];
        face.kps      = all_kps[idx];
        face.det_score = all_boxes[idx].score;
        faces.push_back(face);
    }

    // Sort by area (largest first)
    std::sort(faces.begin(), faces.end(), [](const FaceInfo& a, const FaceInfo& b) {
        return a.bbox.area() > b.bbox.area();
    });

    return faces;
}

// ─── NMS ──────────────────────────────────────────────────────────────────────

std::vector<int> FaceProcessor::nms(
    const std::vector<BoundingBox>& boxes, float threshold)
{
    // Sort by score descending
    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b){ return boxes[a].score > boxes[b].score; });

    std::vector<int> keep;
    std::vector<bool> suppressed(boxes.size(), false);

    for (int i = 0; i < (int)order.size(); i++) {
        int idx = order[i];
        if (suppressed[idx]) continue;
        keep.push_back(idx);

        const auto& a = boxes[idx];
        for (int j = i + 1; j < (int)order.size(); j++) {
            int jdx = order[j];
            if (suppressed[jdx]) continue;
            const auto& b = boxes[jdx];

            // IoU
            float ix1 = std::max(a.x1, b.x1);
            float iy1 = std::max(a.y1, b.y1);
            float ix2 = std::min(a.x2, b.x2);
            float iy2 = std::min(a.y2, b.y2);

            float iw = std::max(0.f, ix2 - ix1);
            float ih = std::max(0.f, iy2 - iy1);
            float inter = iw * ih;
            float iou = inter / (a.area() + b.area() - inter + 1e-6f);

            if (iou > threshold) suppressed[jdx] = true;
        }
    }
    return keep;
}

// ─── extract_embedding() ──────────────────────────────────────────────────────

void FaceProcessor::extract_embedding(
    const uint8_t* image_rgb, int img_h, int img_w,
    FaceInfo& face)
{
    // 1. ArcFace alignment → 112×112 crop
    std::vector<uint8_t> aligned(112 * 112 * 3);
    align_face(image_rgb, img_h, img_w, face.kps,
               aligned.data(), 112);

    // 2. Preprocess for GlintR100
    std::vector<float> input_tensor;
    preprocess_embedding(aligned.data(), input_tensor);

    // 3. Run GlintR100
    std::array<int64_t, 4> shape = {1, 3, 112, 112};
    Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_val = Ort::Value::CreateTensor<float>(
        mem_info, input_tensor.data(), input_tensor.size(),
        shape.data(), shape.size()
    );

    const char* input_names[] = {"input.1"};
    const char* output_names[] = {"683"};  // GlintR100 output node name

    auto outputs = impl_->emb_session->Run(
        Ort::RunOptions{nullptr},
        input_names, &input_val, 1,
        output_names, 1
    );

    float* emb_data = outputs[0].GetTensorMutableData<float>();

    // 4. L2 normalize
    float norm = 0.f;
    for (int i = 0; i < 512; i++) norm += emb_data[i] * emb_data[i];
    norm = std::sqrt(norm) + 1e-6f;
    for (int i = 0; i < 512; i++) face.embedding[i] = emb_data[i] / norm;
}

// ─── Face alignment ───────────────────────────────────────────────────────────

void FaceProcessor::align_face(
    const uint8_t* image_rgb, int img_h, int img_w,
    const std::array<Keypoint, 5>& kps,
    uint8_t* out_aligned, int target_size)
{
    // Estimate similarity transform from detected kps → ArcFace canonical
    auto M = estimate_similarity_transform(kps, ARCFACE_DST);
    // Scale M to target_size (ARCFACE_DST is for 112×112)
    float scale_factor = float(target_size) / 112.f;
    M[2] *= scale_factor;
    M[5] *= scale_factor;

    apply_affine_transform(image_rgb, img_h, img_w,
                           out_aligned, target_size, target_size, M);
}

// ─── Similarity transform estimation ─────────────────────────────────────────
// Least-squares similarity transform (scale + rotation + translation)
// from 5 point correspondences.

std::array<float, 6> FaceProcessor::estimate_similarity_transform(
    const std::array<Keypoint, 5>& src,
    const std::array<Keypoint, 5>& dst)
{
    // Umeyama algorithm (simplified for 2D similarity)
    const int N = 5;

    // Compute centroids
    float mx = 0, my = 0, tx = 0, ty = 0;
    for (int i = 0; i < N; i++) {
        mx += src[i].x; my += src[i].y;
        tx += dst[i].x; ty += dst[i].y;
    }
    mx /= N; my /= N; tx /= N; ty /= N;

    // Compute covariance and variance
    float sxx = 0, sxy = 0, syx = 0, syy = 0;
    float src_var = 0;
    for (int i = 0; i < N; i++) {
        float sx = src[i].x - mx, sy = src[i].y - my;
        float dx = dst[i].x - tx, dy = dst[i].y - ty;
        sxx += sx * dx; sxy += sx * dy;
        syx += sy * dx; syy += sy * dy;
        src_var += sx * sx + sy * sy;
    }
    src_var /= N;

    // Scale
    float det = sxx * syy - sxy * syx;
    float S = det > 0 ? 1.f : -1.f;

    // SVD approximation for 2×2
    float a = sxx + syy;
    float b = std::sqrt(a * a - 4 * (sxx * syy - sxy * syx) * S * S);
    float scale = (a + b * S) / (2.f * src_var * N);
    if (std::isnan(scale) || scale <= 0) scale = 1.f;

    // Rotation
    float cos_r = (sxx * S + syy) / std::max(1e-8f, (a + b));
    float sin_r = (syx * S - sxy) / std::max(1e-8f, (a + b));

    // Affine matrix: [cos, -sin, tx; sin, cos, ty] * scale
    float r_cos = scale * cos_r;
    float r_sin = scale * sin_r;
    float t_x = tx - r_cos * mx + r_sin * my;
    float t_y = ty - r_sin * mx - r_cos * my;

    return {r_cos, -r_sin, t_x,
            r_sin,  r_cos, t_y};
}

void FaceProcessor::apply_affine_transform(
    const uint8_t* src, int src_h, int src_w,
    uint8_t* dst, int dst_h, int dst_w,
    const std::array<float, 6>& M)
{
    // Invert 2×3 affine matrix for inverse mapping
    float a = M[0], b = M[1], c = M[2];
    float d = M[3], e = M[4], f = M[5];
    float det = a * e - b * d;
    if (std::abs(det) < 1e-8f) {
        std::fill(dst, dst + dst_h * dst_w * 3, 128);
        return;
    }
    float inv_a =  e / det, inv_b = -b / det, inv_c = (b*f - c*e) / det;
    float inv_d = -d / det, inv_e =  a / det, inv_f = (c*d - a*f) / det;

    for (int dy = 0; dy < dst_h; dy++) {
        for (int dx = 0; dx < dst_w; dx++) {
            float sx = inv_a * dx + inv_b * dy + inv_c;
            float sy = inv_d * dx + inv_e * dy + inv_f;

            // Bilinear interpolation
            int x0 = int(sx), y0 = int(sy);
            int x1 = x0 + 1, y1 = y0 + 1;
            float wx = sx - x0, wy = sy - y0;

            for (int c = 0; c < 3; c++) {
                float v = 0;
                if (x0 >= 0 && y0 >= 0 && x0 < src_w && y0 < src_h)
                    v += src[(y0 * src_w + x0) * 3 + c] * (1-wx) * (1-wy);
                if (x1 >= 0 && y0 >= 0 && x1 < src_w && y0 < src_h)
                    v += src[(y0 * src_w + x1) * 3 + c] * wx     * (1-wy);
                if (x0 >= 0 && y1 >= 0 && x0 < src_w && y1 < src_h)
                    v += src[(y1 * src_w + x0) * 3 + c] * (1-wx) * wy;
                if (x1 >= 0 && y1 >= 0 && x1 < src_w && y1 < src_h)
                    v += src[(y1 * src_w + x1) * 3 + c] * wx     * wy;
                dst[(dy * dst_w + dx) * 3 + c] = uint8_t(std::max(0.f, std::min(255.f, v)));
            }
        }
    }
}

// ─── preprocess_embedding() ───────────────────────────────────────────────────

void FaceProcessor::preprocess_embedding(
    const uint8_t* aligned_rgb,
    std::vector<float>& out)
{
    // GlintR100: BGR, (pixel - 127.5) / 128.0, CHW
    out.resize(3 * 112 * 112);
    for (int h = 0; h < 112; h++) {
        for (int w = 0; w < 112; w++) {
            for (int c = 0; c < 3; c++) {
                int bgr_c = 2 - c;  // RGB → BGR
                float v = float(aligned_rgb[(h * 112 + w) * 3 + bgr_c]);
                out[c * 112 * 112 + h * 112 + w] = (v - 127.5f) / 128.0f;
            }
        }
    }
}

// ─── process() ────────────────────────────────────────────────────────────────

bool FaceProcessor::process(
    const uint8_t* image_rgb, int img_h, int img_w,
    FaceInfo& out_face)
{
    auto faces = detect_faces(image_rgb, img_h, img_w, 0.5f);
    if (faces.empty()) return false;

    out_face = faces[0];  // Largest face
    extract_embedding(image_rgb, img_h, img_w, out_face);
    return true;
}

// ─── draw_kps() ───────────────────────────────────────────────────────────────

ImageTensor FaceProcessor::draw_kps(
    const std::array<Keypoint, 5>& kps,
    int out_w, int out_h,
    int src_w, int src_h)
{
    // Colors for 5 keypoints (RGB)
    static const uint8_t colors[5][3] = {
        {255, 0,   0  },  // left eye:    red
        {0,   255, 0  },  // right eye:   green
        {0,   0,   255},  // nose:        blue
        {255, 255, 0  },  // left mouth:  yellow
        {255, 0,   255},  // right mouth: magenta
    };

    // Connectivity: [start_kp, end_kp]
    static const int limbs[4][2] = {{0,2},{1,2},{3,2},{4,2}};

    // Output canvas (HWC uint8)
    std::vector<uint8_t> canvas(out_h * out_w * 3, 0);

    // Scale factor from source → output
    float sx = float(out_w) / src_w;
    float sy = float(out_h) / src_h;

    // Scale keypoints
    std::array<Keypoint, 5> scaled_kps;
    for (int i = 0; i < 5; i++) {
        scaled_kps[i] = {kps[i].x * sx, kps[i].y * sy};
    }

    // Draw limb lines
    auto draw_line = [&](int x0, int y0, int x1, int y1,
                         uint8_t r, uint8_t g, uint8_t b, int thickness = 2) {
        int dx = std::abs(x1 - x0), dy = std::abs(y1 - y0);
        int steps = std::max(dx, dy) + 1;
        for (int s = 0; s < steps; s++) {
            float t = steps > 1 ? float(s) / (steps - 1) : 0;
            int px = int(x0 + t * (x1 - x0));
            int py = int(y0 + t * (y1 - y0));
            for (int ty = -thickness; ty <= thickness; ty++) {
                for (int tx = -thickness; tx <= thickness; tx++) {
                    int cx = px + tx, cy = py + ty;
                    if (cx >= 0 && cx < out_w && cy >= 0 && cy < out_h) {
                        canvas[(cy * out_w + cx) * 3 + 0] = r;
                        canvas[(cy * out_w + cx) * 3 + 1] = g;
                        canvas[(cy * out_w + cx) * 3 + 2] = b;
                    }
                }
            }
        }
    };

    for (auto& limb : limbs) {
        int i = limb[0], j = limb[1];
        draw_line(int(scaled_kps[i].x), int(scaled_kps[i].y),
                  int(scaled_kps[j].x), int(scaled_kps[j].y),
                  255, 255, 255, 2);
    }

    // Draw keypoint circles
    const int radius = 5;
    for (int k = 0; k < 5; k++) {
        int cx = int(scaled_kps[k].x);
        int cy = int(scaled_kps[k].y);
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                if (dx*dx + dy*dy <= radius*radius) {
                    int px = cx + dx, py = cy + dy;
                    if (px >= 0 && px < out_w && py >= 0 && py < out_h) {
                        canvas[(py * out_w + px) * 3 + 0] = colors[k][0];
                        canvas[(py * out_w + px) * 3 + 1] = colors[k][1];
                        canvas[(py * out_w + px) * 3 + 2] = colors[k][2];
                    }
                }
            }
        }
    }

    // Convert HWC uint8 [0,255] → CHW float32 [0,1]
    ImageTensor result;
    result.channels = 3;
    result.height   = out_h;
    result.width    = out_w;
    result.data.resize(3 * out_h * out_w);

    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < out_h; h++) {
            for (int w = 0; w < out_w; w++) {
                result.data[c * out_h * out_w + h * out_w + w] =
                    canvas[(h * out_w + w) * 3 + c] / 255.f;
            }
        }
    }
    return result;
}

}  // namespace instantid
