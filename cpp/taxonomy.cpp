// perception-gate: IoU matching + detection error classification in C++.
//
// Usage:
//   taxonomy.exe taxonomy <in.bin> <out.bin> [tp_iou=0.5] [loc_iou=0.1]
//   taxonomy.exe match    <in.bin> <out.bin> <iou>          (COCO-style greedy matching, used for AP)
//
// Input  (little endian):
//   int32 n_img, n_gt, n_pred
//   int32 gt_off[n_img+1], pred_off[n_img+1]
//   float32 gt_box[n_gt][4] (x1 y1 x2 y2), int32 gt_cls[n_gt]
//   float32 pred_box[n_pred][4], int32 pred_cls[n_pred]     (predictions sorted by conf DESC per image)
// Output: int32 n_pred, n_gt, then
//   taxonomy: int32 label[n_pred], int32 ref_gt[n_pred], int32 gt_missed[n_gt]
//   match:    int32 matched_gt[n_pred]
// Labels: 0 correct, 1 classification error, 2 localization error, 3 duplicate, 4 background FP.
//
// All IoU arithmetic is done in double from the float32 inputs, in the same operation order as the
// Python reference (src/taxonomy_ref.py), so results must be bit-identical.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static inline double iou_of(const float* p, const float* g) {
    double px1 = p[0], py1 = p[1], px2 = p[2], py2 = p[3];
    double gx1 = g[0], gy1 = g[1], gx2 = g[2], gy2 = g[3];
    double iw = std::min(px2, gx2) - std::max(px1, gx1);
    double ih = std::min(py2, gy2) - std::max(py1, gy1);
    if (iw <= 0 || ih <= 0) return 0.0;
    double inter = iw * ih;
    double pa = (px2 - px1) * (py2 - py1);
    double ga = (gx2 - gx1) * (gy2 - gy1);
    return inter / (pa + ga - inter);
}

template <class T> static void rd(FILE* f, std::vector<T>& v, size_t n) {
    v.resize(n);
    if (n && fread(v.data(), sizeof(T), n, f) != n) { fprintf(stderr, "short read\n"); exit(2); }
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: taxonomy <taxonomy|match> in.bin out.bin [...]\n"); return 1; }
    std::string mode = argv[1];
    FILE* f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "cannot open input\n"); return 1; }
    int32_t hdr[3];
    if (fread(hdr, 4, 3, f) != 3) return 2;
    int n_img = hdr[0], n_gt = hdr[1], n_pred = hdr[2];
    std::vector<int32_t> gt_off, pred_off, gt_cls, pred_cls;
    std::vector<float> gt_box, pred_box;
    rd(f, gt_off, n_img + 1); rd(f, pred_off, n_img + 1);
    rd(f, gt_box, (size_t)n_gt * 4); rd(f, gt_cls, n_gt);
    rd(f, pred_box, (size_t)n_pred * 4); rd(f, pred_cls, n_pred);
    fclose(f);

    std::vector<int32_t> label(n_pred, -1), ref(n_pred, -1), matched(n_pred, -1);
    std::vector<uint8_t> taken(n_gt, 0);
    double tp_iou = 0.5, loc_iou = 0.1;
    if (mode == "taxonomy") {
        if (argc > 4) tp_iou = atof(argv[4]);
        if (argc > 5) loc_iou = atof(argv[5]);
    } else {
        tp_iou = atof(argv[4]);
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int im = 0; im < n_img; im++) {
        for (int p = pred_off[im]; p < pred_off[im + 1]; p++) {
            const float* pb = &pred_box[(size_t)p * 4];
            if (mode == "match") {
                double best = std::min(tp_iou, 1.0 - 1e-10);
                int m = -1;
                for (int g = gt_off[im]; g < gt_off[im + 1]; g++) {
                    if (taken[g] || gt_cls[g] != pred_cls[p]) continue;
                    double v = iou_of(pb, &gt_box[(size_t)g * 4]);
                    if (v <= 0 || v < best) continue;
                    best = v; m = g;
                }
                if (m >= 0) { taken[m] = 1; matched[p] = m; }
                continue;
            }
            // ---- taxonomy ----
            double best_tp = tp_iou; int m = -1;                  // best same-class untaken GT, IoU >= tp_iou
            double best_dup = -1, best_cls = -1, best_loc = -1;   // best IoU per error type (strict '>' => first wins ties)
            int g_dup = -1, g_cls = -1, g_loc = -1;
            for (int g = gt_off[im]; g < gt_off[im + 1]; g++) {
                double v = iou_of(pb, &gt_box[(size_t)g * 4]);
                bool same = gt_cls[g] == pred_cls[p];
                if (same && !taken[g]) {
                    if (v >= best_tp) { best_tp = v; m = g; }
                }
                if (same && taken[g] && v >= tp_iou && v > best_dup) { best_dup = v; g_dup = g; }
                if (!same && v >= tp_iou && v > best_cls) { best_cls = v; g_cls = g; }
                if (same && v >= loc_iou && v < tp_iou && v > best_loc) { best_loc = v; g_loc = g; }
            }
            if (m >= 0) { taken[m] = 1; label[p] = 0; ref[p] = m; }
            else if (g_dup >= 0) { label[p] = 3; ref[p] = g_dup; }
            else if (g_cls >= 0) { label[p] = 1; ref[p] = g_cls; }
            else if (g_loc >= 0) { label[p] = 2; ref[p] = g_loc; }
            else { label[p] = 4; ref[p] = -1; }
        }
    }
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    FILE* o = fopen(argv[3], "wb");
    int32_t oh[2] = {n_pred, n_gt};
    fwrite(oh, 4, 2, o);
    if (mode == "taxonomy") {
        std::vector<int32_t> missed(n_gt);
        for (int g = 0; g < n_gt; g++) missed[g] = taken[g] ? 0 : 1;
        fwrite(label.data(), 4, n_pred, o); fwrite(ref.data(), 4, n_pred, o); fwrite(missed.data(), 4, n_gt, o);
    } else {
        fwrite(matched.data(), 4, n_pred, o);
    }
    fclose(o);
    printf("compute_seconds %.6f\n", sec);
    return 0;
}
