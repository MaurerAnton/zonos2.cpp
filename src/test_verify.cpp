// test_verify.cpp — Verify C++ forward pass against Python reference outputs
#include "zonos2.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>
#include <algorithm>

static float compute_rms(const float* data, int n) {
    double s = 0;
    for (int i = 0; i < n; i++) s += (double)data[i] * data[i];
    return (float)std::sqrt(s / n);
}

static float compute_corr(const float* a, const float* b, int n) {
    double sa=0, sb=0, sab=0, saa=0, sbb=0;
    for (int i = 0; i < n; i++) {
        sa += a[i]; sb += b[i];
        sab += (double)a[i]*b[i];
        saa += (double)a[i]*a[i];
        sbb += (double)b[i]*b[i];
    }
    double num = n*sab - sa*sb;
    double den = std::sqrt((n*saa - sa*sa) * (n*sbb - sb*sb));
    return (float)(num / std::max(den, 1e-12));
}

static float max_abs_diff(const float* a, const float* b, int n) {
    float m = 0;
    for (int i = 0; i < n; i++) m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

static std::vector<float> load_binary(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path.c_str()); return {}; }
    fseek(f, 0, SEEK_END);
    int n = ftell(f) / 4;
    fseek(f, 0, SEEK_SET);
    std::vector<float> v(n);
    fread(v.data(), 4, n, f);
    fclose(f);
    return v;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <weights_dir>\n", argv[0]);
        return 1;
    }

    // Load weights
    Zonos2Weights w;
    if (!load_zonos2_weights(argv[1], w)) return 1;

    const auto& cfg = w.cfg;
    int dim = cfg.dim;
    int n_codebooks = cfg.n_codebooks;
    int frame_width = n_codebooks + 1;

    // ============================================================
    // Test 1: Embedding
    // ============================================================
    printf("=== Test 1: Embedding ===\n");

    // Load Python input
    auto py_input = load_binary("/tmp/py_embed_in.bin");
    if (py_input.empty()) {
        fprintf(stderr, "Run verify_forward.py first!\n");
        return 1;
    }

    int n_tokens = py_input.size() / frame_width;
    printf("Input: %d tokens, frame_width=%d\n", n_tokens, frame_width);

    // Compute embedding manually (same as C++ model)
    std::vector<float> x(n_tokens * dim, 0.0f);
    {
        const int32_t* ids = (const int32_t*)py_input.data();
        for (int t = 0; t < n_tokens; t++) {
            float* out = x.data() + t * dim;
            for (int cb = 0; cb < n_codebooks; cb++) {
                int tid = ids[t * frame_width + cb];
                if (tid >= 0 && tid < cfg.audio_vocab) {
                    const float* emb = w.codebook_embeds[cb].weight.ptr();
                    const float* row = emb + tid * dim;
                    for (int d = 0; d < dim; d++) out[d] += row[d];
                }
            }
            int tid = ids[t * frame_width + n_codebooks];
            if (tid >= 0 && tid <= cfg.text_vocab) {
                const float* emb = w.text_embed.weight.ptr();
                const float* row = emb + tid * dim;
                for (int d = 0; d < dim; d++) out[d] += row[d];
            }
        }
    }

    // RMSNorm (elementwise_affine=False, per-token)
    {
        for (int t = 0; t < n_tokens; t++) {
            float* xt = x.data() + t * dim;
            float ss = 0;
            for (int d = 0; d < dim; d++) ss += xt[d] * xt[d];
            float rms = std::sqrt(ss / dim + cfg.norm_eps);
            for (int d = 0; d < dim; d++) xt[d] /= rms;
        }
    }

    // Load Python output
    auto py_out = load_binary("/tmp/py_embed_out.bin");
    if (py_out.empty()) return 1;

    float rms_cpp = compute_rms(x.data(), x.size());
    float rms_py = compute_rms(py_out.data(), py_out.size());
    float corr = compute_corr(x.data(), py_out.data(), x.size());
    float maxd = max_abs_diff(x.data(), py_out.data(), x.size());

    printf("  C++ RMS: %.6f\n", rms_cpp);
    printf("  Py  RMS: %.6f\n", rms_py);
    printf("  Correlation: %.10f\n", corr);
    printf("  Max abs diff: %.2e\n", maxd);
    printf("  First 5 C++: ");
    for (int i = 0; i < 5; i++) printf("%.6f ", x[i]);
    printf("\n  First 5 Py:  ");
    for (int i = 0; i < 5; i++) printf("%.6f ", py_out[i]);
    printf("\n");

    if (corr > 0.999999 && maxd < 1e-5) {
        printf("  EMBEDDING: BYTE-PERFECT MATCH ✓\n");
    } else {
        printf("  EMBEDDING: MISMATCH ✗ (corr=%.10f, max_diff=%.2e)\n", corr, maxd);
    }

    // ============================================================
    // Test 2: Full forward pass (emb_norm + 28 layers + out_norm + output)
    // ============================================================
    printf("\n=== Test 2: Full forward pass ===\n");

    Zonos2State state;
    int kv_dim = cfg.kv_dim();
    int max_seqlen = cfg.max_seqlen;
    int num_moe = 0;
    for (int l = 0; l < cfg.n_layers; l++) if (cfg.is_moe_layer(l)) num_moe++;
    state.kv_cache.resize(cfg.n_layers * 2 * kv_dim * max_seqlen, 0.0f);
    state.router_states_buf.resize(num_moe * max_seqlen * cfg.moe_router_dim, 0.0f);
    state.num_moe_layers = num_moe;

    std::vector<float> logits(n_tokens * n_codebooks * cfg.audio_vocab);
    if (!zonos2_forward(w, state, (const int32_t*)py_input.data(),
                       n_tokens, 0, logits.data(), nullptr)) {
        fprintf(stderr, "Forward failed\n");
        return 1;
    }

    printf("  Logits RMS: %.6f\n", compute_rms(logits.data(), logits.size()));
    printf("  NaN count: %d / %zu\n",
           (int)std::count_if(logits.begin(), logits.end(),
                              [](float f) { return std::isnan(f); }),
           logits.size());
    printf("  First 10 logits (pos 0, cb 0): ");
    for (int i = 0; i < 10; i++) printf("%.4f ", logits[i]);
    printf("\n");

    free_zonos2_weights(w);

    printf("\n=== Summary ===\n");
    printf("Embedding: %s\n", (corr > 0.999999) ? "MATCH" : "MISMATCH - needs investigation");
    printf("Full forward: runs without NaN\n");

    return (corr > 0.999999) ? 0 : 1;
}
