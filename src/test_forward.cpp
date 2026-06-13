// test_forward.cpp — Test binary for comparing C++ vs Python forward pass
// Usage: ./test_forward /tmp/weights2

#include "zonos2.h"
#include <cstdio>
#include <cmath>
#include <vector>

static float compute_rms(const float* data, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) sum += (double)data[i] * data[i];
    return (float)std::sqrt(sum / n);
}

static float compute_corr(const float* a, const float* b, int n) {
    double sum_a = 0, sum_b = 0, sum_ab = 0, sum_aa = 0, sum_bb = 0;
    for (int i = 0; i < n; i++) {
        sum_a += a[i];
        sum_b += b[i];
        sum_ab += (double)a[i] * b[i];
        sum_aa += (double)a[i] * a[i];
        sum_bb += (double)b[i] * b[i];
    }
    double num = n * sum_ab - sum_a * sum_b;
    double den = std::sqrt((n * sum_aa - sum_a * sum_a) * (n * sum_bb - sum_b * sum_b));
    return (float)(num / std::max(den, 1e-12));
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <weights_dir>\n", argv[0]);
        return 1;
    }

    // Load weights
    Zonos2Weights w;
    if (!load_zonos2_weights(argv[1], w)) {
        fprintf(stderr, "Failed to load weights\n");
        return 1;
    }

    const auto& cfg = w.cfg;
    int dim = cfg.dim;
    int n_codebooks = cfg.n_codebooks;
    int frame_width = n_codebooks + 1;

    // Load Python reference input
    std::vector<int32_t> py_input;
    {
        FILE* f = fopen("/tmp/py_embed_in.bin", "rb");
        if (!f) { fprintf(stderr, "No Python input file — run test_forward.py first\n"); return 1; }
        fseek(f, 0, SEEK_END);
        py_input.resize(ftell(f) / sizeof(int32_t));
        fseek(f, 0, SEEK_SET);
        fread(py_input.data(), sizeof(int32_t), py_input.size(), f);
        fclose(f);
    }

    int n_tokens = py_input.size() / frame_width;
    printf("Testing with %d tokens, frame_width=%d\n", n_tokens, frame_width);

    // Initialize state
    Zonos2State state;
    int kv_dim = cfg.kv_dim();
    int max_seqlen = cfg.max_seqlen;
    int num_moe = 0;
    for (int l = 0; l < cfg.n_layers; l++) if (cfg.is_moe_layer(l)) num_moe++;

    state.kv_cache.resize(cfg.n_layers * 2 * kv_dim * max_seqlen, 0.0f);
    state.router_states_buf.resize(num_moe * max_seqlen * cfg.moe_router_dim, 0.0f);
    state.num_moe_layers = num_moe;

    // Run emb_norm manually (just the embedding + norm part, not full forward)
    // We'll use zonos2_forward for the full pipeline
    std::vector<float> logits(n_tokens * n_codebooks * cfg.audio_vocab);
    if (!zonos2_forward(w, state, py_input.data(), n_tokens, 0, logits.data(), nullptr)) {
        fprintf(stderr, "Forward pass failed\n");
        return 1;
    }

    // Compare embedding output
    // (We can't easily extract intermediate activations without modifying zonos2_forward)
    // Let's just check the final logits for sanity
    printf("Logits shape: [%d, %d, %d]\n", n_tokens, n_codebooks, cfg.audio_vocab);
    printf("Logits RMS: %.6f\n", compute_rms(logits.data(), (int)logits.size()));

    // Check for NaN
    int nan_count = 0;
    for (size_t i = 0; i < logits.size(); i++)
        if (std::isnan(logits[i])) nan_count++;
    printf("NaN count: %d / %zu\n", nan_count, logits.size());

    // Print first few logits
    printf("First position, codebook 0, first 10 logits:\n  ");
    for (int i = 0; i < 10; i++) {
        printf("%.4f ", logits[i]);
    }
    printf("\n");

    free_zonos2_weights(w);
    return 0;
}
