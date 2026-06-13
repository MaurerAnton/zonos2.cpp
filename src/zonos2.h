// SPDX-License-Identifier: Apache-2.0
// zonos2.h — Pure C++ Zonos2 TTS model (ggml-based)
//
// Zyphra ZONOS2: autoregressive transformer with MoE for text-to-speech
//
// Architecture: 28-layer decoder transformer, MoE layers 3-26 (16 experts, top-1),
// 9-codebook audio output, 44.1kHz via DAC decoder.
//
// Weight loading: from raw F32 binary files extracted from model.pth

#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

// ============================================================
// Model configuration (from params.json)
// ============================================================
struct Zonos2Config {
    int n_layers = 28;
    int dim = 2048;
    int head_dim = 128;
    int n_heads = 16;            // dim / head_dim
    int n_kv_heads = 4;
    int ffn_dim = 3072;          // int(1.5*dim) rounded to 256
    int n_codebooks = 9;
    int codebook_size = 1024;    // actual codebook tokens
    int audio_vocab = 1026;      // codebook_size + 2 (eoa + pad)
    int text_vocab = 519;        // UTF-8 byte vocabulary
    int eoa_id = 1024;
    int audio_pad_id = 1025;
    float norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    int max_seqlen = 6144;
    float loss_softcap = 15.0f;

    // MoE
    int moe_n_experts = 16;
    int moe_router_topk = 1;     // default topk (layer 26 uses 2)
    int moe_router_dim = 128;
    int moe_start_from_layer = 3;
    int moe_end_from_layer = 1;  // last N layers are dense

    // Speaker
    bool speaker_enabled = true;
    int speaker_embedding_dim = 2048;
    int speaker_lda_dim = 1024;

    // Derived
    int n_qo_heads() const { return n_heads; }
    int n_kv() const { return n_kv_heads; }
    int kv_dim() const { return n_kv_heads * head_dim; }
    int q_dim() const { return n_heads * head_dim; }
    int audio_vocab_total() const { return audio_vocab * n_codebooks; }

    bool is_moe_layer(int layer_id) const {
        if (moe_n_experts <= 1) return false;
        if (layer_id < moe_start_from_layer) return false;
        if (n_layers - layer_id <= moe_end_from_layer) return false;
        return true;
    }

    int get_topk(int layer_id) const {
        if (layer_id == 26) return 2;
        return moe_router_topk;
    }
};

// ============================================================
// Weight structures
// ============================================================

// Simple float buffer with shape
struct TensorF32 {
    std::vector<float> data;
    int ne[4] = {0, 0, 0, 0};
    int n_dims = 2;

    int ne0() const { return ne[0]; }
    int ne1() const { return ne[1]; }
    int ne2() const { return ne[2]; }
    int ne3() const { return ne[3]; }
    size_t nbytes() const { return data.size() * sizeof(float); }
    const float* ptr() const { return data.data(); }
    float* ptr() { return data.data(); }
    size_t nelem() const { return (size_t)ne[0] * ne[1] * ne[2] * ne[3]; }
    bool empty() const { return data.empty(); }
};

// Embedded table weight
struct EmbeddingW {
    int num_embeddings;
    int embedding_dim;  // = dim
    TensorF32 weight;   // [embedding_dim, num_embeddings] for ggml_get_rows
};

// RMSNorm weight
struct RMSNormW {
    int size;          // = dim or router_dim
    TensorF32 weight;  // [size]
    bool has_weight;   // elementwise_affine
};

// Linear layer
struct LinearW {
    int in_features;
    int out_features;
    TensorF32 weight;  // [in_features, out_features] for ggml_mul_mat
    TensorF32 bias;    // [out_features] or empty
};

// Attention weights (per layer)
struct AttentionW {
    LinearW wq;        // [dim, dim]
    LinearW wkv;       // K+V fused: [kv_dim*2, dim]
    LinearW wo;        // [dim, dim]
    LinearW gater;     // [dim, n_heads]
    TensorF32 temp;    // [1, n_heads, 1] — QK norm temperature
};

// Dense feedforward (SwiGLU)
struct FeedForwardW {
    LinearW w_in;      // gate+up fused: [ffn_dim*2, dim]
    LinearW w_out;     // [dim, ffn_dim]
};

// MoE router
struct RouterW {
    LinearW down_proj;     // [router_dim, dim] with bias
    LinearW mlp_0;         // [router_dim, router_dim] with bias
    LinearW mlp_2;         // [router_dim, router_dim] with bias
    LinearW mlp_4;         // [n_experts, router_dim] no bias
    RMSNormW rmsnorm_eda;  // [router_dim]
    TensorF32 router_states_scale;  // [router_dim] — EDA blending
    TensorF32 balancing_biases;      // [n_experts] float32
    bool use_eda;           // True for all MoE layers except first
};

// MoE experts (SonicMoE: interleaved gate/up)
struct MoEExpertsW {
    int n_experts;
    int hidden_size;       // dim
    int intermediate_size; // ffn_dim
    // gate_up_proj: [n_experts, intermediate*2, hidden] — interleaved gate/up
    TensorF32 gate_up_proj;
    // down_proj: [n_experts, hidden, intermediate]
    TensorF32 down_proj;
};

// MoE feedforward
struct MoEFeedForwardW {
    RouterW router;
    MoEExpertsW experts;
};

// Per-layer weights
struct LayerW {
    AttentionW attn;
    RMSNormW attention_norm;
    RMSNormW ffn_norm;

    bool is_moe;
    FeedForwardW dense_ffn;     // for dense layers
    MoEFeedForwardW moe_ffn;    // for MoE layers
};

// Full model weights
struct Zonos2Weights {
    Zonos2Config cfg;

    // Embeddings (index 0..8 = codebooks, index 9 = text)
    EmbeddingW codebook_embeds[9];
    EmbeddingW text_embed;

    // Speaker (optional)
    LinearW speaker_lda;         // [lda_dim, speaker_dim] with bias
    LinearW speaker_proj;        // [dim, lda_dim] with bias

    // Layers
    std::vector<LayerW> layers;  // 28 layers

    // Output
    RMSNormW out_norm;
    LinearW multi_output;        // [dim, audio_vocab * n_codebooks]
};

// ============================================================
// Generation state
// ============================================================

struct Zonos2State {
    // KV cache: [n_layers][2][max_seqlen * kv_dim] — [K, V] concatenated
    // Flattened: float cache[n_layers * 2 * max_seqlen * kv_dim]
    std::vector<float> kv_cache;
    int cached_len = 0;

    // EDA router states: [num_moe_layers][max_tokens * router_dim]
    // Accumulated across MoE layers during forward pass
    std::vector<float> router_states_buf;
    std::vector<float*> router_states_per_layer;
    int num_moe_layers = 0;

    // Token buffer: [max_seqlen * (n_codebooks+1)]
    std::vector<int32_t> token_buffer;
    int seq_len = 0;

    // Hidden states (last position output)
    std::vector<float> last_hidden;

    void reset() {
        cached_len = 0;
        seq_len = 0;
        std::fill(kv_cache.begin(), kv_cache.end(), 0.0f);
    }
};

// ============================================================
// Forward declarations
// ============================================================

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;

// Model loading
bool load_zonos2_weights(const std::string& dir_path, Zonos2Weights& w);
void free_zonos2_weights(Zonos2Weights& w);

// Core forward pass
bool zonos2_forward(
    const Zonos2Weights& w,
    Zonos2State& state,
    const int32_t* input_ids,     // [seq_len, n_codebooks+1]
    int n_tokens,                 // number of token positions to process
    int start_pos,                // position in KV cache to start
    float* logits_out,            // [n_tokens, n_codebooks, audio_vocab]
    ggml_context* ctx
);

// Generation
struct Zonos2GenParams {
    int seed = 42;
    int max_tokens = 500;
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 50;
    float min_p = 0.02f;
    float repetition_penalty = 1.1f;
    int repetition_window = 50;
};

bool zonos2_generate(
    const Zonos2Weights& w,
    const std::vector<int32_t>& prompt_ids,  // [(text_len+pre, n_codebooks+1)]
    const std::vector<float>& speaker_emb,   // optional, empty=neutral
    const Zonos2GenParams& params,
    std::vector<std::vector<int32_t>>& audio_codes,  // output: [n_frames][n_codebooks]
    int* eos_frame
);

// DAC decoder bridge (calls external Python DAC)
bool decode_dac_to_wav(
    const std::vector<std::vector<int32_t>>& codes,
    const std::string& output_path
);
