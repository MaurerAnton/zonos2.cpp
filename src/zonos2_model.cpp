// zonos2_model.cpp — Pure C++ Zonos2 model implementation using ggml
// Corrected: per-token RMSNorm, QK norm, KV cache, GQA attention, EDA router

#include "zonos2.h"
#include <ggml.h>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm>

static float* tensor_data(ggml_tensor* t) {
    return (float*)((char*)t->data + t->view_offs);
}

// ============================================================
// RMS Norm: per-token along last dim
// x: [dim, n_tokens], normalized token-by-token
// ============================================================
static void rms_norm_per_token(
    float* x, int dim, int n_tokens, float eps,
    const float* weight = nullptr  // optional scale weight [dim]
) {
    for (int t = 0; t < n_tokens; t++) {
        float* xt = x + t * dim;
        float ss = 0;
        for (int d = 0; d < dim; d++) ss += xt[d] * xt[d];
        float rms = std::sqrt(ss / dim + eps);
        float rcp = 1.0f / rms;
        if (weight) {
            for (int d = 0; d < dim; d++) xt[d] = xt[d] * rcp * weight[d];
        } else {
            for (int d = 0; d < dim; d++) xt[d] *= rcp;
        }
    }
}

// ============================================================
// RoPE interleaved (is_neox=false): alternate cos/sin on [2i, 2i+1]
// q: [n_heads * head_dim, n_tokens] in column-major: q[h*head_dim + d + t*stride]
// Actually stored: q[h*head_dim + d, t] where stride = n_heads * head_dim
//
// Python layout: q.view(-1, n_heads, head_dim) -> [total_tokens, n_heads, head_dim]
// So element [tok, head, dim] at q[tok * n_heads * head_dim + head * head_dim + dim]
//
// In our C++ [n_heads*head_dim, n_tokens]: column=tok, row=head*head_dim+dim
// So q[h*head_dim + d + t * (n_heads*head_dim)]
// ============================================================
static void rope_interleaved_inplace(
    float* q_data, float* k_data,
    int n_tokens, int n_q_heads, int n_kv_heads, int head_dim,
    const int* positions, float theta
) {
    int q_stride = n_q_heads * head_dim;
    int k_stride = n_kv_heads * head_dim;
    int half_dim = head_dim / 2;

    // Precompute frequencies
    std::vector<float> freq(half_dim);
    for (int i = 0; i < half_dim; i++)
        freq[i] = 1.0f / std::pow(theta, (2.0f * i) / head_dim);

    for (int t = 0; t < n_tokens; t++) {
        int pos = positions[t];
        // Compute cos/sin for this position
        std::vector<float> cos_vals(half_dim), sin_vals(half_dim);
        for (int i = 0; i < half_dim; i++) {
            cos_vals[i] = std::cos(pos * freq[i]);
            sin_vals[i] = std::sin(pos * freq[i]);
        }

        // Q
        for (int h = 0; h < n_q_heads; h++) {
            float* qh = q_data + h * head_dim + t * q_stride;
            for (int i = 0; i < half_dim; i++) {
                int d0 = 2 * i, d1 = 2 * i + 1;
                float q0 = qh[d0], q1 = qh[d1];
                qh[d0] = q0 * cos_vals[i] - q1 * sin_vals[i];
                qh[d1] = q1 * cos_vals[i] + q0 * sin_vals[i];
            }
        }
        // K
        for (int h = 0; h < n_kv_heads; h++) {
            float* kh = k_data + h * head_dim + t * k_stride;
            for (int i = 0; i < half_dim; i++) {
                int d0 = 2 * i, d1 = 2 * i + 1;
                float k0 = kh[d0], k1 = kh[d1];
                kh[d0] = k0 * cos_vals[i] - k1 * sin_vals[i];
                kh[d1] = k1 * cos_vals[i] + k0 * sin_vals[i];
            }
        }
    }
}

// ============================================================
// GELU (tanh approximation)
// ============================================================
static inline float gelu_f32(float x) {
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + std::tanh(inner));
}

// ============================================================
// SiLU
// ============================================================
static inline float silu_f32(float x) {
    return x / (1.0f + std::exp(-x));
}

// ============================================================
// Embedding lookup (manual, no ggml needed)
// weight: [embedding_dim, num_embeddings] — row i = tokens i's embedding
// token_ids: [n_tokens]
// output: [dim, n_tokens]
// ============================================================
static void embedding_sum(
    const Zonos2Weights& w, const Zonos2Config& cfg,
    const int32_t* input_ids,  // [n_tokens, frame_width]
    int n_tokens, int frame_width,
    float* output  // [dim, n_tokens]
) {
    int dim = cfg.dim;
    memset(output, 0, n_tokens * dim * sizeof(float));

    for (int t = 0; t < n_tokens; t++) {
        float* out_t = output + t * dim;

        // Codebook embeddings (0..n_codebooks-1)
        for (int cb = 0; cb < cfg.n_codebooks; cb++) {
            int tid = input_ids[t * frame_width + cb];
            if (tid >= 0 && tid < cfg.audio_vocab) {
                const float* emb = w.codebook_embeds[cb].weight.ptr();
                const float* row = emb + tid * dim;
                for (int d = 0; d < dim; d++) out_t[d] += row[d];
            }
        }

        // Text embedding (last column)
        int tid = input_ids[t * frame_width + cfg.n_codebooks];
        if (tid >= 0 && tid <= cfg.text_vocab) {
            const float* emb = w.text_embed.weight.ptr();
            const float* row = emb + tid * dim;
            for (int d = 0; d < dim; d++) out_t[d] += row[d];
        }
    }
}

// ============================================================
// Speaker injection: add speaker embedding at position 0
// (replaces the embedding at position 0 with speaker_proj(speaker_emb))
// ============================================================
static void inject_speaker(
    const Zonos2Weights& w, const Zonos2Config& cfg,
    const std::vector<float>& speaker_emb,
    float* x, int n_tokens, int dim
) {
    if (speaker_emb.empty() || !cfg.speaker_enabled) return;
    if (!w.speaker_proj.weight.empty()) {
        // LDA projection
        std::vector<float> lda_out(cfg.speaker_lda_dim, 0.0f);
        if (!w.speaker_lda.weight.empty()) {
            const float* lda_w = w.speaker_lda.weight.ptr();  // [lda_dim, speaker_dim]
            const float* lda_b = w.speaker_lda.bias.empty() ? nullptr : w.speaker_lda.bias.ptr();
            for (int o = 0; o < cfg.speaker_lda_dim; o++) {
                float s = lda_b ? lda_b[o] : 0;
                for (int i = 0; i < cfg.speaker_embedding_dim; i++)
                    s += speaker_emb[i] * lda_w[i + o * cfg.speaker_embedding_dim];
                lda_out[o] = s;
            }
        } else {
            memcpy(lda_out.data(), speaker_emb.data(),
                   std::min(speaker_emb.size(), lda_out.size()) * sizeof(float));
        }

        // Speaker projection
        const float* sp_w = w.speaker_proj.weight.ptr();  // [dim, lda_dim]
        const float* sp_b = w.speaker_proj.bias.empty() ? nullptr : w.speaker_proj.bias.ptr();
        float* x0 = x;  // position 0
        for (int o = 0; o < dim; o++) {
            float s = sp_b ? sp_b[o] : 0;
            for (int i = 0; i < cfg.speaker_lda_dim; i++)
                s += lda_out[i] * sp_w[i + o * cfg.speaker_lda_dim];
            x0[o] = s;  // Replace embedding at pos 0
        }
    }
}

// ============================================================
// Linear layer (manual)
// weight: [in_features, out_features] — row i = input dim, col o = output dim
// x: [in_features, n_tokens]
// bias: [out_features] or nullptr
// output: [out_features, n_tokens]
// ============================================================
static void linear_forward(
    const float* weight, int in_f, int out_f,
    const float* x, int n_tokens,
    float* output,
    const float* bias = nullptr
) {
    for (int t = 0; t < n_tokens; t++) {
        const float* xt = x + t * in_f;
        float* ot = output + t * out_f;
        for (int o = 0; o < out_f; o++) {
            float s = bias ? bias[o] : 0;
            for (int i = 0; i < in_f; i++)
                s += xt[i] * weight[i + o * in_f];
            ot[o] = s;
        }
    }
}

// ============================================================
// Attention forward
// ============================================================
static void attention_forward(
    const AttentionW& attn, const Zonos2Config& cfg,
    const float* x, int n_tokens, int start_pos,
    const int* positions,
    float* kv_cache_k,   // [kv_dim, max_seqlen]
    float* kv_cache_v,   // [kv_dim, max_seqlen]
    float* output         // [dim, n_tokens]
) {
    int dim = cfg.dim;
    int n_heads = cfg.n_qo_heads();
    int n_kv = cfg.n_kv();
    int head_dim = cfg.head_dim;
    int kv_dim = cfg.kv_dim();

    // Q projection
    std::vector<float> q(n_heads * head_dim * n_tokens);
    linear_forward(attn.wq.weight.ptr(), dim, dim, x, n_tokens, q.data());

    // KV projection (fused)
    std::vector<float> kv(kv_dim * 2 * n_tokens);
    linear_forward(attn.wkv.weight.ptr(), dim, kv_dim * 2, x, n_tokens, kv.data());

    // Split K, V
    std::vector<float> k(kv_dim * n_tokens), v(kv_dim * n_tokens);
    for (int t = 0; t < n_tokens; t++) {
        memcpy(k.data() + t * kv_dim, kv.data() + t * (kv_dim * 2), kv_dim * sizeof(float));
        memcpy(v.data() + t * kv_dim, kv.data() + t * (kv_dim * 2) + kv_dim, kv_dim * sizeof(float));
    }

    // QK RMS Norm (per head, per token)
    // Q: [n_heads, head_dim, n_tokens] stored as [n_heads*head_dim, n_tokens]
    {
        const float* temp_data = attn.temp.ptr();
        for (int h = 0; h < n_heads; h++) {
            for (int t = 0; t < n_tokens; t++) {
                float* qh = q.data() + h * head_dim + t * (n_heads * head_dim);
                float ss = 0;
                for (int d = 0; d < head_dim; d++) ss += qh[d] * qh[d];
                float rms = std::sqrt(ss / head_dim + 1e-6f);
                float scale = std::fabs(temp_data[h]) / rms;
                for (int d = 0; d < head_dim; d++) qh[d] *= scale;
            }
        }
    }
    {
        for (int h = 0; h < n_kv; h++) {
            for (int t = 0; t < n_tokens; t++) {
                float* kh = k.data() + h * head_dim + t * kv_dim;
                float ss = 0;
                for (int d = 0; d < head_dim; d++) ss += kh[d] * kh[d];
                float rms = std::sqrt(ss / head_dim + 1e-6f);
                float scale = 1.0f / rms;
                for (int d = 0; d < head_dim; d++) kh[d] *= scale;
            }
        }
    }

    // RoPE
    rope_interleaved_inplace(q.data(), k.data(), n_tokens, n_heads, n_kv, head_dim, positions, cfg.rope_theta);

    // Write K, V to cache
    for (int t = 0; t < n_tokens; t++) {
        memcpy(kv_cache_k + (start_pos + t) * kv_dim, k.data() + t * kv_dim, kv_dim * sizeof(float));
        memcpy(kv_cache_v + (start_pos + t) * kv_dim, v.data() + t * kv_dim, kv_dim * sizeof(float));
    }

    // Attention computation (GQA)
    int total_kv = start_pos + n_tokens;
    float scale = 1.0f / std::sqrt((float)head_dim);
    std::vector<float> attn_out(n_heads * head_dim * n_tokens, 0.0f);

    for (int hq = 0; hq < n_heads; hq++) {
        int hkv = hq * n_kv / n_heads;  // GQA mapping

        for (int t = 0; t < n_tokens; t++) {
            // Query for this head/token
            float* qh = q.data() + hq * head_dim + t * (n_heads * head_dim);

            // Compute scores with ALL KV positions (cached + current)
            int cur_pos = start_pos + t;
            std::vector<float> scores(cur_pos + 1);

            float max_score = -1e9f;
            for (int s = 0; s <= cur_pos; s++) {
                float dot = 0;
                if (s < start_pos) {
                    // From cache
                    const float* kc = kv_cache_k + hkv * head_dim + s * kv_dim;
                    for (int d = 0; d < head_dim; d++) dot += qh[d] * kc[d];
                } else {
                    // From current batch
                    int local_s = s - start_pos;
                    const float* kc = k.data() + hkv * head_dim + local_s * kv_dim;
                    for (int d = 0; d < head_dim; d++) dot += qh[d] * kc[d];
                }
                scores[s] = dot * scale;
                if (s > cur_pos) scores[s] = -1e9f;  // causal mask
                if (scores[s] > max_score) max_score = scores[s];
            }

            // Softmax
            float sum_exp = 0;
            for (int s = 0; s <= cur_pos; s++) {
                scores[s] = std::exp(scores[s] - max_score);
                sum_exp += scores[s];
            }
            for (int s = 0; s <= cur_pos; s++) scores[s] /= sum_exp;

            // Weighted sum of V
            float* out_h = attn_out.data() + hq * head_dim + t * (n_heads * head_dim);
            memset(out_h, 0, head_dim * sizeof(float));
            for (int s = 0; s <= cur_pos; s++) {
                float w = scores[s];
                const float* vc;
                if (s < start_pos) {
                    vc = kv_cache_v + hkv * head_dim + s * kv_dim;
                } else {
                    int local_s = s - start_pos;
                    vc = v.data() + hkv * head_dim + local_s * kv_dim;
                }
                for (int d = 0; d < head_dim; d++) out_h[d] += w * vc[d];
            }
        }
    }

    // Headwise gating
    {
        std::vector<float> gate(n_heads * n_tokens);
        linear_forward(attn.gater.weight.ptr(), dim, n_heads, x, n_tokens, gate.data());
        // Sigmoid
        for (int i = 0; i < n_heads * n_tokens; i++)
            gate[i] = 1.0f / (1.0f + std::exp(-gate[i]));

        for (int h = 0; h < n_heads; h++) {
            for (int t = 0; t < n_tokens; t++) {
                float g = gate[h + t * n_heads];
                float* out_h = attn_out.data() + h * head_dim + t * (n_heads * head_dim);
                for (int d = 0; d < head_dim; d++) out_h[d] *= g;
            }
        }
    }

    // Concatenate heads: [n_heads*head_dim, n_tokens]
    std::vector<float> o_in(dim * n_tokens);
    for (int t = 0; t < n_tokens; t++) {
        for (int h = 0; h < n_heads; h++) {
            const float* src = attn_out.data() + h * head_dim + t * (n_heads * head_dim);
            memcpy(o_in.data() + t * dim + h * head_dim, src, head_dim * sizeof(float));
        }
    }

    // Output projection
    linear_forward(attn.wo.weight.ptr(), dim, dim, o_in.data(), n_tokens, output);
}

// ============================================================
// Dense FeedForward (SwiGLU: up * silu(gate))
// ============================================================
static void feedforward_forward(
    const FeedForwardW& ffn, const Zonos2Config& cfg,
    const float* x, int n_tokens,
    float* output  // [dim, n_tokens]
) {
    int dim = cfg.dim;
    int ffn_dim = cfg.ffn_dim;

    // w_in: [ffn_dim*2, dim] -> gate (first ffn_dim rows) + up (last ffn_dim rows)
    std::vector<float> h_gate(ffn_dim * 2 * n_tokens);
    linear_forward(ffn.w_in.weight.ptr(), dim, ffn_dim * 2, x, n_tokens, h_gate.data());

    // Split: h = first half (up), gate = second half
    std::vector<float> y(ffn_dim * n_tokens);
    for (int t = 0; t < n_tokens; t++) {
        for (int i = 0; i < ffn_dim; i++) {
            float up_val = h_gate[t * (ffn_dim * 2) + i];
            float gate_val = h_gate[t * (ffn_dim * 2) + ffn_dim + i];
            y[t * ffn_dim + i] = up_val * silu_f32(gate_val);
        }
    }

    // w_out: [dim, ffn_dim]
    linear_forward(ffn.w_out.weight.ptr(), ffn_dim, dim, y.data(), n_tokens, output);
}

// ============================================================
// MoE Router (with EDA)
// ============================================================
static void router_forward(
    const RouterW& router, const Zonos2Config& cfg,
    const float* x, int n_tokens, int layer_id,
    float* router_states_in,
    float* router_states_out,
    float* topk_weights,  // [n_tokens * topk]
    int* topk_indices      // [n_tokens * topk]
) {
    int dim = cfg.dim;
    int router_dim = cfg.moe_router_dim;
    int n_experts = cfg.moe_n_experts;
    int topk = cfg.get_topk(layer_id);

    // Down project
    std::vector<float> hs(n_tokens * router_dim);
    linear_forward(router.down_proj.weight.ptr(), dim, router_dim, x, n_tokens, hs.data(),
                   router.down_proj.bias.empty() ? nullptr : router.down_proj.bias.ptr());

    // EDA: blend with previous router states
    if (router.use_eda && router_states_in) {
        const float* scale = router.router_states_scale.ptr();
        for (int t = 0; t < n_tokens; t++) {
            for (int d = 0; d < router_dim; d++) {
                hs[t * router_dim + d] += router_states_in[t * router_dim + d] * scale[d];
            }
        }
    }

    // Save for next layer
    if (router_states_out) {
        memcpy(router_states_out, hs.data(), n_tokens * router_dim * sizeof(float));
    }

    // RMSNorm (EDA norm)
    rms_norm_per_token(hs.data(), router_dim, n_tokens, cfg.norm_eps,
                       router.rmsnorm_eda.weight.ptr());

    // Router MLP: GELU -> Linear -> GELU -> Linear
    std::vector<float> tmp1(n_tokens * router_dim);
    linear_forward(router.mlp_0.weight.ptr(), router_dim, router_dim, hs.data(), n_tokens, tmp1.data(),
                   router.mlp_0.bias.empty() ? nullptr : router.mlp_0.bias.ptr());
    for (int i = 0; i < n_tokens * router_dim; i++) tmp1[i] = gelu_f32(tmp1[i]);

    std::vector<float> tmp2(n_tokens * router_dim);
    linear_forward(router.mlp_2.weight.ptr(), router_dim, router_dim, tmp1.data(), n_tokens, tmp2.data(),
                   router.mlp_2.bias.empty() ? nullptr : router.mlp_2.bias.ptr());
    for (int i = 0; i < n_tokens * router_dim; i++) tmp2[i] = gelu_f32(tmp2[i]);

    // Final linear -> expert logits [n_experts, n_tokens]
    std::vector<float> expert_logits(n_tokens * n_experts);
    linear_forward(router.mlp_4.weight.ptr(), router_dim, n_experts, tmp2.data(), n_tokens, expert_logits.data());

    // Softmax over experts
    for (int t = 0; t < n_tokens; t++) {
        float* el = expert_logits.data() + t * n_experts;
        float max_val = -1e9f;
        for (int e = 0; e < n_experts; e++) max_val = std::max(max_val, el[e]);
        float sum_exp = 0;
        for (int e = 0; e < n_experts; e++) {
            el[e] = std::exp(el[e] - max_val);
            sum_exp += el[e];
        }
        for (int e = 0; e < n_experts; e++) el[e] /= sum_exp;
    }

    // Balanced top-k
    const float* biases = router.balancing_biases.ptr();
    for (int t = 0; t < n_tokens; t++) {
        const float* probs = expert_logits.data() + t * n_experts;

        // Apply bias and sort
        std::vector<std::pair<float, int>> scored;
        for (int e = 0; e < n_experts; e++)
            scored.push_back({probs[e] + biases[e], e});
        std::sort(scored.begin(), scored.end(), std::greater<>());

        // Top-k
        float sum_w = 0;
        for (int k = 0; k < topk; k++) {
            int e = scored[k].second;
            float w = probs[e];  // original prob (pre-bias)
            topk_weights[t * topk + k] = w;
            topk_indices[t * topk + k] = e;
            sum_w += w;
        }
        if (sum_w > 0) {
            for (int k = 0; k < topk; k++)
                topk_weights[t * topk + k] /= sum_w;
        }
    }
}

// ============================================================
// MoE FeedForward
// ============================================================
static void moe_ffn_forward(
    const MoEFeedForwardW& moe, const Zonos2Config& cfg,
    const float* x, int n_tokens, int layer_id,
    float* router_states_in,
    float* router_states_out,
    float* output  // [dim, n_tokens]
) {
    int dim = cfg.dim;
    int ffn_dim = cfg.ffn_dim;
    int n_experts = cfg.moe_n_experts;
    int topk = cfg.get_topk(layer_id);

    // Router
    std::vector<float> topk_weights(n_tokens * topk);
    std::vector<int> topk_indices(n_tokens * topk);
    router_forward(moe.router, cfg, x, n_tokens, layer_id,
                   router_states_in, router_states_out,
                   topk_weights.data(), topk_indices.data());

    // Expert computation
    memset(output, 0, n_tokens * dim * sizeof(float));

    const float* gate_up = moe.experts.gate_up_proj.ptr();  // [n_experts, ffn_dim*2, dim]
    const float* down = moe.experts.down_proj.ptr();         // [n_experts, dim, ffn_dim]

    for (int t = 0; t < n_tokens; t++) {
        for (int k = 0; k < topk; k++) {
            float weight = topk_weights[t * topk + k];
            int e = topk_indices[t * topk + k];

            // SonicMoE: interleaved gate/up
            // Even rows (0,2,4,...) = gate projection
            // Odd rows (1,3,5,...) = up projection
            const float* e_gate_up = gate_up + e * (ffn_dim * 2) * dim;
            const float* e_down = down + e * dim * ffn_dim;

            std::vector<float> hidden(ffn_dim);
            for (int o = 0; o < ffn_dim; o++) {
                // gate: row (o*2), up: row (o*2+1)
                const float* gate_row = e_gate_up + (o * 2) * dim;
                const float* up_row = e_gate_up + (o * 2 + 1) * dim;

                float g = 0, u = 0;
                for (int i = 0; i < dim; i++) {
                    g += x[t * dim + i] * gate_row[i];
                    u += x[t * dim + i] * up_row[i];
                }
                hidden[o] = g * silu_f32(u);
            }

            // Down projection
            for (int o = 0; o < dim; o++) {
                float s = 0;
                for (int i = 0; i < ffn_dim; i++)
                    s += hidden[i] * e_down[o * ffn_dim + i];
                output[t * dim + o] += weight * s;
            }
        }
    }
}

// ============================================================
// Transformer Block
// ============================================================
static void transformer_block_forward(
    const LayerW& layer, const Zonos2Config& cfg,
    float* x, int n_tokens, int start_pos,
    const int* positions,
    float* kv_cache_k, float* kv_cache_v,
    float* residual,
    float* router_states_in,
    float* router_states_out,
    int layer_id
) {
    int dim = cfg.dim;

    // attention_norm: per-token RMSNorm with fused residual
    // Residual = ORIGINAL x (before normalization)
    memcpy(residual, x, n_tokens * dim * sizeof(float));
    rms_norm_per_token(x, dim, n_tokens, cfg.norm_eps,
                       layer.attention_norm.has_weight ? layer.attention_norm.weight.ptr() : nullptr);

    // Attention
    std::vector<float> attn_out(dim * n_tokens);
    attention_forward(layer.attn, cfg, x, n_tokens, start_pos, positions,
                      kv_cache_k, kv_cache_v, attn_out.data());

    // Residual add
    for (int i = 0; i < n_tokens * dim; i++)
        x[i] = attn_out.data()[i] + residual[i];

    // ffn_norm with fused residual
    // Residual = ORIGINAL x (before normalization)
    memcpy(residual, x, n_tokens * dim * sizeof(float));
    rms_norm_per_token(x, dim, n_tokens, cfg.norm_eps,
                       layer.ffn_norm.has_weight ? layer.ffn_norm.weight.ptr() : nullptr);

    // FeedForward
    std::vector<float> ffn_out(dim * n_tokens);
    if (layer.is_moe) {
        moe_ffn_forward(layer.moe_ffn, cfg, x, n_tokens, layer_id,
                        router_states_in, router_states_out, ffn_out.data());
    } else {
        feedforward_forward(layer.dense_ffn, cfg, x, n_tokens, ffn_out.data());
    }

    // Residual add
    for (int i = 0; i < n_tokens * dim; i++)
        x[i] = ffn_out.data()[i] + residual[i];
}

// ============================================================
// Main model forward pass
// ============================================================

bool zonos2_forward(
    const Zonos2Weights& w,
    Zonos2State& state,
    const int32_t* input_ids,
    int n_tokens,
    int start_pos,
    float* logits_out,
    ggml_context* ctx
) {
    (void)ctx;  // ggml not used in this manual implementation

    const Zonos2Config& cfg = w.cfg;
    int dim = cfg.dim;
    int audio_vocab = cfg.audio_vocab;
    int n_codebooks = cfg.n_codebooks;
    int frame_width = n_codebooks + 1;
    int kv_dim = cfg.kv_dim();
    int max_seqlen = cfg.max_seqlen;

    // 1. Multi-embedding
    std::vector<float> x(n_tokens * dim);
    embedding_sum(w, cfg, input_ids, n_tokens, frame_width, x.data());

    // 2. Embedding norm (elementwise_affine=False → no weight)
    rms_norm_per_token(x.data(), dim, n_tokens, cfg.norm_eps);

    // 3. Transformer layers
    std::vector<float> residual(n_tokens * dim);
    std::vector<int> positions(n_tokens);
    for (int t = 0; t < n_tokens; t++) positions[t] = start_pos + t;

    int moe_layer_idx = 0;
    for (int layer_id = 0; layer_id < cfg.n_layers; layer_id++) {
        const LayerW& layer = w.layers[layer_id];

        // KV cache for this layer
        float* kc = state.kv_cache.data() + layer_id * (2 * kv_dim * max_seqlen);
        float* vc = kc + kv_dim * max_seqlen;

        // Router states
        float* rs_in = nullptr;
        float* rs_out = nullptr;
        if (layer.is_moe) {
            rs_in = (moe_layer_idx > 0)
                ? state.router_states_buf.data() + (moe_layer_idx - 1) * max_seqlen * cfg.moe_router_dim
                : nullptr;
            rs_out = state.router_states_buf.data() + moe_layer_idx * max_seqlen * cfg.moe_router_dim;
            moe_layer_idx++;
        }

        transformer_block_forward(
            layer, cfg, x.data(), n_tokens, start_pos, positions.data(),
            kc, vc, residual.data(), rs_in, rs_out, layer_id
        );
    }

    // 4. Output norm (with fused residual from last layer)
    // x already contains the residual-add result, apply out_norm
    rms_norm_per_token(x.data(), dim, n_tokens, cfg.norm_eps,
                       w.out_norm.has_weight ? w.out_norm.weight.ptr() : nullptr);

    // 5. Multi-output head
    int out_dim = audio_vocab * n_codebooks;
    std::vector<float> logits_flat(n_tokens * out_dim);
    linear_forward(w.multi_output.weight.ptr(), dim, out_dim, x.data(), n_tokens, logits_flat.data());

    // Softcap and reshape to [n_tokens, n_codebooks, audio_vocab]
    for (int t = 0; t < n_tokens; t++) {
        for (int cb = 0; cb < n_codebooks; cb++) {
            float* cb_logits = logits_out + t * (n_codebooks * audio_vocab) + cb * audio_vocab;
            for (int v = 0; v < audio_vocab; v++) {
                float l = logits_flat[t * out_dim + cb * audio_vocab + v];
                if (cfg.loss_softcap > 0)
                    l = cfg.loss_softcap * std::tanh(l / cfg.loss_softcap);
                cb_logits[v] = l;
            }
        }
    }

    // DEBUG: save logits
    {
        FILE* df = fopen("/tmp/cpp_logits.bin", "wb");
        fwrite(logits_out, sizeof(float), n_tokens * n_codebooks * audio_vocab, df);
        fclose(df);
    }

    return true;
}
