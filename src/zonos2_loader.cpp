// zonos2_loader.cpp — Load Zonos2 weights from extracted F32 binaries
// Expects extraction naming from scripts/extract_weights.py

#include "zonos2.h"
#include <cstdio>
#include <cstring>
#include <sys/stat.h>

static bool read_binary(const std::string& path, std::vector<float>& data, int expected_count = 0) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    int count = size / 4;
    if (expected_count > 0 && count != expected_count) {
        fprintf(stderr, "WARNING: %s has %d floats, expected %d\n", path.c_str(), count, expected_count);
    }
    data.resize(count);
    fread(data.data(), 4, count, f);
    fclose(f);
    return true;
}

static bool file_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

static bool load_tensor(const std::string& path, TensorF32& t, int n0, int n1 = 0, int n2 = 0) {
    if (!file_exists(path)) {
        fprintf(stderr, "MISSING: %s\n", path.c_str());
        return false;
    }
    int expected = n0;
    if (n1 > 0) expected *= n1;
    if (n2 > 0) expected *= n2;
    if (!read_binary(path, t.data, expected)) return false;
    t.ne[0] = n0;
    t.ne[1] = n1 > 0 ? n1 : 1;
    t.ne[2] = n2 > 0 ? n2 : 1;
    t.n_dims = n2 > 0 ? 3 : (n1 > 0 ? 2 : 1);
    return true;
}

static bool load_linear(const std::string& ldir, const std::string& name,
                         int in_f, int out_f, LinearW& w, bool has_bias) {
    if (!load_tensor(ldir + "/" + name + ".bin", w.weight, in_f, out_f)) return false;
    w.in_features = in_f;
    w.out_features = out_f;
    // Convert to BF16 and free F32 weights (like llama.cpp)
    w.w_bf16.from_f32(w.weight.ptr(), in_f * out_f);
    w.w_bf16.ne[0] = in_f;
    w.w_bf16.ne[1] = out_f;
    w.weight.data.clear();  // free F32 memory
    w.weight.data.shrink_to_fit();
    if (has_bias) {
        std::string bp = ldir + "/" + name + "_b.bin";
        if (file_exists(bp)) {
            if (!load_tensor(bp, w.bias, out_f)) return false;
        }
    }
    return true;
}

bool load_zonos2_weights(const std::string& weights_dir, Zonos2Weights& w) {
    auto& cfg = w.cfg;

    // Load config if available
    std::string cpath = weights_dir + "/config.bin";
    std::vector<float> cfg_data;
    if (read_binary(cpath, cfg_data) && cfg_data.size() >= 12) {
        int* ci = (int*)cfg_data.data();
        cfg.n_layers = ci[0];
        cfg.dim = ci[1];
        cfg.head_dim = ci[2];
        cfg.n_heads = ci[3];
        cfg.n_kv_heads = ci[4];
        cfg.ffn_dim = ci[5];
        cfg.n_codebooks = ci[6];
        cfg.codebook_size = ci[7];
        cfg.audio_vocab = ci[8];
        cfg.text_vocab = ci[9];
        cfg.eoa_id = ci[10];
        cfg.audio_pad_id = ci[11];
    }
    printf("Config: %d layers, dim=%d, heads=%d, kv=%d, ffn=%d, codebooks=%d, vocab=%d\n",
           cfg.n_layers, cfg.dim, cfg.n_heads, cfg.n_kv_heads, cfg.ffn_dim,
           cfg.n_codebooks, cfg.audio_vocab);

    // ---- Embeddings ----
    for (int i = 0; i < cfg.n_codebooks; i++) {
        char fname[64];
        snprintf(fname, sizeof(fname), "codebook_%d.bin", i);
        std::string path = weights_dir + "/" + fname;
        auto& emb = w.codebook_embeds[i];
        emb.num_embeddings = cfg.audio_vocab;
        emb.embedding_dim = cfg.dim;
        if (!load_tensor(path, emb.weight, cfg.dim, cfg.audio_vocab)) return false;
    }

    // Text embedding
    {
        std::string path = weights_dir + "/text_embed.bin";
        auto& emb = w.text_embed;
        emb.num_embeddings = cfg.text_vocab + 1;
        emb.embedding_dim = cfg.dim;
        if (!load_tensor(path, emb.weight, cfg.dim, cfg.text_vocab + 1)) return false;
    }

    // ---- Speaker (optional) ----
    {
        std::string path = weights_dir + "/speaker_lda.bin";
        if (file_exists(path)) {
            load_tensor(path, w.speaker_lda.weight, cfg.speaker_embedding_dim, cfg.speaker_lda_dim);
            w.speaker_lda.in_features = cfg.speaker_embedding_dim;
            w.speaker_lda.out_features = cfg.speaker_lda_dim;
            cfg.speaker_enabled = true;

            path = weights_dir + "/speaker_lda_b.bin";
            if (file_exists(path)) load_tensor(path, w.speaker_lda.bias, cfg.speaker_lda_dim);
        }
        path = weights_dir + "/speaker_proj.bin";
        if (file_exists(path)) {
            load_tensor(path, w.speaker_proj.weight, cfg.speaker_lda_dim, cfg.dim);
            w.speaker_proj.in_features = cfg.speaker_lda_dim;
            w.speaker_proj.out_features = cfg.dim;

            path = weights_dir + "/speaker_proj_b.bin";
            if (file_exists(path)) load_tensor(path, w.speaker_proj.bias, cfg.dim);
        }
    }

    // ---- Output ----
    if (!load_tensor(weights_dir + "/out_norm.bin", w.out_norm.weight, cfg.dim)) return false;
    w.out_norm.size = cfg.dim;
    w.out_norm.has_weight = true;

    if (!load_tensor(weights_dir + "/output.bin", w.multi_output.weight,
                     cfg.dim, cfg.audio_vocab * cfg.n_codebooks)) return false;
    w.multi_output.in_features = cfg.dim;
    w.multi_output.out_features = cfg.audio_vocab * cfg.n_codebooks;

    // ---- Layers ----
    w.layers.resize(cfg.n_layers);
    int moe_count = 0;
    int dense_count = 0;

    for (int l = 0; l < cfg.n_layers; l++) {
        char ldir[64];
        snprintf(ldir, sizeof(ldir), "/layer_%02d", l);
        std::string lpath = weights_dir + ldir;

        auto& layer = w.layers[l];
        layer.is_moe = cfg.is_moe_layer(l);

        // Attention
        if (!load_linear(lpath, "wq",    cfg.dim, cfg.dim, layer.attn.wq, false)) return false;
        if (!load_linear(lpath, "wkv",   cfg.dim, cfg.kv_dim() * 2, layer.attn.wkv, false)) return false;
        if (!load_linear(lpath, "wo",    cfg.dim, cfg.dim, layer.attn.wo, false)) return false;
        if (!load_linear(lpath, "gater", cfg.dim, cfg.n_qo_heads(), layer.attn.gater, false)) return false;

        // temp: (1, n_heads, 1) flattened → n_heads floats
        if (!load_tensor(lpath + "/temp.bin", layer.attn.temp, cfg.n_qo_heads())) return false;

        // Norms
        if (!load_tensor(lpath + "/attn_norm.bin", layer.attention_norm.weight, cfg.dim)) return false;
        layer.attention_norm.size = cfg.dim;
        layer.attention_norm.has_weight = true;

        if (!load_tensor(lpath + "/ffn_norm.bin", layer.ffn_norm.weight, cfg.dim)) return false;
        layer.ffn_norm.size = cfg.dim;
        layer.ffn_norm.has_weight = true;

        if (layer.is_moe) {
            // Router
            auto& r = layer.moe_ffn.router;
            if (!load_linear(lpath, "router_down", cfg.dim, cfg.moe_router_dim, r.down_proj, true)) return false;
            if (!load_linear(lpath, "router_mlp0", cfg.moe_router_dim, cfg.moe_router_dim, r.mlp_0, true)) return false;
            if (!load_linear(lpath, "router_mlp2", cfg.moe_router_dim, cfg.moe_router_dim, r.mlp_2, true)) return false;
            if (!load_linear(lpath, "router_mlp4", cfg.moe_router_dim, cfg.moe_n_experts, r.mlp_4, false)) return false;

            if (!load_tensor(lpath + "/router_eda.bin", r.rmsnorm_eda.weight, cfg.moe_router_dim)) return false;
            r.rmsnorm_eda.size = cfg.moe_router_dim;
            r.rmsnorm_eda.has_weight = true;

            r.use_eda = (l != cfg.moe_start_from_layer);
            if (r.use_eda) {
                if (!load_tensor(lpath + "/router_scale.bin", r.router_states_scale, cfg.moe_router_dim)) return false;
            } else {
                r.router_states_scale.data.assign(cfg.moe_router_dim, 0.0f);
                r.router_states_scale.ne[0] = cfg.moe_router_dim;
            }
            if (!load_tensor(lpath + "/router_bias.bin", r.balancing_biases, cfg.moe_n_experts)) return false;

            // Experts
            auto& exp = layer.moe_ffn.experts;
            exp.n_experts = cfg.moe_n_experts;
            exp.hidden_size = cfg.dim;
            exp.intermediate_size = cfg.ffn_dim;

            // SonicMoE w13: [n_experts, ffn_dim*2, dim]
            if (!load_tensor(lpath + "/experts_w13.bin", exp.gate_up_proj,
                            cfg.dim, cfg.ffn_dim * 2, cfg.moe_n_experts)) return false;
            // w2: [n_experts, dim, ffn_dim]
            if (!load_tensor(lpath + "/experts_w2.bin", exp.down_proj,
                            cfg.ffn_dim, cfg.dim, cfg.moe_n_experts)) return false;

            moe_count++;
        } else {
            // Dense FFN
            auto& ffn = layer.dense_ffn;
            if (!load_linear(lpath, "ffn_win",  cfg.dim, cfg.ffn_dim * 2, ffn.w_in, false)) return false;
            if (!load_linear(lpath, "ffn_wout", cfg.ffn_dim, cfg.dim, ffn.w_out, false)) return false;
            dense_count++;
        }
    }

    printf("Loaded: %d MoE layers, %d dense layers\n", moe_count, dense_count);
    return true;
}

void free_zonos2_weights(Zonos2Weights& w) {
    w.layers.clear();
}
