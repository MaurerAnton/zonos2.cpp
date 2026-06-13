// zonos2_generate.cpp — Autoregressive generation loop and sampling

#include "zonos2.h"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <random>
#include <algorithm>
#include <vector>

// Xoshiro256** PRNG
struct Xoshiro256 {
    uint64_t s[4];
    Xoshiro256(uint64_t seed = 42) {
        s[0] = seed ^ 0x9E3779B97F4A7C15ULL;
        s[1] = seed ^ 0xBF58476D1CE4E5B9ULL;
        s[2] = seed ^ 0x94D049BB133111EBULL;
        s[3] = seed ^ 0x3243F6A8885A308DULL;
        for (int i = 0; i < 20; i++) next();
    }
    uint64_t next() {
        uint64_t result = ((s[1] * 5) << 7) | ((s[1] * 5) >> 57);
        result *= 9;
        uint64_t t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = (s[3] << 45) | (s[3] >> 19);
        return result;
    }
    float randf() { return (next() >> 11) * 0x1.0p-53; }
};

// Sample one token from logits
static int sample_token(Xoshiro256& rng, const float* logits, int vocab_size,
                         float temp, int top_k, float top_p, float min_p) {
    struct Candidate { float prob; int idx; };
    std::vector<Candidate> candidates(vocab_size);

    float max_logit = -1e9f;
    for (int i = 0; i < vocab_size; i++) max_logit = std::max(max_logit, logits[i]);

    float sum_exp = 0;
    for (int i = 0; i < vocab_size; i++) {
        candidates[i].prob = std::exp((logits[i] - max_logit) / temp);
        candidates[i].idx = i;
        sum_exp += candidates[i].prob;
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.prob > b.prob; });

    // top-k
    if (top_k > 0 && top_k < vocab_size) {
        candidates.resize(top_k);
        sum_exp = 0;
        for (auto& c : candidates) sum_exp += c.prob;
    }

    // top-p
    if (top_p < 1.0f) {
        float cumsum = 0;
        size_t cutoff = 0;
        for (size_t i = 0; i < candidates.size(); i++) {
            cumsum += candidates[i].prob / sum_exp;
            cutoff = i + 1;
            if (cumsum >= top_p) break;
        }
        candidates.resize(cutoff);
        sum_exp = 0;
        for (auto& c : candidates) sum_exp += c.prob;
    }

    // min-p
    if (min_p > 0 && !candidates.empty()) {
        float max_prob = candidates[0].prob / sum_exp;
        float thresh = max_prob * min_p;
        auto it = candidates.begin();
        while (it != candidates.end() && it->prob / sum_exp >= thresh) ++it;
        candidates.erase(it, candidates.end());
        sum_exp = 0;
        for (auto& c : candidates) sum_exp += c.prob;
    }

    float r = rng.randf();
    float cumsum = 0;
    for (auto& c : candidates) {
        cumsum += c.prob / sum_exp;
        if (r <= cumsum) return c.idx;
    }
    return candidates.back().idx;
}

// ============================================================
// Main generation
// ============================================================

bool zonos2_generate(
    const Zonos2Weights& w,
    const std::vector<int32_t>& prompt_ids,
    const std::vector<float>& speaker_emb,
    const Zonos2GenParams& params,
    std::vector<std::vector<int32_t>>& audio_codes,
    int* eos_frame
) {
    const Zonos2Config& cfg = w.cfg;
    int n_codebooks = cfg.n_codebooks;
    int audio_vocab = cfg.audio_vocab;
    int frame_width = n_codebooks + 1;
    int dim = cfg.dim;
    int kv_dim = cfg.kv_dim();
    int max_seqlen = cfg.max_seqlen;

    // Initialize state
    Zonos2State state;
    int num_moe_layers = 0;
    for (int l = 0; l < cfg.n_layers; l++)
        if (cfg.is_moe_layer(l)) num_moe_layers++;

    state.kv_cache.resize(cfg.n_layers * 2 * kv_dim * max_seqlen, 0.0f);
    state.router_states_buf.resize(num_moe_layers * max_seqlen * cfg.moe_router_dim, 0.0f);
    state.num_moe_layers = num_moe_layers;
    state.cached_len = 0;
    state.seq_len = 0;
    state.token_buffer.resize(max_seqlen * frame_width, 0);

    // Copy prompt tokens
    int prompt_len = (int)prompt_ids.size() / frame_width;
    if (prompt_len < 1) {
        fprintf(stderr, "Empty prompt\n");
        return false;
    }

    memcpy(state.token_buffer.data(), prompt_ids.data(),
           prompt_ids.size() * sizeof(int32_t));
    state.seq_len = prompt_len;

    // ============================================================
    // PREFILL: run forward on all prompt tokens
    // ============================================================
    printf("Prefilling %d prompt tokens...\n", prompt_len);
    std::vector<float> prefill_logits(prompt_len * n_codebooks * audio_vocab);
    {
        if (!zonos2_forward(w, state, state.token_buffer.data(),
                           prompt_len, 0, prefill_logits.data(), nullptr)) {
            fprintf(stderr, "Prefill failed\n");
            return false;
        }
        state.cached_len = prompt_len;
    }

    // NOTE: Speaker injection happens at the embedding step.
    // For simplicity, we skip speaker injection for now.
    // The model works without speaker embedding (neutral voice).

    // ============================================================
    // SAMPLE FROM LAST PREFILL POSITION
    // ============================================================
    // The prefill returned logits for ALL prompt positions.
    // We sample from the LAST position's logits.
    Xoshiro256 rng(params.seed);

    int max_steps = params.max_tokens;
    bool eos_detected = false;
    int eos_at = -1;

    // Sample first frame from prefill logits (last position)
    {
        // logits from prefill are arranged as [n_tokens, n_codebooks, audio_vocab]
        // We want position prompt_len-1
        const float* last_logits = prefill_logits.data() + (prompt_len - 1) * n_codebooks * audio_vocab;
        
        std::vector<int32_t> first_tokens(n_codebooks);
        for (int cb = 0; cb < n_codebooks; cb++) {
            const float* cb_logits = last_logits + cb * audio_vocab;
            int tok = sample_token(rng, cb_logits, audio_vocab - 2,
                                   params.temperature, params.top_k,
                                   params.top_p, params.min_p);
            if (tok >= cfg.codebook_size) tok = cfg.codebook_size - 1;
            first_tokens[cb] = tok;
        }

        // Check EOS on first frame
        for (int cb = 0; cb < n_codebooks; cb++) {
            if (first_tokens[cb] == cfg.eoa_id) {
                eos_detected = true;
                eos_at = 0;
                break;
            }
        }
        audio_codes.push_back(first_tokens);
    }

    // ============================================================
    // GENERATION LOOP (remaining steps)
    // ============================================================
    for (int step = 1; step < max_steps && !eos_detected; step++) {
        if ((step) % 50 == 0)
            printf("  Step %d/%d...\n", step, max_steps);

        int current_pos = state.seq_len;

        // Build single-token input from last generated frame
        std::vector<int32_t> single_input(frame_width);
        const auto& last_frame = audio_codes.back();
        for (int cb = 0; cb < n_codebooks; cb++)
            single_input[cb] = (cb < (int)last_frame.size()) ? last_frame[cb] : cfg.audio_pad_id;
        single_input[n_codebooks] = cfg.text_vocab;  // text padding

        std::vector<float> logits(n_codebooks * audio_vocab);
        if (!zonos2_forward(w, state, single_input.data(),
                           1, current_pos, logits.data(), nullptr)) {
            fprintf(stderr, "Decode step %d failed\n", step);
            break;
        }
        state.cached_len++;
        state.seq_len++;

        // Append to token buffer
        memcpy(state.token_buffer.data() + current_pos * frame_width,
               single_input.data(), frame_width * sizeof(int32_t));

        // Sample next tokens
        std::vector<int32_t> next_tokens(n_codebooks);
        for (int cb = 0; cb < n_codebooks; cb++) {
            const float* cb_logits = logits.data() + cb * audio_vocab;
            int tok = sample_token(rng, cb_logits, audio_vocab - 2,
                                   params.temperature, params.top_k,
                                   params.top_p, params.min_p);
            if (tok >= cfg.codebook_size) tok = cfg.codebook_size - 1;
            next_tokens[cb] = tok;
        }

        // Check EOS
        for (int cb = 0; cb < n_codebooks; cb++) {
            if (next_tokens[cb] == cfg.eoa_id) {
                eos_detected = true;
                eos_at = step;
                break;
            }
        }

        audio_codes.push_back(next_tokens);
    }

    if (eos_frame) *eos_frame = eos_at;

    printf("Generated %zu frames (EOS at frame %d)\n", audio_codes.size(), eos_at);
    return true;
}

// ============================================================
// DAC decoder bridge (calls external Python DAC)
// ============================================================

bool decode_dac_to_wav(
    const std::vector<std::vector<int32_t>>& codes,
    const std::string& output_path
) {
    int n_frames = (int)codes.size();
    int n_codebooks = codes.empty() ? 0 : (int)codes[0].size();

    if (n_frames == 0) {
        fprintf(stderr, "No audio frames to decode\n");
        return false;
    }

    // Write codes to temp binary
    std::string tmp = output_path + ".codes.bin";
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f) return false;

    fwrite(&n_frames, sizeof(int), 1, f);
    fwrite(&n_codebooks, sizeof(int), 1, f);
    for (auto& frame : codes)
        for (auto tok : frame)
            fwrite(&tok, sizeof(int32_t), 1, f);
    fclose(f);

    // Write Python decoder script
    std::string script_path = output_path + "_decode.py";
    FILE* sf = fopen(script_path.c_str(), "w");
    if (!sf) return false;
    fprintf(sf,
        "import struct, numpy as np, torch\n"
        "with open('%s','rb') as f:\n"
        " nf=struct.unpack('i',f.read(4))[0]\n"
        " nc=struct.unpack('i',f.read(4))[0]\n"
        " codes=np.frombuffer(f.read(),dtype=np.int32).reshape(nf,nc)\n"
        "codes_t=torch.tensor(codes,dtype=torch.int64).unsqueeze(0)\n"
        "import dac\n"
        "m=dac.DAC.load(dac.utils.download(model_type='44khz')).eval()\n"
        "codes_t=torch.clamp(codes_t,max=1023)\n"
        "z=m.quantizer.from_codes(codes_t.permute(0,2,1))[0]\n"
        "audio=m.decode(z).float().squeeze()\n"
        "audio.detach().numpy().astype(np.float32).tofile('%s')\n"
        "print(f'Decoded {len(audio)} samples')\n",
        tmp.c_str(), output_path.c_str());
    fclose(sf);

    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "python3.12 %s", script_path.c_str());

    int ret = system(cmd);
    if (ret != 0) {
        fprintf(stderr, "DAC decode failed (ret=%d)\n", ret);
        return false;
    }

    // Verify output
    struct stat st;
    if (stat(output_path.c_str(), &st) == 0) {
        printf("Decoded audio: %ld samples (%.1f sec)\n",
               st.st_size / 4, (st.st_size / 4) / 44100.0);
    }

    return true;
}
