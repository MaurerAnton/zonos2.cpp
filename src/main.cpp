// main.cpp — Zonos2 CLI TTS tool
//
// Usage: zonos2_cli [options] "text to speak"
//
// Options:
//   --model-dir PATH    Path to extracted weights directory (required)
//   --output PATH       Output WAV file path (default: output.wav)
//   --seed N            Random seed (default: 42)
//   --temperature F     Sampling temperature (default: 0.7)
//   --top-p F           Nucleus sampling threshold (default: 0.9)
//   --top-k N           Top-k sampling (default: 50)
//   --max-tokens N      Maximum audio tokens (default: 500)
//   --speaker FILE      Speaker embedding .bin file (optional)

#include "zonos2.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

// Simple UTF-8 byte tokenizer with legacy symbol offset
// Zonos2 uses: token = byte + LEGACY_SYMBOL_VOCAB_SIZE (192)
// So 'H'=72 becomes 264, and BOS=2, EOS=3 wrap the sequence
static std::vector<int32_t> tokenize_text(const std::string& text) {
    std::vector<int32_t> tokens;
    tokens.push_back(2);  // BOS
    for (unsigned char c : text) {
        tokens.push_back(192 + (int32_t)c);  // byte + 192
    }
    tokens.push_back(3);  // EOS
    return tokens;
}

int main(int argc, char** argv) {
    std::string model_dir;
    std::string output_path = "output.codes";
    std::string text;
    std::string speaker_path;
    Zonos2GenParams params;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            params.seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            params.temperature = atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            params.top_p = atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            params.top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            params.max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--speaker") == 0 && i + 1 < argc) {
            speaker_path = argv[++i];
        } else if (argv[i][0] != '-') {
            if (!text.empty()) text += " ";
            text += argv[i];
        }
    }

    if (model_dir.empty() || text.empty()) {
        fprintf(stderr, "Usage: zonos2_cli --model-dir <dir> [options] \"text to speak\"\n");
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  --output PATH     Output file (default: output.pcm)\n");
        fprintf(stderr, "  --seed N          Random seed (default: 42)\n");
        fprintf(stderr, "  --temperature F   Temperature (default: 0.7)\n");
        fprintf(stderr, "  --top-p F         Top-p (default: 0.9)\n");
        fprintf(stderr, "  --top-k N         Top-k (default: 50)\n");
        fprintf(stderr, "  --max-tokens N    Max output tokens (default: 500)\n");
        fprintf(stderr, "  --speaker FILE    Speaker embedding .bin\n");
        return 1;
    }

    printf("Loading Zonos2 model from %s...\n", model_dir.c_str());

    // Load weights
    Zonos2Weights weights;
    if (!load_zonos2_weights(model_dir, weights)) {
        fprintf(stderr, "Failed to load weights\n");
        return 1;
    }

    const Zonos2Config& cfg = weights.cfg;
    int n_codebooks = cfg.n_codebooks;
    int audio_pad_id = cfg.audio_pad_id;
    int frame_width = n_codebooks + 1;

    // Tokenize text
    std::vector<int32_t> text_tokens = tokenize_text(text);
    printf("Text: \"%s\" -> %zu tokens\n", text.c_str(), text_tokens.size());

    // Build prompt matching Python zonos2 format:
    // [speaker_slot] [text_frames...] [silence_suffix_frames...]
    //
    // Speaker slot: [audio_pad*9, text_vocab]
    // Text frame: [audio_pad*9, text_token]
    // Silence suffix: 17 frames of pre-computed silence tokens (0.2s at 44.1kHz)
    
    // Pre-computed silence tokens from zonos2 prompt.py _SILENCE_TOKENS_0_2S
    const int silence_frames = 17;
    const int32_t silence_tokens[17][9] = {
        {568, 778, 338, 524, 967, 360, 728, 550, 90},
        {568, 778, 10, 674, 364, 981, 741, 378, 731},
        {568, 804, 10, 674, 364, 981, 568, 378, 731},
        {568, 804, 10, 674, 364, 981, 568, 378, 731},
        {568, 804, 10, 674, 364, 981, 568, 378, 731},
        {568, 804, 10, 674, 364, 981, 568, 378, 731},
        {568, 804, 10, 674, 364, 981, 568, 378, 731},
        {568, 804, 10, 674, 364, 981, 568, 378, 731},
        {568, 804, 10, 674, 364, 981, 568, 378, 731},
        {568, 804, 10, 674, 364, 981, 568, 378, 731},
        {568, 804, 10, 674, 364, 981, 568, 378, 731},
        {568, 804, 10, 674, 364, 981, 568, 378, 731},
        {568, 804, 10, 674, 364, 981, 568, 378, 731},
        {568, 804, 10, 674, 364, 981, 568, 378, 731},
        {568, 804, 10, 674, 364, 981, 568, 378, 731},
        {568, 804, 10, 674, 364, 981, 568, 378, 731},
        {568, 778, 721, 842, 264, 974, 989, 507, 308},
    };

    const int sheared_silence_frames = silence_frames + n_codebooks - 1;  // 17 + 9 - 1 = 25
    int total_prompt_frames = 1 + (int)text_tokens.size() + sheared_silence_frames;
    std::vector<int32_t> prompt(total_prompt_frames * frame_width);
    int pf = 0;

    // Speaker slot
    for (int cb = 0; cb < n_codebooks; cb++)
        prompt[pf * frame_width + cb] = audio_pad_id;
    prompt[pf * frame_width + n_codebooks] = cfg.text_vocab;
    pf++;

    // Text frames
    for (size_t i = 0; i < text_tokens.size(); i++) {
        for (int cb = 0; cb < n_codebooks; cb++)
            prompt[pf * frame_width + cb] = audio_pad_id;
        prompt[pf * frame_width + n_codebooks] = text_tokens[i];
        pf++;
    }

    // Silence suffix — apply shear pattern (codebook j delayed by j frames)
    // Python: shear(silence[:, :n_codebooks], audio_pad_id)
    for (int i = 0; i < sheared_silence_frames; i++) {
        for (int cb = 0; cb < n_codebooks; cb++) {
            int src_frame = i - (n_codebooks - 1 - cb);
            if (src_frame >= 0 && src_frame < silence_frames)
                prompt[pf * frame_width + cb] = silence_tokens[src_frame][cb];
            else
                prompt[pf * frame_width + cb] = audio_pad_id;
        }
        prompt[pf * frame_width + n_codebooks] = cfg.text_vocab;
        pf++;
    }

    // Load speaker embedding (optional)
    std::vector<float> speaker_emb;
    if (!speaker_path.empty()) {
        FILE* sf = fopen(speaker_path.c_str(), "rb");
        if (sf) {
            fseek(sf, 0, SEEK_END);
            speaker_emb.resize(ftell(sf) / sizeof(float));
            fseek(sf, 0, SEEK_SET);
            fread(speaker_emb.data(), sizeof(float), speaker_emb.size(), sf);
            fclose(sf);
            printf("Loaded speaker embedding: %zu floats\n", speaker_emb.size());
        }
    }

    // Generate
    printf("Generating...\n");
    std::vector<std::vector<int32_t>> audio_codes;
    int eos_frame = -1;

    if (!zonos2_generate(weights, prompt, speaker_emb, params, audio_codes, &eos_frame)) {
        fprintf(stderr, "Generation failed\n");
        return 1;
    }

    printf("Generated %zu audio frames (EOS at frame %d)\n", audio_codes.size(), eos_frame);

    // Save codes as raw binary
    {
        std::string codes_path = output_path + ".codes.bin";
        FILE* f = fopen(codes_path.c_str(), "wb");
        if (f) {
            int nf = (int)audio_codes.size();
            int nc = n_codebooks;
            fwrite(&nf, sizeof(int), 1, f);
            fwrite(&nc, sizeof(int), 1, f);
            for (auto& frame : audio_codes) {
                for (int cb = 0; cb < n_codebooks; cb++) {
                    int32_t tok = (cb < (int)frame.size()) ? frame[cb] : audio_pad_id;
                    fwrite(&tok, sizeof(int32_t), 1, f);
                }
            }
            fclose(f);
            printf("Saved %d codes to %s\n", nf, codes_path.c_str());
        }
    }

    // Save DAC codec tokens (audio decode is external)
    decode_dac_to_wav(audio_codes, output_path);

    free_zonos2_weights(weights);
    printf("Done.\n");
    return 0;
}
