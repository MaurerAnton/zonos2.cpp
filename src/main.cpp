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

// Simple UTF-8 bytes tokenizer: maps each byte [0..518] to itself
// Bytes 0-518 are mapped; 519 is reserved as text_vocab padding
// Bytes > 518 are mapped to 518 (or a fallback)
static std::vector<int32_t> tokenize_text(const std::string& text, int text_vocab) {
    std::vector<int32_t> tokens;
    for (unsigned char c : text) {
        if (c <= text_vocab - 1) {
            tokens.push_back((int32_t)c);
        } else {
            tokens.push_back(text_vocab - 1);  // fallback
        }
    }
    return tokens;
}

int main(int argc, char** argv) {
    std::string model_dir;
    std::string output_path = "output.pcm";
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
    std::vector<int32_t> text_tokens = tokenize_text(text, cfg.text_vocab);
    printf("Text: \"%s\" -> %zu tokens\n", text.c_str(), text_tokens.size());

    // Build prompt: [audio_pad, ..., audio_pad, text_token] for each text token
    // Format: prepend a silence prefix (short pause before speech)
    // Following Zonos2 server convention: prepend 2 frames of silence
    int silence_frames = 2;
    int total_prompt_frames = silence_frames + text_tokens.size();
    std::vector<int32_t> prompt(total_prompt_frames * frame_width);

    // Silence frames (audio_pad_id for all codebooks, text_vocab for text)
    for (int f = 0; f < silence_frames; f++) {
        for (int cb = 0; cb < n_codebooks; cb++) {
            prompt[f * frame_width + cb] = audio_pad_id;
        }
        prompt[f * frame_width + n_codebooks] = cfg.text_vocab;  // text padding
    }

    // Text frames
    for (size_t i = 0; i < text_tokens.size(); i++) {
        int f = silence_frames + i;
        for (int cb = 0; cb < n_codebooks; cb++) {
            prompt[f * frame_width + cb] = audio_pad_id;
        }
        prompt[f * frame_width + n_codebooks] = text_tokens[i];
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

    // Decode to audio using DAC (if available)
    printf("Decoding audio with DAC...\n");
    if (!decode_dac_to_wav(audio_codes, output_path)) {
        fprintf(stderr, "DAC decode failed — saving raw codes only\n");
        fprintf(stderr, "To decode manually: python3 -c \"...\" (see README)\n");
    } else {
        printf("Saved audio to %s\n", output_path.c_str());
    }

    free_zonos2_weights(weights);
    printf("Done.\n");
    return 0;
}
