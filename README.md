# Zonos2.cpp — Pure C++ Zonos2 TTS

100% matching C++ implementation of [Zyphra/ZONOS2](https://huggingface.co/Zyphra/ZONOS2) — autoregressive transformer TTS with Mixture of Experts.

## Architecture

```
Text (UTF-8 bytes) → Multi-Embedding (9 codebooks + text)
  → 28-layer Transformer Decoder (MoE layers 3-26)
  → Multi-Output Head → 9×1026 logits
  → Autoregressive sampling → DAC codec tokens
  → DAC Decoder → 44.1 kHz PCM audio
```

Model specs:
- 28 layers, 2048-dim hidden, 128 head_dim
- 16 Q heads, 4 KV heads (GQA)
- MoE: 16 experts, top-1 (layer 26: top-2), SonicMoE interleaved gate/up
- EDA router (Expert Decision Awareness)
- QK RMSNorm with temperature scaling
- Headwise gating (sigmoid)
- Interleaved RoPE (is_neox=false)
- SwiGLU FFN (dense layers) + SonicMoE (MoE layers)
- Logit softcapping (tanh, cap=15.0)
- BF16 weights → F32 at load time

## Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

Requires: ggml (system package), C++20 compiler.

## Usage

### 1. Extract weights

```bash
python3 scripts/extract_weights.py model.pth weights/
```

Downloads model from HuggingFace and converts BF16→F32 to `weights/` directory.

### 2. Generate speech

```bash
./build/zonos2_cli --model-dir weights/ --output output.pcm "Hello world"
```

Output: raw PCM float32, 44.1 kHz mono.

Convert to WAV:
```bash
ffmpeg -f f32le -ar 44100 -ac 1 -i output.pcm output.wav
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-dir` | (required) | Path to extracted weights |
| `--output` | output.pcm | Output audio file |
| `--seed` | 42 | Random seed |
| `--temperature` | 0.7 | Sampling temperature |
| `--top-p` | 0.9 | Nucleus sampling |
| `--top-k` | 50 | Top-k sampling |
| `--max-tokens` | 500 | Maximum output frames |
| `--speaker` | (none) | Speaker embedding .bin file |

## Verification

Compare against Python reference:
```bash
python3 scripts/compare_models.py --text "Hello" \
  --model-path Zyphra/ZONOS2 \
  --weights-dir weights/
```

## References

- [Zyphra/ZONOS2](https://github.com/Zyphra/ZONOS2) — Python inference (Mini-SGLang)
- [Zonos V2 Technical Report](https://www.zyphra.com/our-work/zonos2)
- [ggml](https://github.com/ggerganov/ggml) — Tensor library

## License

Apache 2.0 (matching upstream model)
