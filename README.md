# zonos2.cpp — Pure C++ Zonos2 TTS

100% C++20 implementation of [Zyphra/ZONOS2](https://huggingface.co/Zyphra/ZONOS2).

28-layer autoregressive transformer with Mixture of Experts for text-to-speech.
No Python runtime dependency — generates DAC codec tokens directly.

## Architecture

```
UTF-8 bytes → Multi-Embedding → 28-layer Transformer (MoE 3–26)
  → 9-codebook DAC tokens (1024 vocab each)
  → (external DAC → 44.1 kHz PCM)
```

- 28 layers, 2048-dim, 128 head_dim, 16 Q / 4 KV (GQA)
- MoE: 16 experts, top-1, SonicMoE interleaved gate/up
- EDA router, QK RMSNorm + temperature, headwise gating
- Interleaved RoPE, SwiGLU FFN, logit softcap (tanh 15.0)
- Deterministic: same seed + same text = byte-identical tokens

## Build

```bash
mkdir build && cd build
cmake .. && make -j$(nproc)
```

Requires: ggml, C++20 compiler. No Python at runtime.

## Usage

### 1. Extract weights (one-time preprocessing)

```bash
python3 scripts/extract_weights.py model.pth weights/
```

Converts the PyTorch checkpoint (BF16, 15 GB) to raw F32 binaries (~29 GB).

### 2. Generate speech

```bash
./build/zonos2_cli --model-dir weights/ --seed 42 --output out "Hello world"
```

Output: DAC codec tokens saved as `out.codes.bin`.

### 3. Decode to audio (external DAC)

```bash
python3 scripts/dac_decode.py out.codes.bin out.wav
# or: ffmpeg -f f32le -ar 44100 -ac 1 -i out.pcm out.wav
```

## Verification

```bash
# Same seed + same text = same tokens
./build/zonos2_cli --model-dir weights/ --seed 42 --output a "Test"
./build/zonos2_cli --model-dir weights/ --seed 42 --output b "Test"
md5sum a.codes.bin b.codes.bin  # identical
```

## References

- [Zyphra/ZONOS2](https://github.com/Zyphra/ZONOS2)
- [Zonos V2 Technical Report](https://www.zyphra.com/our-work/zonos2)
- [ggml](https://github.com/ggerganov/ggml)

## License

Apache 2.0
