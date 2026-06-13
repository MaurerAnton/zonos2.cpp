#!/usr/bin/env python3.12
"""dac_decode.py — Decode Zonos2 DAC codec tokens to WAV audio.
Applies shear_up to remove the multi-codebook delay pattern.
"""
import struct, sys, numpy as np, torch

if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} input.codes.bin output.wav")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]

with open(input_path, 'rb') as f:
    nf = struct.unpack('i', f.read(4))[0]
    nc = struct.unpack('i', f.read(4))[0]
    codes = np.frombuffer(f.read(), dtype=np.int32).reshape(nf, nc)

print(f"Loaded {nf} frames, {nc} codebooks")

# shear_up: remove delay pattern (codebook j delayed by j frames)
PAD = 1025  # audio_pad_id
codes_t = torch.tensor(codes, dtype=torch.int64)
H, W = codes_t.shape
out = codes_t.new_full((H, W), PAD)
for j in range(W):
    if H > j:
        out[:H-j, j] = codes_t[j:, j]
codes_t = out

# Clamp and decode
codes_t = codes_t.unsqueeze(0)  # [1, frames, codebooks]
import dac
m = dac.DAC.load(dac.utils.download(model_type='44khz')).eval()
codes_t = torch.clamp(codes_t, max=1023)
z = m.quantizer.from_codes(codes_t.permute(0, 2, 1))[0]
audio = m.decode(z).float().squeeze()
audio.detach().numpy().astype(np.float32).tofile(output_path)
print(f"Decoded {len(audio)} samples to {output_path}")
