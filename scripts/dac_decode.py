#!/usr/bin/env python3.12
"""dac_decode.py — Decode Zonos2 DAC codec tokens to WAV audio.

Usage: python3.12 dac_decode.py input.codes.bin output.wav

Requires: descript-audio-codec (pip install descript-audio-codec)
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

codes_t = torch.tensor(codes, dtype=torch.int64).unsqueeze(0)
import dac
m = dac.DAC.load(dac.utils.download(model_type='44khz')).eval()
codes_t = torch.clamp(codes_t, max=1023)
z = m.quantizer.from_codes(codes_t.permute(0, 2, 1))[0]
audio = m.decode(z).float().squeeze()
audio.detach().numpy().astype(np.float32).tofile(output_path)
print(f"Decoded {len(audio)} samples to {output_path}")
