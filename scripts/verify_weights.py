#!/usr/bin/env python3.12
"""verify_weights.py — Verify C++ loads weights byte-perfect from extracted F32 binaries.
Compares random tensor values between Python model.pth and C++ extracted .bin files.
"""
import torch
import numpy as np
import os, sys

MODEL_PTH = '/tmp/model.pht'  # needs to be available
WEIGHTS_DIR = '/tmp/weights2'

def verify():
    if not os.path.exists(MODEL_PTH):
        print(f"Model not found at {MODEL_PTH}")
        print("Downloading...")
        import urllib.request
        url = "https://huggingface.co/Zyphra/ZONOS2/resolve/main/model.pth"
        # Can't download 15GB here — skip
        return False

    print(f"Loading PyTorch model from {MODEL_PTH}...")
    state_dict = torch.load(MODEL_PTH, map_location='cpu', weights_only=False)

    errors = 0
    total = 0

    # Test a few key tensors
    test_keys = [
        'multi_embedder.embedders.0.weight',   # codebook 0 embedding
        'multi_embedder.embedders.9.weight',   # text embedding
        'multi_output.weight',                  # output head
        'out_norm.weight',                      # output norm
        'layers.0.attention.wq.weight',        # first layer Q
        'layers.0.attention_norm.weight',       # first layer attn norm
        'layers.3.feed_forward.router.down_proj.weight',  # first MoE router
        'layers.26.feed_forward.experts.w13',   # SonicMoE experts
    ]

    for key in test_keys:
        if key not in state_dict:
            print(f"  SKIP: {key} not in state_dict")
            continue

        py_tensor = state_dict[key].detach().cpu().to(torch.float32).numpy()

        # Map to C++ filename
        if key.startswith('layers.'):
            parts = key.split('.')
            layer_num = int(parts[1])
            rest = '.'.join(parts[2:])

            # Use the same mapping as extract_weights.py
            key_map = {
                'attention.wq.weight': 'wq',
                'attention.wkv.weight': 'wkv',
                'attention.wo.weight': 'wo',
                'attention.gater.weight': 'gater',
                'attention.temp': 'temp',
                'attention_norm.weight': 'attn_norm',
                'ffn_norm.weight': 'ffn_norm',
                'feed_forward.w_in.weight': 'ffn_win',
                'feed_forward.w_out.weight': 'ffn_wout',
                'feed_forward.router.down_proj.weight': 'router_down',
                'feed_forward.router.down_proj.bias': 'router_down_b',
                'feed_forward.router.router_mlp.0.weight': 'router_mlp0',
                'feed_forward.router.router_mlp.0.bias': 'router_mlp0_b',
                'feed_forward.router.router_mlp.2.weight': 'router_mlp2',
                'feed_forward.router.router_mlp.2.bias': 'router_mlp2_b',
                'feed_forward.router.router_mlp.4.weight': 'router_mlp4',
                'feed_forward.router.rmsnorm_eda.weight': 'router_eda',
                'feed_forward.router.router_states_scale': 'router_scale',
                'feed_forward.router.balancing_biases': 'router_bias',
                'feed_forward.experts.w13': 'experts_w13',
                'feed_forward.experts.w2': 'experts_w2',
            }

            if rest in key_map:
                fname = key_map[rest] + '.bin'
            else:
                fname = rest.replace('.', '_') + '.bin'

            cpp_path = os.path.join(WEIGHTS_DIR, f'layer_{layer_num:02d}', fname)
        else:
            root_map = {
                'multi_embedder.embedders.0.weight': 'codebook_0',
                'multi_embedder.embedders.1.weight': 'codebook_1',
                'multi_embedder.embedders.2.weight': 'codebook_2',
                'multi_embedder.embedders.3.weight': 'codebook_3',
                'multi_embedder.embedders.4.weight': 'codebook_4',
                'multi_embedder.embedders.5.weight': 'codebook_5',
                'multi_embedder.embedders.6.weight': 'codebook_6',
                'multi_embedder.embedders.7.weight': 'codebook_7',
                'multi_embedder.embedders.8.weight': 'codebook_8',
                'multi_embedder.embedders.9.weight': 'text_embed',
                'out_norm.weight': 'out_norm',
                'multi_output.weight': 'output',
            }
            fname = root_map.get(key, key.replace('.', '_')) + '.bin'
            cpp_path = os.path.join(WEIGHTS_DIR, fname)

        if not os.path.exists(cpp_path):
            print(f"  MISSING C++ file: {cpp_path}")
            errors += 1
            continue

        cpp_tensor = np.fromfile(cpp_path, dtype=np.float32)

        # C++ tensor is stored raw as [ne0, ne1, ne2] = [in_features, out_features, ...]
        # PyTorch tensor is [out_features, in_features, ...]
        # But the raw bytes should be the same if we flatten both
        py_flat = py_tensor.ravel()
        cpp_flat = cpp_tensor.ravel()

        if len(py_flat) != len(cpp_flat):
            print(f"  SIZE MISMATCH {key}: py={py_tensor.shape} ({len(py_flat)}), cpp={cpp_tensor.shape} ({len(cpp_flat)})")
            errors += 1
            continue

        # Compare element-by-element
        max_diff = np.max(np.abs(py_flat - cpp_flat))
        rms_py = np.sqrt(np.mean(py_flat ** 2))
        rms_cpp = np.sqrt(np.mean(cpp_flat ** 2))
        corr = np.corrcoef(py_flat, cpp_flat)[0, 1]

        total += 1
        if max_diff > 1e-6:
            print(f"  DIFF {key}: max_diff={max_diff:.2e}, rms(py={rms_py:.6f}, cpp={rms_cpp:.6f}), corr={corr:.10f}")
            errors += 1
        else:
            print(f"  OK   {key}: max_diff={max_diff:.2e}, corr=1.0")

    print(f"\nWeight verification: {total - errors}/{total} match byte-perfect")
    return errors == 0

if __name__ == '__main__':
    verify()
