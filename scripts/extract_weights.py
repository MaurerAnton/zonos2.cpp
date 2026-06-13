#!/usr/bin/env python3.12
"""extract_weights.py — Extract Zonos2 model weights from model.pth to raw F32 binaries.

Uses torch.load to handle PyTorch's zip-based checkpoint format.
Writes F32 raw binary files organized by layer.
"""

import argparse, os, sys
import torch
import numpy as np

# Mapping from state_dict key suffixes to output filenames
KEY_MAP = {
    'attention.wq.weight':           'wq',
    'attention.wkv.weight':          'wkv',
    'attention.wo.weight':           'wo',
    'attention.gater.weight':        'gater',
    'attention.temp':                'temp',
    'attention_norm.weight':         'attn_norm',
    'ffn_norm.weight':               'ffn_norm',
    'feed_forward.w_in.weight':      'ffn_win',
    'feed_forward.w_out.weight':     'ffn_wout',
    'feed_forward.router.down_proj.weight':   'router_down',
    'feed_forward.router.down_proj.bias':     'router_down_b',
    'feed_forward.router.router_mlp.0.weight': 'router_mlp0',
    'feed_forward.router.router_mlp.0.bias':   'router_mlp0_b',
    'feed_forward.router.router_mlp.2.weight': 'router_mlp2',
    'feed_forward.router.router_mlp.2.bias':   'router_mlp2_b',
    'feed_forward.router.router_mlp.4.weight': 'router_mlp4',
    'feed_forward.router.rmsnorm_eda.weight':  'router_eda',
    'feed_forward.router.router_states_scale':  'router_scale',
    'feed_forward.router.balancing_biases':     'router_bias',
    'feed_forward.experts.w13':      'experts_w13',
    'feed_forward.experts.w2':       'experts_w2',
}

ROOT_MAP = {
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
    'speaker_lda_projection.weight': 'speaker_lda',
    'speaker_lda_projection.bias':   'speaker_lda_b',
    'speaker_projection.weight':     'speaker_proj',
    'speaker_projection.bias':       'speaker_proj_b',
    'out_norm.weight':               'out_norm',
    'multi_output.weight':           'output',
}


def extract_weights(model_pth_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {model_pth_path} with torch.load...")
    print(f"File size: {os.path.getsize(model_pth_path) / 1e9:.2f} GB")

    # Load with torch.load — handles the zip format correctly
    state_dict = torch.load(model_pth_path, map_location='cpu', weights_only=False)

    print(f"Loaded {len(state_dict)} keys")

    # Create layer directories
    for layer_num in range(28):
        os.makedirs(os.path.join(output_dir, f'layer_{layer_num:02d}'), exist_ok=True)

    extracted = 0
    skipped = 0

    for key, val in sorted(state_dict.items()):
        # Convert to float32 numpy
        if isinstance(val, torch.Tensor):
            arr = val.detach().cpu().to(torch.float32).numpy()
        elif isinstance(val, np.ndarray):
            arr = val.astype(np.float32)
        else:
            print(f"  SKIP: {key} is {type(val).__name__}")
            skipped += 1
            continue

        # Determine output path
        if key.startswith('layers.'):
            parts = key.split('.')
            layer_num = int(parts[1])
            rest = '.'.join(parts[2:])

            if rest in KEY_MAP:
                fname = KEY_MAP[rest] + '.bin'
            else:
                fname = rest.replace('.', '_') + '.bin'

            out_path = os.path.join(output_dir, f'layer_{layer_num:02d}', fname)
        elif key in ROOT_MAP:
            out_path = os.path.join(output_dir, ROOT_MAP[key] + '.bin')
        else:
            fname = key.replace('.', '_') + '.bin'
            out_path = os.path.join(output_dir, fname)

        arr.tofile(out_path)
        extracted += 1
        if extracted % 50 == 0:
            print(f"  [{extracted}] {key}: {arr.shape}")

    # Save config
    config = np.array([
        28,     # n_layers
        2048,   # dim
        128,    # head_dim
        16,     # n_heads
        4,      # n_kv_heads
        3072,   # ffn_dim
        9,      # n_codebooks
        1024,   # codebook_size
        1026,   # audio_vocab
        519,    # text_vocab
        1024,   # eoa_id
        1025,   # audio_pad_id
    ], dtype=np.int32)
    config.tofile(os.path.join(output_dir, 'config.bin'))

    print(f"\nExtracted {extracted} tensors (skipped {skipped} non-tensor) to {output_dir}/")
    print(f"Model: 28 layers, 2048-dim, 9 codebooks, 1024-token vocab")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Path to model.pth')
    parser.add_argument('output_dir', nargs='?', default='./weights')
    args = parser.parse_args()
    extract_weights(args.model_path, args.output_dir)
