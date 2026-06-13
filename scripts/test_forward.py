#!/usr/bin/env python3.12
"""test_forward.py — Test C++ forward pass against Python reference.
Dumps intermediate activations for comparison.
"""

import sys, struct, os
import torch
import numpy as np

# Add repo to path
sys.path.insert(0, '/home/bym/zonos2/repo/python')

def test_embedding():
    """Test embedding layer in isolation."""
    from zonos2.models.config import ModelConfig
    from zonos2.models.zonos2 import MultiEmbedding
    from zonos2.models.config import RotaryConfig

    # Build minimal config matching params.json
    class Zonos2Config:
        n_layers = 28
        dim = 2048
        head_dim = 128
        n_heads = 16
        n_kv_heads = 4
        ffn_dim_multiplier = 1.5
        multiple_of = 256
        norm_eps = 1e-5
        rope_theta = 10000.0
        max_seqlen = 6144
        n_codebooks = 9
        codebook_size = 1024
        eoa_id = 1024
        audio_pad_id = 1025
        text_vocab = 519
        moe_n_experts = 16
        moe_router_topk = 1
        special_topk_layers = {26: 2}
        moe_router_dim = 128
        moe_start_from_layer = 3
        moe_end_from_layer = 1
        moe_impl = "sonic"
        moe_balancing_strategy = "legacy"
        speaker_enabled = True
        speaker_embedding_dim = 2048
        speaker_lda_dim = 1024
        speaker_background_token_enabled = True
        accurate_mode_token_enabled = True
        speaking_rate_num_buckets = 8
        quality_num_buckets = 60

    config = ModelConfig.from_zonos2_config(Zonos2Config())

    # Load state dict
    model_path = '/tmp/model.pht'
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)

    # Test embedding
    emb = MultiEmbedding(config)
    emb.load_state_dict({k: v for k, v in state_dict.items() if k.startswith('multi_embedder.')})

    # Create a test input: 2 frames with pad tokens and text token 72 ('H')
    codes = torch.tensor([
        [1025, 1025, 1025, 1025, 1025, 1025, 1025, 1025, 1025, 519],  # silence
        [1025, 1025, 1025, 1025, 1025, 1025, 1025, 1025, 1025, 72],   # 'H'
    ], dtype=torch.int64)

    result = emb.forward(codes)
    print(f"Embedding output shape: {result.shape}")
    print(f"Embedding output RMS: {result.norm().item():.6f}")
    print(f"Embedding output[:5]: {result[0,:5].tolist()}")

    # Save for C++ comparison
    result.float().numpy().tofile('/tmp/py_embed_out.bin')
    codes.numpy().astype(np.int32).tofile('/tmp/py_embed_in.bin')
    print("Saved embedding I/O to /tmp/py_embed_*.bin")


if __name__ == '__main__':
    test_embedding()
