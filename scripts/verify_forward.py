#!/usr/bin/env python3.12
"""verify_forward.py — Verify C++ forward pass against Python reference.
Uses extracted F32 weights from /tmp/weights2.
Computes embedding + RMSNorm + first transformer layer.
"""
import numpy as np
import os

WEIGHTS_DIR = '/tmp/weights2'

def load_tensor(path):
    return np.fromfile(path, dtype=np.float32)

def rms_norm(x, weight, eps=1e-5):
    """Per-token RMSNorm like PyTorch F.rms_norm(x, (dim,), eps)."""
    # x: [dim, n_tokens] — but stored as [n_tokens * dim]
    dim = len(weight)
    n_tokens = len(x) // dim
    x2d = x.reshape(n_tokens, dim)
    rms = np.sqrt(np.mean(x2d ** 2, axis=1, keepdims=True) + eps)
    out = (x2d / rms) * weight
    return out.ravel()

def linear(x, weight, bias=None):
    """Linear: weight [out_features, in_features], x [in_features, n_tokens].
    Returns [out_features, n_tokens].
    """
    in_f = weight.shape[1]
    out_f = weight.shape[0]
    n_tokens = len(x) // in_f
    x2d = x.reshape(in_f, n_tokens, order='F') if x.shape[0] == in_f else x.reshape(n_tokens, in_f)
    if x2d.shape[1] != in_f:
        x2d = x2d.T
    result = weight @ x2d.T
    if bias is not None:
        result += bias[:, np.newaxis]
    return result  # [out_features, n_tokens] but row-major

def test_embedding():
    """Test multi-embedding + emb_norm."""
    cfg = {
        'dim': 2048, 'n_codebooks': 9, 'audio_vocab': 1026, 'text_vocab': 519,
        'audio_pad_id': 1025, 'norm_eps': 1e-5
    }
    dim = cfg['dim']
    n_codebooks = cfg['n_codebooks']
    audio_vocab = cfg['audio_vocab']
    audio_pad_id = cfg['audio_pad_id']
    frame_width = n_codebooks + 1

    # Load codebook embeddings — PyTorch layout (num_emb, dim) in file
    cb_embeds = []
    for i in range(n_codebooks):
        w = load_tensor(os.path.join(WEIGHTS_DIR, f'codebook_{i}.bin'))
        cb_embeds.append(w.reshape(audio_vocab, dim))  # [num_emb, dim] — row tid = embedding
    
    # Load text embedding
    text_w = load_tensor(os.path.join(WEIGHTS_DIR, 'text_embed.bin'))
    text_emb = text_w.reshape(cfg['text_vocab'] + 1, dim)  # [520, 2048]

    # Test input: silence frame + 'H' frame
    input_ids = np.array([
        [audio_pad_id]*9 + [cfg['text_vocab']],  # silence: pad=519 for text
        [audio_pad_id]*9 + [72],                   # 'H' = byte 72
    ], dtype=np.int32)  # [2, 10]

    # Embedding sum: x[t, d] = sum_i embed_i[input_ids[t,i], d]
    n_tokens = len(input_ids)
    x = np.zeros((n_tokens, dim), dtype=np.float32)  # [n_tokens, dim] token-major
    for t in range(n_tokens):
        for cb in range(n_codebooks):
            tid = input_ids[t, cb]
            if 0 <= tid < audio_vocab:
                x[t] += cb_embeds[cb][tid]  # row tid = embedding
        tid = input_ids[t, n_codebooks]
        if 0 <= tid <= cfg['text_vocab']:
            x[t] += text_emb[tid]

    # emb_norm: elementwise_affine=False → no weight, just RMSNorm per token
    # x is already [n_tokens, dim]
    rms = np.sqrt(np.mean(x ** 2, axis=1, keepdims=True) + cfg['norm_eps'])
    x_normed = x / rms  # [n_tokens, dim] — token-major
    x_flat = x_normed.ravel()  # [tok0_d0, tok0_d1, ..., tok0_d2047, tok1_d0, ...]

    print(f"Embedding output: shape=(dim={dim}, n_tokens={n_tokens})")
    print(f"  RMS: {np.sqrt(np.mean(x_flat**2)):.6f}")
    print(f"  First 10 values: {x_flat[:10]}")
    x_flat.astype(np.float32).tofile('/tmp/py_embed_out.bin')
    input_ids.astype(np.int32).tofile('/tmp/py_embed_in.bin')
    print("Saved to /tmp/py_embed_*.bin")


def test_attention_layer0():
    """Test first attention layer."""
    cfg = {
        'dim': 2048, 'head_dim': 128, 'n_heads': 16, 'n_kv_heads': 4,
        'rope_theta': 10000.0, 'norm_eps': 1e-5
    }
    dim = cfg['dim']
    head_dim = cfg['head_dim']
    n_heads = cfg['n_heads']
    n_kv = cfg['n_kv_heads']
    kv_dim = n_kv * head_dim

    layer_dir = os.path.join(WEIGHTS_DIR, 'layer_00')

    # Load embedding output as input
    if not os.path.exists('/tmp/py_embed_out.bin'):
        print("Run test_embedding first!")
        return

    x_flat = load_tensor('/tmp/py_embed_out.bin')
    n_tokens = len(x_flat) // dim
    x = x_flat.reshape(dim, n_tokens)  # [dim, n_tokens] — C++ layout

    # Load weights
    attn_norm_w = load_tensor(os.path.join(layer_dir, 'attn_norm.bin'))
    wq_w = load_tensor(os.path.join(layer_dir, 'wq.bin')).reshape(dim, dim)   # [in_f, out_f] in file
    wkv_w = load_tensor(os.path.join(layer_dir, 'wkv.bin')).reshape(kv_dim*2, dim)
    wo_w = load_tensor(os.path.join(layer_dir, 'wo.bin')).reshape(dim, dim)
    gater_w = load_tensor(os.path.join(layer_dir, 'gater.bin')).reshape(n_heads, dim)
    temp = load_tensor(os.path.join(layer_dir, 'temp.bin'))

    # attention_norm
    x_normed = x.T  # [n_tokens, dim]
    rms = np.sqrt(np.mean(x_normed**2, axis=1, keepdims=True) + cfg['norm_eps'])
    x_in = ((x_normed / rms) * attn_norm_w).T  # [dim, n_tokens]
    residual = x_in.copy()

    # Q projection
    q = (wq_w @ x_in.T).T  # wq_w: [out_f, in_f] = [dim, dim] in file, so wq_w @ x_in
    # But wait: wq_w is stored as [dim, dim] = [in_features, out_features] in C++ layout
    # PyTorch weight is [out_features, in_features] = [dim, dim]
    # File has [dim, dim] which in C++ is interpreted as ne[0] = dim, ne[1] = dim
    # The bytes are the same. To do linear: result[o] = sum_i x[i] * w[o,i]
    # File has PyTorch layout: w[o,i] at offset o*dim + i
    # We loaded as reshape(dim, dim) which is row-major: w_2d[o,i] = file[o*dim + i]
    # w @ x.T where @ is matrix multiply: (dim,dim) @ (dim,n_tokens) → (dim,n_tokens)
    q_out = wq_w @ x_in  # [dim, n_tokens]

    print(f"\nAttention layer 0:")
    print(f"  attn_norm output RMS: {np.sqrt(np.mean(x_in**2)):.6f}")
    print(f"  Q output RMS: {np.sqrt(np.mean(q_out**2)):.6f}")
    print(f"  Q[:5]: {q_out.ravel()[:5]}")

    # Save for C++ comparison
    x_in.astype(np.float32).tofile('/tmp/py_attn0_in.bin')
    print("Saved to /tmp/py_attn0_in.bin")


if __name__ == '__main__':
    test_embedding()
    test_attention_layer0()
