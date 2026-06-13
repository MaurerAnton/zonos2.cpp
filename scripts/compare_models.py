#!/usr/bin/env python3
"""compare_models.py — Compare C++ and Python Zonos2 outputs for byte-perfect matching.

Usage:
  python3 compare_models.py --text "Hello world" --cpp-output codes.bin --python-output py_codes.bin

Dumps audio tokens from both implementations and compares them.
"""

import sys, struct, argparse, os
import numpy as np

def load_codes(path):
    """Load codes from C++ output format: [n_frames, n_codebooks] int32."""
    with open(path, 'rb') as f:
        n_frames = struct.unpack('i', f.read(4))[0]
        n_codebooks = struct.unpack('i', f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.int32)
        return data.reshape(n_frames, n_codebooks)

def run_python_reference(text, model_path, output_path, seed=42):
    """Run Zonos2 Python inference to get reference audio tokens."""
    # Import after setting up
    import torch
    sys.path.insert(0, os.path.dirname(model_path))
    
    from minisgl.message import TTSSamplingParams
    from minisgl.tts import TTSLLM
    
    tts = TTSLLM(model_path=model_path, decode_audio=False)
    
    results = tts.generate(
        [text],
        TTSSamplingParams(
            seed=seed,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            max_tokens=100,
        ),
    )
    
    result = results[0]
    audio_tokens = result['audio_tokens']  # list of lists
    eos_frame = result.get('eos_frame', len(audio_tokens))
    
    # Save as same format as C++ output
    n_frames = len(audio_tokens)
    n_codebooks = len(audio_tokens[0]) if n_frames > 0 else 9
    
    with open(output_path, 'wb') as f:
        f.write(struct.pack('i', n_frames))
        f.write(struct.pack('i', n_codebooks))
        for frame in audio_tokens:
            for tok in frame:
                f.write(struct.pack('i', int(tok)))
    
    print(f"Python generated {n_frames} frames, EOS at frame {eos_frame}")
    return audio_tokens

def run_cpp_reference(text, weights_dir, output_path, exe_path, seed=42):
    """Run C++ inference."""
    import subprocess
    cmd = [
        exe_path,
        '--model-dir', weights_dir,
        '--output', output_path,
        '--seed', str(seed),
        '--max-tokens', '100',
        text
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("C++ ERROR:", result.stderr)
        return None
    
    codes_path = output_path + '.codes.bin'
    if os.path.exists(codes_path):
        return load_codes(codes_path)
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default='Hello world')
    parser.add_argument('--model-path', help='Path to model.pth or repo dir')
    parser.add_argument('--weights-dir', help='Path to extracted weights dir for C++')
    parser.add_argument('--cpp-exe', default='./build/zonos2_cli')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--python-only', action='store_true')
    parser.add_argument('--cpp-only', action='store_true')
    args = parser.parse_args()

    py_codes = None
    cpp_codes = None

    if not args.cpp_only and args.model_path:
        out = '/tmp/py_codes.bin'
        py_codes = run_python_reference(args.text, args.model_path, out, args.seed)
        py_loaded = load_codes(out)
        print(f"Python codes: {py_loaded.shape}")

    if not args.python_only and args.weights_dir:
        out = '/tmp/cpp_codes'
        cpp_codes = run_cpp_reference(args.text, args.weights_dir, out, args.cpp_exe, args.seed)
        if cpp_codes is not None:
            print(f"C++ codes: {cpp_codes.shape}")

    # Compare
    if py_codes is not None and cpp_codes is not None:
        py = load_codes('/tmp/py_codes.bin')
        cpp = load_codes('/tmp/cpp_codes.codes.bin')
        
        min_frames = min(py.shape[0], cpp.shape[0])
        matches = 0
        total = 0
        for f in range(min_frames):
            for c in range(min(py.shape[1], cpp.shape[1])):
                total += 1
                if py[f, c] == cpp[f, c]:
                    matches += 1

        print(f"\nComparison ({min_frames} frames):")
        print(f"  Matching tokens: {matches}/{total} ({100*matches/total:.1f}%)")
        print(f"  Frames: Python={py.shape[0]}, C++={cpp.shape[0]}")
        
        if matches == total:
            print("  ✓ BYTE-PERFECT MATCH!")
        else:
            print("  ✗ MISMATCH")
            # Show first difference
            for f in range(min_frames):
                for c in range(min(py.shape[1], cpp.shape[1])):
                    if py[f, c] != cpp[f, c]:
                        print(f"  First diff: frame {f}, codebook {c}: py={py[f,c]}, cpp={cpp[f,c]}")
                        break
                else:
                    continue
                break

if __name__ == '__main__':
    main()
