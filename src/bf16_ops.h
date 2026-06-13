// bf16_ops.h — BF16 operations with AVX-512/AVX2 like llama.cpp
// Provides: bf16_t type, F32↔BF16 conversion, BF16 dot product
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <cpuid.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

// BF16 type
using bf16_t = uint16_t;

// ============================================================
// CPU feature detection at runtime
// ============================================================
static inline bool cpu_has_avx512_bf16() {
#ifdef __AVX512BF16__
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 1, &eax, &ebx, &ecx, &edx)) {
        return (eax & (1 << 5)) != 0;  // AVX512_BF16 bit
    }
#endif
    return false;
}

static inline bool cpu_has_avx2() {
#ifdef __AVX2__
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        return (ecx & (1 << 28)) != 0;  // AVX bit → check AVX2 via cpuid 7
    }
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return (ebx & (1 << 5)) != 0;  // AVX2 bit
    }
#endif
    return false;
}

// ============================================================
// F32 ↔ BF16 conversion
// ============================================================
static inline bf16_t f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    return (bf16_t)(bits >> 16);
}

static inline float bf16_to_f32(bf16_t bf) {
    uint32_t bits = ((uint32_t)bf) << 16;
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

// ============================================================
// BF16 dot product: sum(weight[i] * x[i]) for i=0..n-1
// weight is BF16, x is F32, result is F32
// ============================================================

// Scalar fallback
static inline float bf16_dot_scalar(const bf16_t* weight, const float* x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += bf16_to_f32(weight[i]) * x[i];
    }
    return sum;
}

// AVX2: convert 16 BF16→F32, do F32 mul-add
static inline float bf16_dot_avx2(const bf16_t* weight, const float* x, int n) {
    float sum = 0.0f;
    int i = 0;

    // Process 8 elements at a time
    for (; i + 7 < n; i += 8) {
        // Load 8 BF16 weights (lower 128 bits of a 256-bit register)
        __m128i w16 = _mm_loadu_si128((const __m128i*)(weight + i));
        // Convert BF16 to F32: left-shift each 16-bit element by 16
        __m256i w32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(w16), 16);
        __m256 wf = _mm256_castsi256_ps(w32);

        // Load 8 F32 activations
        __m256 xf = _mm256_loadu_ps(x + i);

        // FMA: wf * xf
        __m256 prod = _mm256_mul_ps(wf, xf);

        // Horizontal sum
        __m128 lo = _mm256_castps256_ps128(prod);
        __m128 hi = _mm256_extractf128_ps(prod, 1);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum += _mm_cvtss_f32(sum128);
    }

    // Remainder
    for (; i < n; i++) {
        sum += bf16_to_f32(weight[i]) * x[i];
    }
    return sum;
}

#if HAS_AVX512_BF16
// AVX-512 BF16: uses VDPBF16PS instruction
static inline float bf16_dot_avx512(const bf16_t* weight, const float* x, int n) {
    __m512 acc = _mm512_setzero_ps();
    int i = 0;

    // VDPBF16PS: 32 BF16 × 32 BF16 → F32 accumulate
    // Input A: BF16 values (lower 16 bits of each 32-bit slot)
    // Input B: BF16 values (lower 16 bits of each 32-bit slot)
    // But our x is F32, so we need to convert x to BF16 first
    // Actually VDPBF16PS takes two BF16 inputs — x must be BF16 too
    // For activations that are F32, we must convert them to BF16

    // Process 32 elements at a time
    for (; i + 31 < n; i += 32) {
        // Load 32 BF16 weights
        __m256i w_bf16 = _mm256_loadu_si256((const __m256i*)(weight + i));
        // Convert 32 F32 activations to BF16 (need to round/truncate)
        __m256 x_f32 = _mm256_loadu_ps(x + i);
        // ... VDPBF16PS expects both inputs as BF16 in lower 16 bits
        // This gets complex — for now, fall through to AVX2
        break;
    }
    // Fallback to AVX2 for remainder
    return bf16_dot_avx2(weight, x, n);
}
#endif

// Dispatch to best available implementation
static inline float bf16_dot(const bf16_t* weight, const float* x, int n) {
    if (n < 8) return bf16_dot_scalar(weight, x, n);
#ifdef __AVX2__
    if (cpu_has_avx2()) return bf16_dot_avx2(weight, x, n);
#endif
    return bf16_dot_scalar(weight, x, n);
}

// ============================================================
// BF16 vector storage (replaces F32 weight arrays)
// ============================================================
struct Bf16Tensor {
    std::vector<bf16_t> data;
    int ne[4] = {0, 0, 0, 0};

    void from_f32(const float* src, int n) {
        data.resize(n);
        for (int i = 0; i < n; i++) {
            data[i] = f32_to_bf16(src[i]);
        }
    }
    float to_f32(int i) const { return bf16_to_f32(data[i]); }
    const bf16_t* ptr() const { return data.data(); }
    size_t nelem() const { return (size_t)ne[0] * ne[1] * ne[2] * ne[3]; }
};

// ============================================================
// BF16 linear layer
// Weight: [in_features, out_features] stored as BF16
// x: [in_features, n_tokens] stored as F32
// output: [out_features, n_tokens] stored as F32
// ============================================================
static inline void bf16_linear(
    const bf16_t* weight, int in_f, int out_f,
    const float* x, int n_tokens,
    float* output,
    const float* bias = nullptr
) {
    for (int t = 0; t < n_tokens; t++) {
        const float* xt = x + t * in_f;
        float* ot = output + t * out_f;
        for (int o = 0; o < out_f; o++) {
            float s = bias ? bias[o] : 0.0f;
            s += bf16_dot(weight + o * in_f, xt, in_f);
            ot[o] = s;
        }
    }
}
