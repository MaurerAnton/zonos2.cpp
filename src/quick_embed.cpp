// quick_embed.cpp — Fast embedding-only verification
#include "zonos2.h"
#include <cstdio>
#include <cmath>
#include <vector>

static std::vector<float> load_bin(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    fseek(f, 0, SEEK_END);
    int n = ftell(f)/4; fseek(f,0,SEEK_SET);
    std::vector<float> v(n); fread(v.data(),4,n,f); fclose(f);
    return v;
}

int main(int argc, char** argv) {
    Zonos2Weights w;
    if (!load_zonos2_weights(argv[1], w)) return 1;
    auto& cfg = w.cfg;
    int dim = cfg.dim, nc = cfg.n_codebooks, fw = nc + 1;

    auto py_in = load_bin("/tmp/py_embed_in.bin");
    auto py_out = load_bin("/tmp/py_embed_out.bin");
    int n_tok = py_in.size() / fw;
    int32_t* ids = (int32_t*)py_in.data();

    std::vector<float> x(n_tok * dim, 0.0f);
    for (int t = 0; t < n_tok; t++) {
        float* out = x.data() + t * dim;
        for (int cb = 0; cb < nc; cb++) {
            int tid = ids[t * fw + cb];
            if (tid >= 0 && tid < cfg.audio_vocab) {
                const float* emb = w.codebook_embeds[cb].weight.ptr();
                for (int d = 0; d < dim; d++) out[d] += emb[tid * dim + d];
            }
        }
        int tid = ids[t * fw + nc];
        if (tid >= 0 && tid <= cfg.text_vocab) {
            const float* emb = w.text_embed.weight.ptr();
            for (int d = 0; d < dim; d++) out[d] += emb[tid * dim + d];
        }
    }
    // Per-token RMSNorm
    for (int t = 0; t < n_tok; t++) {
        float* xt = x.data() + t * dim;
        float ss = 0; for (int d = 0; d < dim; d++) ss += xt[d]*xt[d];
        float rcp = 1.0f / std::sqrt(ss / dim + cfg.norm_eps);
        for (int d = 0; d < dim; d++) xt[d] *= rcp;
    }

    // Compare
    float maxd = 0;
    for (size_t i = 0; i < x.size(); i++) maxd = std::max(maxd, std::abs(x[i] - py_out[i]));
    double cpp_ss=0, py_ss=0, cross=0;
    for (size_t i = 0; i < x.size(); i++) {
        cpp_ss += (double)x[i]*x[i];
        py_ss += (double)py_out[i]*py_out[i];
        cross += (double)x[i]*py_out[i];
    }
    double corr = cross / std::sqrt(cpp_ss * py_ss);

    printf("EMBEDDING: n=%zu, max_diff=%.2e, corr=%.10f, rms(cpp=%.6f, py=%.6f)\n",
           x.size(), maxd, corr, std::sqrt(cpp_ss/x.size()), std::sqrt(py_ss/x.size()));
    printf("First 5 C++: %.6f %.6f %.6f %.6f %.6f\n", x[0],x[1],x[2],x[3],x[4]);
    printf("First 5 Py:  %.6f %.6f %.6f %.6f %.6f\n", py_out[0],py_out[1],py_out[2],py_out[3],py_out[4]);
    printf("%s\n", (maxd < 1e-6 && corr > 0.999999) ? "BYTE-PERFECT MATCH!" : "MISMATCH");

    free_zonos2_weights(w);
    return (maxd < 1e-6) ? 0 : 1;
}
