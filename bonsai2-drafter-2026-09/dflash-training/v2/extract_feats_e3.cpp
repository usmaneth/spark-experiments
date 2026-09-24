// extract_feats_e3.cpp
// Compact feature extractor for the EAGLE-3 drafter (BON3 format).
// This is a copy of extract_feats_tf.cpp with two changes:
//   1. It keeps only the three taps that the EAGLE-3 trainer reads: layer inputs [6, 34, 62].
//   2. It writes the taps and final_hidden as f16 (ggml round-to-nearest) instead of f32.
// The teacher-forced pass, the loss_mask rule and the CLI are identical to extract_feats_tf.
//
// Input : jsonl of {"tokens":[...ids...], "n_prompt":N} (produced by gen_client.py).
// Output: header "BON3"(4) | embd u32 | n_taps u32=3 | tap_layers[3] u32={6,34,62}
//   per sample: n u32 | tokens i32[n] | loss_mask u8[n] | taps f16[n*3*embd] | final_hidden f16[n*embd]
// A sample takes n*(5 + 4*embd*2) bytes: 40% of the BON2 size.
//
// Usage: extract_feats_e3 <model.gguf> <gen.jsonl> <out.bin> [max_samples]
// Build (node_a or node_b, CPU compile, links against bin/cuda):
//   cd /home/REDACTED/Bonsai-demo && g++ -O2 -std=c++17 \
//     -I llama.cpp/include -I llama.cpp/ggml/include -I llama.cpp/src \
//     dflash-training/v2/extract_feats_e3.cpp -o dflash-training/v2/extract_feats_e3 \
//     -L bin/cuda -lllama -lggml-base \
//     -Wl,-rpath,/home/REDACTED/Bonsai-demo/bin/cuda:/home/REDACTED/Bonsai-demo/llama.cpp/build-cuda/bin
// Run:  LD_LIBRARY_PATH=/home/REDACTED/Bonsai-demo/bin/cuda ./extract_feats_e3 <model> <gen.jsonl> <out.bin> 100000

#include "llama.h"
#include "llama-ext.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <string>
#include <vector>
#include <fstream>

// parse {"tokens":[...], "n_prompt":N}; returns false on parse failure
static bool parse_line(const std::string & s, std::vector<int32_t> & toks, int & n_prompt) {
    toks.clear(); n_prompt = -1;
    size_t tp = s.find("\"tokens\"");
    if (tp == std::string::npos) return false;
    size_t lb = s.find('[', tp);
    if (lb == std::string::npos) return false;
    size_t rb = s.find(']', lb);
    if (rb == std::string::npos) return false;
    size_t i = lb + 1;
    while (i < rb) {
        while (i < rb && (s[i]==' '||s[i]==',')) ++i;
        if (i >= rb) break;
        long v = strtol(s.c_str()+i, nullptr, 10);
        toks.push_back((int32_t)v);
        while (i < rb && s[i] != ',') ++i;
    }
    size_t np = s.find("\"n_prompt\"");
    if (np == std::string::npos) return false;
    size_t c = s.find(':', np);
    if (c == std::string::npos) return false;
    n_prompt = (int) strtol(s.c_str()+c+1, nullptr, 10);
    return !toks.empty() && n_prompt >= 0 && n_prompt <= (int)toks.size();
}

int main(int argc, char ** argv) {
    const char * model_path = (argc>1)?argv[1] : "/home/REDACTED/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf";
    const char * data_path  = (argc>2)?argv[2] : "/home/REDACTED/Bonsai-demo/dflash-training/v2/prompts_gen_batch2.jsonl";
    const char * out_path   = (argc>3)?argv[3] : "/home/REDACTED/Bonsai-demo/dflash-training/v2/feats_e3/batch2.bin";
    const int max_samples   = (argc>4)?atoi(argv[4]) : 100000;

    // EAGLE-3 reads the layer inputs of layers 6, 34 and 62 (BON2 tap indices 0, 2, 4).
    const uint32_t tap_layers[3] = {6,34,62};
    const int n_taps = 3;

    printf("=== Bonsai 2 EAGLE-3 feature extractor (BON3, 3 taps, f16) ===\n");
    printf("model:%s\ndata:%s\nout:%s\n", model_path, data_path, out_path);

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model){ fprintf(stderr,"ERROR: model load failed\n"); return 1; }
    const int n_embd = llama_model_n_embd(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx=2048; cparams.n_batch=2048; cparams.n_ubatch=2048;
    cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx){ fprintf(stderr,"ERROR: ctx create failed\n"); return 1; }

    // features-only: embeddings on + taps on for the whole run (no toggling)
    llama_set_embeddings(ctx, true);
    for (int l=0;l<n_taps;++l) llama_set_embeddings_layer_inp(ctx, tap_layers[l], true);
    llama_memory_t mem = llama_get_memory(ctx);

    FILE * fout = fopen(out_path,"wb");
    if (!fout){ fprintf(stderr,"ERROR: cannot open %s\n",out_path); return 1; }
    const char magic[4]={'B','O','N','3'};
    uint32_t embd_u=(uint32_t)n_embd, taps_u=(uint32_t)n_taps;
    fwrite(magic,1,4,fout); fwrite(&embd_u,4,1,fout); fwrite(&taps_u,4,1,fout); fwrite(tap_layers,4,3,fout);

    std::ifstream fin(data_path);
    if (!fin.is_open()){ fprintf(stderr,"ERROR: cannot open %s\n",data_path); return 1; }

    std::string line;
    int n_done=0, n_skip=0; size_t total_tok=0, total_gen=0;
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC,&t0);
    std::vector<int32_t> toks; std::vector<float> taps, final_hidden;
    std::vector<ggml_fp16_t> taps_h, final_hidden_h;

    while (std::getline(fin,line) && n_done<max_samples) {
        if (line.empty()) continue;
        int n_prompt=-1;
        if (!parse_line(line, toks, n_prompt)) { n_skip++; continue; }
        const int n=(int)toks.size();
        if (n<2 || n>cparams.n_ctx) { n_skip++; continue; }

        std::vector<uint8_t> loss_mask(n,0);
        for (int i=n_prompt;i<n;++i) loss_mask[i]=1;

        llama_memory_clear(mem, true);
        llama_batch fb = llama_batch_init(n,0,1);
        for (int i=0;i<n;++i){ fb.token[i]=toks[i]; fb.pos[i]=i; fb.n_seq_id[i]=1; fb.seq_id[i][0]=0; fb.logits[i]=true; }
        fb.n_tokens=n;
        if (llama_decode(ctx, fb)!=0){ llama_batch_free(fb); n_skip++; continue; }
        llama_batch_free(fb);

        taps.assign((size_t)n*n_taps*n_embd, 0.0f);
        bool ok=true;
        for (int l=0;l<n_taps;++l){ float * e=llama_get_embeddings_layer_inp(ctx, tap_layers[l]);
            if(!e){ok=false;break;}
            for (int t=0;t<n;++t) memcpy(&taps[((size_t)t*n_taps+l)*n_embd], &e[(size_t)t*n_embd], n_embd*sizeof(float)); }
        if(!ok){ n_skip++; continue; }
        final_hidden.assign((size_t)n*n_embd, 0.0f);
        for (int t=0;t<n;++t){ const float * em=llama_get_embeddings_ith(ctx,t); if(!em){ok=false;break;}
            memcpy(&final_hidden[(size_t)t*n_embd], em, n_embd*sizeof(float)); }
        if(!ok){ n_skip++; continue; }

        // f32 -> f16 with the ggml conversion (round to nearest even)
        taps_h.resize(taps.size());
        final_hidden_h.resize(final_hidden.size());
        ggml_fp32_to_fp16_row(taps.data(), taps_h.data(), (int64_t)taps.size());
        ggml_fp32_to_fp16_row(final_hidden.data(), final_hidden_h.data(), (int64_t)final_hidden.size());

        uint32_t n_u=(uint32_t)n;
        fwrite(&n_u,4,1,fout);
        fwrite(toks.data(),sizeof(int32_t),n,fout);
        fwrite(loss_mask.data(),1,n,fout);
        fwrite(taps_h.data(),sizeof(ggml_fp16_t),taps_h.size(),fout);
        fwrite(final_hidden_h.data(),sizeof(ggml_fp16_t),final_hidden_h.size(),fout);
        n_done++; total_tok+=n; total_gen+=(n-n_prompt);
        if (n_done%50==0){ struct timespec t1; clock_gettime(CLOCK_MONOTONIC,&t1);
            double el=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
            printf("  feat %d (skip %d) | tok %zu gen %zu | %.1f tok/s | %.1fs\n",
                   n_done,n_skip,total_tok,total_gen,total_tok/el,el); fflush(stdout); fflush(fout); }
    }
    fclose(fout);
    struct timespec t1; clock_gettime(CLOCK_MONOTONIC,&t1);
    double el=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
    printf("\n=== done === samples=%d skip=%d tokens=%zu gen=%zu time=%.1fs (%.1f tok/s)\nout:%s\n",
           n_done,n_skip,total_tok,total_gen,el,total_tok/el,out_path);
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
