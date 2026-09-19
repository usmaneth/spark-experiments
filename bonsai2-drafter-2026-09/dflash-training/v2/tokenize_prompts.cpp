// tokenize_prompts.cpp
// Offline tokenizer for prompt files. It loads only the vocabulary of the GGUF (no weights,
// no GPU) and tokenizes each text the same way the llama-server /tokenize endpoint does
// (add_special=true, parse_special=true). build_prompts_broad2.py uses it to enforce the
// 768-token limit of gen_client.py and to dedupe against token-id records.
//
// Input : a binary file of records: u32 byte_length | utf-8 text (the full chat-templated prompt).
// Output: one text line per record: "<n_tokens> <id_0> <id_1> ...". A failed record prints "-1".
//
// Usage: tokenize_prompts <model.gguf> <in.bin> <out.txt>
// Build (same flags as extract_feats_e3, only libllama is needed):
//   cd /home/usman/Bonsai-demo && g++ -O2 -std=c++17 \
//     -I llama.cpp/include -I llama.cpp/ggml/include -I llama.cpp/src \
//     dflash-training/v2/tokenize_prompts.cpp -o dflash-training/v2/tokenize_prompts \
//     -L bin/cuda -lllama -lggml-base \
//     -Wl,-rpath,/home/usman/Bonsai-demo/bin/cuda:/home/usman/Bonsai-demo/llama.cpp/build-cuda/bin

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

static void quiet_log(enum ggml_log_level level, const char * text, void * user) {
    (void) user;
    if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: tokenize_prompts <model.gguf> <in.bin> <out.txt>\n");
        return 2;
    }
    const char * model_path = argv[1];
    const char * in_path    = argv[2];
    const char * out_path   = argv[3];

    llama_log_set(quiet_log, nullptr);
    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "ERROR: model load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    FILE * fin = fopen(in_path, "rb");
    if (!fin) { fprintf(stderr, "ERROR: cannot open %s\n", in_path); return 1; }
    FILE * fout = fopen(out_path, "w");
    if (!fout) { fprintf(stderr, "ERROR: cannot open %s\n", out_path); return 1; }

    std::vector<char> text;
    std::vector<llama_token> toks(16384);
    size_t n_rec = 0, n_fail = 0;
    while (true) {
        uint32_t len = 0;
        if (fread(&len, 4, 1, fin) != 1) break;
        text.resize(len);
        if (len > 0 && fread(text.data(), 1, len, fin) != len) { fprintf(stderr, "ERROR: truncated input\n"); break; }
        int32_t n = llama_tokenize(vocab, text.data(), (int32_t) len, toks.data(), (int32_t) toks.size(), true, true);
        if (n < 0) {
            // buffer too small: -n is the required size
            toks.resize((size_t) (-n) + 16);
            n = llama_tokenize(vocab, text.data(), (int32_t) len, toks.data(), (int32_t) toks.size(), true, true);
        }
        if (n < 0) { fputs("-1\n", fout); n_fail++; n_rec++; continue; }
        fprintf(fout, "%d", n);
        for (int32_t i = 0; i < n; ++i) fprintf(fout, " %d", toks[i]);
        fputc('\n', fout);
        n_rec++;
    }
    fclose(fin);
    fclose(fout);
    fprintf(stderr, "tokenized %zu records (%zu failed) -> %s\n", n_rec, n_fail, out_path);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
