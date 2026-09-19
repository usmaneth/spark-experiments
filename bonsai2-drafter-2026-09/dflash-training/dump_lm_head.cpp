#include "llama.h"
#include "ggml.h"
#include "ggml-quants.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        printf("Usage: dump_lm_head <model.gguf> <output.bin>\n");
        return 1;
    }

    const char * model_path = argv[1];
    const char * out_path   = argv[2];

    printf("Loading model %s to locate output.weight...\n", model_path);
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU is fine for dequantization dump
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    const int n_embd  = llama_model_n_embd(model);
    const int n_vocab = llama_model_n_vocab(model);
    printf("Model specs: n_embd = %d, n_vocab = %d\n", n_embd, n_vocab);

    // Look for output.weight in gguf
    FILE * fout = fopen(out_path, "wb");
    if (!fout) {
        fprintf(stderr, "Failed to open output file %s\n", out_path);
        llama_model_free(model);
        return 1;
    }

    // Write header: n_embd, n_vocab
    uint32_t header[2] = {(uint32_t) n_embd, (uint32_t) n_vocab};
    fwrite(header, sizeof(uint32_t), 2, fout);

    printf("Dequantizing output.weight (%d x %d)...\n", n_vocab, n_embd);
    
    // We dequantize row-by-row using ggml_get_type_traits
    // Or we can evaluate an empty forward pass or get tensor from gguf
    fclose(fout);
    llama_model_free(model);
    printf("Header written to %s\n", out_path);
    return 0;
}
