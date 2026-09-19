#include "llama.h"
#include "llama-ext.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

struct TokenSample {
    std::vector<int32_t> tokens;
    std::vector<float> features; // [n_tokens, 5 * 5120]
};

int main(int argc, char ** argv) {
    const char * model_path = (argc > 1) ? argv[1] : "models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf";
    const char * data_path  = (argc > 2) ? argv[2] : "dflash-training/SpecForge/cache/dataset/codealpaca-20k_train.jsonl";
    const char * out_path   = (argc > 3) ? argv[3] : "dflash-training/bonsai2_features.bin";
    const int max_samples   = (argc > 4) ? std::atoi(argv[4]) : 1000;

    printf("=== Bonsai 2 Feature Extractor ===\n");
    printf("Model:   %s\n", model_path);
    printf("Data:    %s\n", data_path);
    printf("Output:  %s\n", out_path);
    printf("Samples: %d\n", max_samples);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99; // offload all to GB10 GPU

    printf("Loading model into GB10 VRAM...\n");
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Error: Failed to load model from %s\n", model_path);
        return 1;
    }

    const int n_embd = llama_model_n_embd(model);
    const int n_layers = llama_model_n_layer(model);
    printf("Model loaded: n_embd=%d, n_layers=%d\n", n_embd, n_layers);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 4096;
    cparams.n_batch = 4096;
    cparams.n_ubatch = 2048;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED; // Flash Attention on GB10

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Error: Failed to create llama_context\n");
        llama_model_free(model);
        return 1;
    }

    // Enable layer input embeddings for the 5 target layers
    const uint32_t target_layers[5] = {6, 20, 34, 48, 62};
    printf("Enabling layer input embeddings for layers [6, 20, 34, 48, 62]...\n");
    for (uint32_t lid : target_layers) {
        llama_set_embeddings_layer_inp(ctx, lid, true);
    }

    // Open output binary file
    FILE * fout = std::fopen(out_path, "wb");
    if (!fout) {
        fprintf(stderr, "Error: Failed to open %s for writing\n", out_path);
        return 1;
    }

    // Write header: magic, n_embd, n_layers_extracted
    const char magic[4] = {'B', 'O', 'N', 'S'};
    uint32_t n_taps = 5;
    uint32_t embd_dim = n_embd;
    std::fwrite(magic, 1, 4, fout);
    std::fwrite(&embd_dim, sizeof(uint32_t), 1, fout);
    std::fwrite(&n_taps, sizeof(uint32_t), 1, fout);
    std::fwrite(target_layers, sizeof(uint32_t), 5, fout);

    // Read jsonl lines
    std::ifstream fin(data_path);
    if (!fin.is_open()) {
        fprintf(stderr, "Error: Failed to open %s\n", data_path);
        return 1;
    }

    std::string line;
    int sample_count = 0;
    size_t total_tokens = 0;

    const llama_vocab * vocab = llama_model_get_vocab(model);

    printf("Starting extraction across %d samples...\n", max_samples);
    clock_t start_time = clock();

    while (std::getline(fin, line) && sample_count < max_samples) {
        if (line.empty()) continue;

        // Extract conversation roles and content
        std::string prompt_text;
        size_t pos = 0;
        while ((pos = line.find("\"role\":", pos)) != std::string::npos) {
            size_t role_val_start = line.find('"', pos + 7);
            if (role_val_start == std::string::npos) break;
            size_t role_val_end = line.find('"', role_val_start + 1);
            if (role_val_end == std::string::npos) break;
            std::string role = line.substr(role_val_start + 1, role_val_end - role_val_start - 1);
            
            size_t content_pos = line.find("\"content\":", role_val_end);
            if (content_pos == std::string::npos) break;
            size_t val_start = line.find('"', content_pos + 10);
            if (val_start == std::string::npos) break;
            size_t val_end = val_start + 1;
            while (val_end < line.size() && !(line[val_end] == '"' && line[val_end - 1] != '\\')) {
                val_end++;
            }
            std::string content = line.substr(val_start + 1, val_end - val_start - 1);
            
            prompt_text += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
            pos = val_end + 1;
        }

        if (prompt_text.empty()) continue;

        // Tokenize
        std::vector<llama_token> tokens(prompt_text.size() + 16);
        int n_tok = llama_tokenize(vocab, prompt_text.c_str(), prompt_text.size(), tokens.data(), tokens.size(), true, false);
        if (n_tok <= 4 || n_tok > 2048) continue; // skip too short or too long
        tokens.resize(n_tok);

        // Run forward pass
        llama_memory_clear(llama_get_memory(ctx), true);

        llama_batch batch = llama_batch_init(n_tok, 0, 1);
        for (int i = 0; i < n_tok; ++i) {
            batch.token[i]     = tokens[i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = false;
        }
        batch.n_tokens = n_tok;

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "Warning: decode failed on sample %d\n", sample_count);
            llama_batch_free(batch);
            continue;
        }
        llama_batch_free(batch);

        // Extract layer input features for each token
        // Write: n_tok (uint32), tokens[n_tok], features[n_tok * 5 * n_embd]
        uint32_t n_tokens_u32 = n_tok;
        std::fwrite(&n_tokens_u32, sizeof(uint32_t), 1, fout);
        std::fwrite(tokens.data(), sizeof(llama_token), n_tok, fout);

        // Concatenate the 5 layer states per token
        std::vector<float> token_features(n_tok * 5 * n_embd);
        for (int l = 0; l < 5; ++l) {
            float * embd = llama_get_embeddings_layer_inp(ctx, target_layers[l]);
            if (!embd) {
                fprintf(stderr, "Error: null embedding for layer %d\n", target_layers[l]);
                break;
            }
            for (int t = 0; t < n_tok; ++t) {
                std::memcpy(&token_features[(t * 5 + l) * n_embd], &embd[t * n_embd], n_embd * sizeof(float));
            }
        }
        std::fwrite(token_features.data(), sizeof(float), token_features.size(), fout);

        sample_count++;
        total_tokens += n_tok;

        if (sample_count % 100 == 0) {
            double elapsed = (double)(clock() - start_time) / CLOCKS_PER_SEC;
            double tok_per_sec = total_tokens / elapsed;
            printf("  Processed %d/%d samples (%zu tokens) | %.1f tok/s | %.1f s elapsed\n",
                   sample_count, max_samples, total_tokens, tok_per_sec, elapsed);
            std::fflush(fout);
        }
    }

    std::fclose(fout);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    double total_elapsed = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    printf("=== Extraction Complete ===\n");
    printf("Samples: %d | Tokens: %zu | Time: %.1f s (%.1f tok/s)\n",
           sample_count, total_tokens, total_elapsed, total_tokens / total_elapsed);
    printf("Saved to %s\n", out_path);

    return 0;
}
