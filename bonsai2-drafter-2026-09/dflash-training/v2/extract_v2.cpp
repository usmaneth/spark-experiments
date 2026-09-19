// extract_v2.cpp
// v2 Bonsai 2 feature extractor.
// Fixes the v1 JSON bug (v1 never unescaped content, so ~87% of code rows carried
// literal backslash-n). v2 JSON-decodes each message content (\n \t \r \" \\ \/ \uXXXX
// incl. surrogate pairs) before it builds the chat template.
// Self-distillation: for each prompt, generate a greedy (temp 0) completion FROM the
// model, then teacher-force (prompt + completion) to extract features.
// loss_mask = 1 on generated completion tokens, 0 on prompt tokens.
//
// Output (v2/feats/*.bin):
//   header: "BON2"(4) | embd u32 | n_taps u32=5 | tap_layers[5] u32 = {6,20,34,48,62}
//   per sample: n_tokens u32 | token_ids i32[n] | loss_mask u8[n]
//               | taps f32[n*5*embd]  (per token, 5 taps contiguous in layer order)
//               | final_hidden f32[n*embd]  (post-final-norm LM-head input == t_embd)

#include "llama.h"
#include "llama-ext.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <string>
#include <vector>
#include <fstream>

// --- JSON string decode ---------------------------------------------------
static void append_utf8(std::string & out, uint32_t cp) {
    if (cp <= 0x7F) { out += (char) cp; }
    else if (cp <= 0x7FF) {
        out += (char) (0xC0 | (cp >> 6));
        out += (char) (0x80 | (cp & 0x3F));
    } else if (cp <= 0xFFFF) {
        out += (char) (0xE0 | (cp >> 12));
        out += (char) (0x80 | ((cp >> 6) & 0x3F));
        out += (char) (0x80 | (cp & 0x3F));
    } else {
        out += (char) (0xF0 | (cp >> 18));
        out += (char) (0x80 | ((cp >> 12) & 0x3F));
        out += (char) (0x80 | ((cp >> 6) & 0x3F));
        out += (char) (0x80 | (cp & 0x3F));
    }
}

static int hex4(const std::string & s, size_t p) {
    int v = 0;
    for (int i = 0; i < 4; ++i) {
        char c = s[p + i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= c - '0';
        else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
        else return -1;
    }
    return v;
}

// Decode a JSON string starting at s[pos] == opening quote. Returns decoded content,
// sets pos to index just past the closing quote. Handles all JSON escapes.
static std::string json_read_string(const std::string & s, size_t & pos) {
    std::string out;
    if (pos >= s.size() || s[pos] != '"') return out;
    ++pos;
    while (pos < s.size()) {
        char c = s[pos];
        if (c == '"') { ++pos; break; }
        if (c == '\\') {
            ++pos;
            if (pos >= s.size()) break;
            char e = s[pos];
            switch (e) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case '/': out += '/';  break;
                case '\\': out += '\\'; break;
                case '"': out += '"';  break;
                case 'u': {
                    if (pos + 4 < s.size()) {
                        int u = hex4(s, pos + 1);
                        if (u < 0) { out += 'u'; break; }
                        pos += 4;
                        uint32_t cp = (uint32_t) u;
                        if (cp >= 0xD800 && cp <= 0xDBFF && pos + 6 < s.size()
                            && s[pos + 1] == '\\' && s[pos + 2] == 'u') {
                            int lo = hex4(s, pos + 3);
                            if (lo >= 0xDC00 && lo <= 0xDFFF) {
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                pos += 6;
                            }
                        }
                        append_utf8(out, cp);
                    }
                    break;
                }
                default: out += e; break;
            }
            ++pos;
        } else {
            out += c;
            ++pos;
        }
    }
    return out;
}

// Extract the first user-role message content, JSON-decoded. Falls back to a top-level
// "prompt" field. Returns empty on failure.
static std::string extract_user_prompt(const std::string & line) {
    // look for "role" : "user" then the next "content"
    size_t p = 0;
    while ((p = line.find("\"role\"", p)) != std::string::npos) {
        size_t colon = line.find(':', p + 6);
        if (colon == std::string::npos) break;
        size_t q = line.find('"', colon + 1);
        if (q == std::string::npos) break;
        size_t rp = q;
        std::string role = json_read_string(line, rp);
        if (role == "user") {
            size_t cp = line.find("\"content\"", rp);
            if (cp == std::string::npos) return "";
            size_t cc = line.find(':', cp + 9);
            if (cc == std::string::npos) return "";
            size_t cq = line.find('"', cc + 1);
            if (cq == std::string::npos) return "";
            return json_read_string(line, cq);
        }
        p = rp;
    }
    // fallback: top-level "prompt"
    size_t pp = line.find("\"prompt\"");
    if (pp != std::string::npos) {
        size_t cc = line.find(':', pp + 8);
        size_t cq = line.find('"', cc + 1);
        if (cq != std::string::npos) return json_read_string(line, cq);
    }
    return "";
}

int main(int argc, char ** argv) {
    const char * model_path = (argc > 1) ? argv[1]
        : "/home/usman/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf";
    const char * data_path  = (argc > 2) ? argv[2]
        : "/home/usman/Bonsai-demo/dflash-training/v2/prompts_batch1.jsonl";
    const char * out_path   = (argc > 3) ? argv[3]
        : "/home/usman/Bonsai-demo/dflash-training/v2/feats/batch1.bin";
    const int max_samples   = (argc > 4) ? atoi(argv[4]) : 2000;
    const int max_gen        = (argc > 5) ? atoi(argv[5]) : 256;

    const uint32_t tap_layers[5] = { 6, 20, 34, 48, 62 };
    const int      n_taps = 5;
    const int      max_prompt_tok = 1536;

    printf("=== Bonsai 2 v2 feature extractor (self-distill) ===\n");
    printf("model: %s\ndata:  %s\nout:   %s\nsamples<=%d max_gen=%d\n",
           model_path, data_path, out_path, max_samples, max_gen);

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "ERROR: model load failed\n"); return 1; }

    const int n_embd = llama_model_n_embd(model);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const llama_token eos = llama_vocab_eos(vocab);

    // resolve <|im_end|> special token id
    llama_token im_end = -1;
    {
        const char * s = "<|im_end|>";
        llama_token buf[8];
        int n = llama_tokenize(vocab, s, (int) strlen(s), buf, 8, false, true);
        if (n == 1) im_end = buf[0];
        printf("im_end token id = %d (n=%d), eos = %d\n", im_end, n, eos);
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048; cparams.n_batch = 2048; cparams.n_ubatch = 2048;
    cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "ERROR: ctx create failed\n"); return 1; }

    FILE * fout = fopen(out_path, "wb");
    if (!fout) { fprintf(stderr, "ERROR: cannot open %s\n", out_path); return 1; }
    const char magic[4] = { 'B', 'O', 'N', '2' };
    uint32_t embd_u = (uint32_t) n_embd, taps_u = (uint32_t) n_taps;
    fwrite(magic, 1, 4, fout);
    fwrite(&embd_u, sizeof(uint32_t), 1, fout);
    fwrite(&taps_u, sizeof(uint32_t), 1, fout);
    fwrite(tap_layers, sizeof(uint32_t), 5, fout);

    std::ifstream fin(data_path);
    if (!fin.is_open()) { fprintf(stderr, "ERROR: cannot open %s\n", data_path); return 1; }

    std::string line;
    int n_done = 0, n_skip = 0;
    size_t total_feat_tok = 0, total_gen_tok = 0;
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);

    std::vector<llama_token> full;
    std::vector<float> taps;         // n*5*embd
    std::vector<float> final_hidden; // n*embd

    while (std::getline(fin, line) && n_done < max_samples) {
        if (line.empty()) continue;
        std::string user = extract_user_prompt(line);
        if (user.empty()) { n_skip++; continue; }

        std::string prompt_text = "<|im_start|>user\n" + user
            + "<|im_end|>\n<|im_start|>assistant\n";

        std::vector<llama_token> ptoks(prompt_text.size() + 16);
        int n_p = llama_tokenize(vocab, prompt_text.c_str(), (int) prompt_text.size(),
                                 ptoks.data(), (int) ptoks.size(), true, true);
        if (n_p <= 4 || n_p > max_prompt_tok) { n_skip++; continue; }
        ptoks.resize(n_p);

        // --- generation phase (embeddings off, taps off, greedy) ---
        llama_set_embeddings(ctx, false);
        for (int l = 0; l < n_taps; ++l) llama_set_embeddings_layer_inp(ctx, tap_layers[l], false);
        llama_memory_clear(llama_get_memory(ctx), true);

        llama_batch pb = llama_batch_init(n_p, 0, 1);
        for (int i = 0; i < n_p; ++i) {
            pb.token[i] = ptoks[i]; pb.pos[i] = i;
            pb.n_seq_id[i] = 1; pb.seq_id[i][0] = 0;
            pb.logits[i] = (i == n_p - 1);
        }
        pb.n_tokens = n_p;
        if (llama_decode(ctx, pb) != 0) { llama_batch_free(pb); n_skip++; continue; }
        llama_batch_free(pb);

        std::vector<llama_token> gen;
        int cur_pos = n_p;
        const float * lg = llama_get_logits_ith(ctx, n_p - 1); // last prompt position
        bool gen_ok = (lg != nullptr);
        while (gen_ok && (int) gen.size() < max_gen && cur_pos < cparams.n_ctx - 1) {
            int am = 0; float mx = lg[0];
            for (int v = 1; v < (int) llama_vocab_n_tokens(vocab); ++v)
                if (lg[v] > mx) { mx = lg[v]; am = v; }
            gen.push_back(am);
            if (am == im_end || am == eos) break;
            llama_batch sb = llama_batch_init(1, 0, 1);
            sb.token[0] = am; sb.pos[0] = cur_pos;
            sb.n_seq_id[0] = 1; sb.seq_id[0][0] = 0; sb.logits[0] = true;
            sb.n_tokens = 1;
            if (llama_decode(ctx, sb) != 0) { llama_batch_free(sb); break; }
            llama_batch_free(sb);
            cur_pos++;
            lg = llama_get_logits_ith(ctx, 0);
            if (!lg) break;
        }
        if (gen.empty()) { n_skip++; continue; }

        // --- assemble full sequence + loss mask ---
        full.clear();
        full.insert(full.end(), ptoks.begin(), ptoks.end());
        full.insert(full.end(), gen.begin(), gen.end());
        const int n = (int) full.size();
        if (n > cparams.n_ctx) { n_skip++; continue; }
        std::vector<uint8_t> loss_mask(n, 0);
        for (int i = n_p; i < n; ++i) loss_mask[i] = 1;

        // --- feature phase (embeddings on + taps on, single teacher-forced pass) ---
        llama_set_embeddings(ctx, true);
        for (int l = 0; l < n_taps; ++l) llama_set_embeddings_layer_inp(ctx, tap_layers[l], true);
        llama_memory_clear(llama_get_memory(ctx), true);

        llama_batch fb = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; ++i) {
            fb.token[i] = full[i]; fb.pos[i] = i;
            fb.n_seq_id[i] = 1; fb.seq_id[i][0] = 0; fb.logits[i] = true;
        }
        fb.n_tokens = n;
        if (llama_decode(ctx, fb) != 0) { llama_batch_free(fb); n_skip++; continue; }
        llama_batch_free(fb);

        taps.assign((size_t) n * n_taps * n_embd, 0.0f);
        bool feat_ok = true;
        for (int l = 0; l < n_taps; ++l) {
            float * e = llama_get_embeddings_layer_inp(ctx, tap_layers[l]);
            if (!e) { feat_ok = false; break; }
            for (int t = 0; t < n; ++t)
                memcpy(&taps[((size_t) t * n_taps + l) * n_embd], &e[(size_t) t * n_embd],
                       n_embd * sizeof(float));
        }
        if (!feat_ok) { n_skip++; continue; }

        final_hidden.assign((size_t) n * n_embd, 0.0f);
        for (int t = 0; t < n; ++t) {
            const float * em = llama_get_embeddings_ith(ctx, t);
            if (!em) { feat_ok = false; break; }
            memcpy(&final_hidden[(size_t) t * n_embd], em, n_embd * sizeof(float));
        }
        if (!feat_ok) { n_skip++; continue; }

        // --- write sample ---
        uint32_t n_u = (uint32_t) n;
        fwrite(&n_u, sizeof(uint32_t), 1, fout);
        fwrite(full.data(), sizeof(int32_t), n, fout);
        fwrite(loss_mask.data(), sizeof(uint8_t), n, fout);
        fwrite(taps.data(), sizeof(float), taps.size(), fout);
        fwrite(final_hidden.data(), sizeof(float), final_hidden.size(), fout);

        n_done++;
        total_feat_tok += n;
        total_gen_tok += gen.size();
        if (n_done % 50 == 0) {
            struct timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
            double el = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
            printf("  done %d (skip %d) | feat_tok %zu gen_tok %zu | %.1f feat_tok/s %.1f gen_tok/s | %.1fs\n",
                   n_done, n_skip, total_feat_tok, total_gen_tok,
                   total_feat_tok / el, total_gen_tok / el, el);
            fflush(stdout); fflush(fout);
        }
    }

    fclose(fout);
    struct timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
    double el = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    printf("\n=== done ===\n");
    printf("samples=%d skipped=%d feat_tok=%zu gen_tok=%zu time=%.1fs\n",
           n_done, n_skip, total_feat_tok, total_gen_tok, el);
    printf("throughput: %.1f feat_tok/s, %.1f gen_tok/s\n",
           total_feat_tok / el, total_gen_tok / el);
    printf("out: %s\n", out_path);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
