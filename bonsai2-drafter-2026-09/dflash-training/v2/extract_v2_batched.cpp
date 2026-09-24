// extract_v2_batched.cpp
// Batched self-distillation feature extractor (GB10 is memory-bandwidth bound, so
// generating G completions in parallel costs ~1 weight read per step, not G).
// Phase 1: batched greedy generation of G sequences at once (one token/seq/step).
// Phase 2: teacher-force each finished (prompt+completion) single-stream for taps + final_hidden.
// Same BON2 output format and JSON-decode fix as extract_v2.cpp.
//
// Output: header "BON2"(4)|embd u32|n_taps u32=5|tap_layers[5] u32={6,20,34,48,62}
//   per sample: n u32 | tokens i32[n] | loss_mask u8[n] | taps f32[n*5*embd] | final_hidden f32[n*embd]

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

// ---- JSON string decode (unescape \n \t \r \" \\ \/ \uXXXX incl. surrogates) ----
static void append_utf8(std::string & out, uint32_t cp) {
    if (cp <= 0x7F) out += (char) cp;
    else if (cp <= 0x7FF) { out += (char)(0xC0|(cp>>6)); out += (char)(0x80|(cp&0x3F)); }
    else if (cp <= 0xFFFF) { out += (char)(0xE0|(cp>>12)); out += (char)(0x80|((cp>>6)&0x3F)); out += (char)(0x80|(cp&0x3F)); }
    else { out += (char)(0xF0|(cp>>18)); out += (char)(0x80|((cp>>12)&0x3F)); out += (char)(0x80|((cp>>6)&0x3F)); out += (char)(0x80|(cp&0x3F)); }
}
static int hex4(const std::string & s, size_t p) {
    int v=0; for (int i=0;i<4;++i){ char c=s[p+i]; v<<=4;
        if(c>='0'&&c<='9')v|=c-'0'; else if(c>='a'&&c<='f')v|=c-'a'+10; else if(c>='A'&&c<='F')v|=c-'A'+10; else return -1; }
    return v;
}
static std::string json_read_string(const std::string & s, size_t & pos) {
    std::string out;
    if (pos>=s.size()||s[pos]!='"') return out;
    ++pos;
    while (pos<s.size()) {
        char c=s[pos];
        if (c=='"'){ ++pos; break; }
        if (c=='\\'){ ++pos; if(pos>=s.size())break; char e=s[pos];
            switch(e){
                case 'n':out+='\n';break; case 't':out+='\t';break; case 'r':out+='\r';break;
                case 'b':out+='\b';break; case 'f':out+='\f';break; case '/':out+='/';break;
                case '\\':out+='\\';break; case '"':out+='"';break;
                case 'u': { if(pos+4<s.size()){ int u=hex4(s,pos+1); if(u<0){out+='u';break;} pos+=4;
                        uint32_t cp=(uint32_t)u;
                        if(cp>=0xD800&&cp<=0xDBFF&&pos+6<s.size()&&s[pos+1]=='\\'&&s[pos+2]=='u'){
                            int lo=hex4(s,pos+3); if(lo>=0xDC00&&lo<=0xDFFF){ cp=0x10000+((cp-0xD800)<<10)+(lo-0xDC00); pos+=6; } }
                        append_utf8(out,cp); } break; }
                default: out+=e; break;
            } ++pos;
        } else { out+=c; ++pos; }
    }
    return out;
}
static std::string extract_user_prompt(const std::string & line) {
    size_t p=0;
    while ((p=line.find("\"role\"",p))!=std::string::npos) {
        size_t colon=line.find(':',p+6); if(colon==std::string::npos)break;
        size_t q=line.find('"',colon+1); if(q==std::string::npos)break;
        size_t rp=q; std::string role=json_read_string(line,rp);
        if (role=="user") {
            size_t cp=line.find("\"content\"",rp); if(cp==std::string::npos)return "";
            size_t cc=line.find(':',cp+9); if(cc==std::string::npos)return "";
            size_t cq=line.find('"',cc+1); if(cq==std::string::npos)return "";
            return json_read_string(line,cq);
        }
        p=rp;
    }
    size_t pp=line.find("\"prompt\"");
    if (pp!=std::string::npos){ size_t cc=line.find(':',pp+8); size_t cq=line.find('"',cc+1);
        if(cq!=std::string::npos) return json_read_string(line,cq); }
    return "";
}

int main(int argc, char ** argv) {
    const char * model_path = (argc>1)?argv[1] : "/home/REDACTED/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf";
    const char * data_path  = (argc>2)?argv[2] : "/home/REDACTED/Bonsai-demo/dflash-training/v2/prompts_batch1.jsonl";
    const char * out_path   = (argc>3)?argv[3] : "/home/REDACTED/Bonsai-demo/dflash-training/v2/feats/batch2.bin";
    const int max_samples = (argc>4)?atoi(argv[4]) : 2000;
    const int max_gen     = (argc>5)?atoi(argv[5]) : 256;
    const int group_size  = (argc>6)?atoi(argv[6]) : 16;   // parallel sequences
    const int skip_lines  = (argc>7)?atoi(argv[7]) : 0;    // skip already-consumed prompts

    const uint32_t tap_layers[5] = {6,20,34,48,62};
    const int n_taps = 5;
    const int per_seq = 1024;                 // ctx per sequence (prompt<=768 + gen<=256)
    const int max_prompt_tok = per_seq - max_gen;

    printf("=== Bonsai 2 v2 BATCHED self-distill extractor ===\n");
    printf("model:%s\ndata:%s\nout:%s\nsamples<=%d max_gen=%d group=%d skip_lines=%d\n",
           model_path,data_path,out_path,max_samples,max_gen,group_size,skip_lines);

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model){ fprintf(stderr,"ERROR: model load failed\n"); return 1; }
    const int n_embd = llama_model_n_embd(model);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const llama_token eos = llama_vocab_eos(vocab);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_token im_end = -1;
    { const char * s="<|im_end|>"; llama_token b[8]; int n=llama_tokenize(vocab,s,(int)strlen(s),b,8,false,true); if(n==1) im_end=b[0]; }
    printf("im_end=%d eos=%d n_vocab=%d n_embd=%d\n", im_end, eos, n_vocab, n_embd);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_seq_max = group_size;
    cparams.n_ctx     = group_size * per_seq;   // total; per-seq = n_ctx / n_seq_max
    cparams.n_batch   = 2048;
    cparams.n_ubatch  = 2048;
    cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx){ fprintf(stderr,"ERROR: ctx create failed\n"); return 1; }

    FILE * fout = fopen(out_path,"wb");
    if (!fout){ fprintf(stderr,"ERROR: cannot open %s\n",out_path); return 1; }
    const char magic[4]={'B','O','N','2'};
    uint32_t embd_u=(uint32_t)n_embd, taps_u=(uint32_t)n_taps;
    fwrite(magic,1,4,fout); fwrite(&embd_u,4,1,fout); fwrite(&taps_u,4,1,fout); fwrite(tap_layers,4,5,fout);

    std::ifstream fin(data_path);
    if (!fin.is_open()){ fprintf(stderr,"ERROR: cannot open %s\n",data_path); return 1; }
    std::string line;
    for (int i=0;i<skip_lines && std::getline(fin,line);++i) {}

    int n_done=0, n_skip=0;
    size_t total_feat_tok=0, total_gen_tok=0;
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC,&t0);
    double gen_seconds=0.0;

    std::vector<float> taps, final_hidden;

    llama_memory_t mem = llama_get_memory(ctx);

    while (n_done < max_samples) {
        // ---- collect a group of valid prompts ----
        std::vector<std::vector<llama_token>> ptoks;   // per-seq prompt tokens
        while ((int)ptoks.size() < group_size && std::getline(fin, line)) {
            if (line.empty()) continue;
            std::string user = extract_user_prompt(line);
            if (user.empty()) { n_skip++; continue; }
            std::string pt = "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
            std::vector<llama_token> t(pt.size()+16);
            int np = llama_tokenize(vocab, pt.c_str(), (int)pt.size(), t.data(), (int)t.size(), true, true);
            if (np<=4 || np>max_prompt_tok) { n_skip++; continue; }
            t.resize(np);
            ptoks.push_back(std::move(t));
        }
        const int G = (int)ptoks.size();
        if (G==0) break;

        // ---- phase 1: batched greedy generation ----
        struct timespec g0; clock_gettime(CLOCK_MONOTONIC,&g0);
        llama_set_embeddings(ctx, false);
        for (int l=0;l<n_taps;++l) llama_set_embeddings_layer_inp(ctx, tap_layers[l], false);
        llama_memory_clear(mem, true);

        std::vector<std::vector<llama_token>> gen(G);
        std::vector<int> cur_pos(G);
        std::vector<int> nexttok(G, -1);
        std::vector<char> active(G, 1);

        // prefill each sequence separately (fast); read its last-token logits immediately
        for (int g=0; g<G; ++g) {
            const int np = (int)ptoks[g].size();
            llama_batch pb = llama_batch_init(np, 0, 1);
            for (int i=0;i<np;++i){ pb.token[i]=ptoks[g][i]; pb.pos[i]=i; pb.n_seq_id[i]=1; pb.seq_id[i][0]=g; pb.logits[i]=(i==np-1); }
            pb.n_tokens = np;
            if (llama_decode(ctx, pb)!=0) { active[g]=0; llama_batch_free(pb); continue; }
            const float * lg = llama_get_logits_ith(ctx, np-1);
            if (!lg) { active[g]=0; llama_batch_free(pb); continue; }
            int am=0; float mx=lg[0]; for (int v=1;v<n_vocab;++v) if(lg[v]>mx){mx=lg[v];am=v;}
            nexttok[g]=am; cur_pos[g]=np;
            llama_batch_free(pb);
        }

        // batched autoregressive steps: one token per active seq per decode
        for (int step=0; step<max_gen; ++step) {
            std::vector<int> order; order.reserve(G);
            for (int g=0; g<G; ++g) if (active[g]) order.push_back(g);
            if (order.empty()) break;
            llama_batch b = llama_batch_init((int)order.size(), 0, 1);
            for (size_t k=0;k<order.size();++k){ int g=order[k];
                b.token[k]=nexttok[g]; b.pos[k]=cur_pos[g]; b.n_seq_id[k]=1; b.seq_id[k][0]=g; b.logits[k]=true; }
            b.n_tokens=(int)order.size();
            if (llama_decode(ctx, b)!=0) { llama_batch_free(b); break; }
            for (size_t k=0;k<order.size();++k){ int g=order[k];
                gen[g].push_back(nexttok[g]);           // commit the token we just fed forward
                if (nexttok[g]==im_end || nexttok[g]==eos || (int)gen[g].size()>=max_gen) { active[g]=0; continue; }
                const float * lg = llama_get_logits_ith(ctx, (int)k);
                if (!lg) { active[g]=0; continue; }
                int am=0; float mx=lg[0]; for (int v=1;v<n_vocab;++v) if(lg[v]>mx){mx=lg[v];am=v;}
                nexttok[g]=am; cur_pos[g]++;
            }
            llama_batch_free(b);
        }
        struct timespec g1; clock_gettime(CLOCK_MONOTONIC,&g1);
        gen_seconds += (g1.tv_sec-g0.tv_sec)+(g1.tv_nsec-g0.tv_nsec)/1e9;

        // ---- phase 2: teacher-force each sequence for taps + final_hidden ----
        llama_set_embeddings(ctx, true);
        for (int l=0;l<n_taps;++l) llama_set_embeddings_layer_inp(ctx, tap_layers[l], true);

        for (int g=0; g<G && n_done<max_samples; ++g) {
            if (gen[g].empty()) { n_skip++; continue; }
            const int np=(int)ptoks[g].size();
            std::vector<llama_token> full; full.reserve(np+gen[g].size());
            full.insert(full.end(), ptoks[g].begin(), ptoks[g].end());
            full.insert(full.end(), gen[g].begin(), gen[g].end());
            const int n=(int)full.size();
            std::vector<uint8_t> loss_mask(n,0);
            for (int i=np;i<n;++i) loss_mask[i]=1;

            llama_memory_clear(mem, true);
            llama_batch fb = llama_batch_init(n, 0, 1);
            for (int i=0;i<n;++i){ fb.token[i]=full[i]; fb.pos[i]=i; fb.n_seq_id[i]=1; fb.seq_id[i][0]=0; fb.logits[i]=true; }
            fb.n_tokens=n;
            if (llama_decode(ctx, fb)!=0) { llama_batch_free(fb); n_skip++; continue; }
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

            uint32_t n_u=(uint32_t)n;
            fwrite(&n_u,4,1,fout);
            fwrite(full.data(),sizeof(int32_t),n,fout);
            fwrite(loss_mask.data(),1,n,fout);
            fwrite(taps.data(),sizeof(float),taps.size(),fout);
            fwrite(final_hidden.data(),sizeof(float),final_hidden.size(),fout);
            n_done++; total_feat_tok+=n; total_gen_tok+=gen[g].size();
        }

        struct timespec t1; clock_gettime(CLOCK_MONOTONIC,&t1);
        double el=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
        printf("  done %d (skip %d) | feat_tok %zu gen_tok %zu | gen %.1f tok/s (aggregate) | overall %.1f feat_tok/s | %.1fs\n",
               n_done, n_skip, total_feat_tok, total_gen_tok,
               gen_seconds>0? total_gen_tok/gen_seconds : 0.0, total_feat_tok/el, el);
        fflush(stdout); fflush(fout);
    }

    fclose(fout);
    struct timespec t1; clock_gettime(CLOCK_MONOTONIC,&t1);
    double el=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
    printf("\n=== done ===\n");
    printf("samples=%d skipped=%d feat_tok=%zu gen_tok=%zu time=%.1fs gen_time=%.1fs\n",
           n_done,n_skip,total_feat_tok,total_gen_tok,el,gen_seconds);
    printf("gen throughput=%.1f tok/s (aggregate over %d parallel), overall=%.1f feat_tok/s\n",
           gen_seconds>0? total_gen_tok/gen_seconds:0.0, group_size, total_feat_tok/el);
    printf("out: %s\n", out_path);
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
