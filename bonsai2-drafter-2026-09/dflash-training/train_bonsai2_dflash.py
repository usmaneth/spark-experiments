#!/usr/bin/env python3
"""
Train/Fine-tune DFlash/DSpark drafter head directly on Bonsai 2 Hadamard-rotated activations.
Warm-starts from RadixArk/Qwen3.8-27B-DSpark weights for fast convergence.
"""

import os, sys, time, struct, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from safetensors import safe_open
from safetensors.torch import save_file

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class DFlashAttention(nn.Module):
    def __init__(self, hidden_size=5120, num_heads=32, num_kv_heads=8, head_dim=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, x):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim)
        
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        
        # GQA repeat kv
        if self.num_heads != self.num_kv_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)

class DFlashMLP(nn.Module):
    def __init__(self, hidden_size=5120, intermediate_size=17408):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class DFlashBlock(nn.Module):
    def __init__(self, hidden_size=5120, intermediate_size=17408):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = DFlashAttention(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = DFlashMLP(hidden_size, intermediate_size)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class DFlashDraftHead(nn.Module):
    def __init__(self, n_taps=5, hidden_size=5120, n_layers=5, vocab_size=248320):
        super().__init__()
        self.fc = nn.Linear(n_taps * hidden_size, hidden_size, bias=False)
        self.hidden_norm = RMSNorm(hidden_size)
        self.layers = nn.ModuleList([DFlashBlock(hidden_size) for _ in range(n_layers)])
        self.norm = RMSNorm(hidden_size)
        
        # Markov and confidence heads
        self.markov_w1 = nn.Parameter(torch.zeros(vocab_size, 256))
        self.markov_w2 = nn.Parameter(torch.zeros(vocab_size, 256))
        self.conf_proj_w = nn.Parameter(torch.zeros(1, 5376))
        self.conf_proj_b = nn.Parameter(torch.zeros(1))

    def forward(self, features):
        # features: [B, S, 25600]
        h = self.hidden_norm(self.fc(features))
        for layer in self.layers:
            h = layer(h)
        return self.norm(h)

class BonsaiFeatureDataset(Dataset):
    def __init__(self, binary_paths, max_seq_len=512):
        if isinstance(binary_paths, str):
            binary_paths = [binary_paths]
        self.binary_paths = binary_paths
        self.max_seq_len = max_seq_len
        self.offsets = []
        
        for p_idx, bp in enumerate(binary_paths):
            if not os.path.exists(bp):
                print(f"Skipping non-existent dataset: {bp}")
                continue
            print(f"Indexing binary features from {bp}...")
            with open(bp, "rb") as f:
                magic = f.read(4)
                if magic != b"BONS":
                    raise ValueError(f"Invalid magic in {bp}")
                self.embd_dim, self.n_taps = struct.unpack("<II", f.read(8))
                self.layers = struct.unpack("<5I", f.read(20))
                self.stride = self.n_taps * self.embd_dim
                
                while True:
                    offset = f.tell()
                    tok_bytes = f.read(4)
                    if not tok_bytes: break
                    n_tokens = struct.unpack("<I", tok_bytes)[0]
                    skip_bytes = n_tokens * 4 + n_tokens * self.stride * 4
                    f.seek(skip_bytes, os.SEEK_CUR)
                    self.offsets.append((p_idx, offset, n_tokens))
                    
        print(f"Total indexed samples across {len(self.binary_paths)} file(s): {len(self.offsets)}")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        p_idx, offset, n_tokens = self.offsets[idx]
        bp = self.binary_paths[p_idx]
        with open(bp, "rb") as f:
            f.seek(offset + 4)
            tokens = np.frombuffer(f.read(n_tokens * 4), dtype=np.int32)
            feats = np.frombuffer(f.read(n_tokens * self.stride * 4), dtype=np.float32).reshape(n_tokens, self.stride)
            
        if n_tokens > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
            feats = feats[:self.max_seq_len]
            
        return (
            torch.from_numpy(tokens.copy()).long(),
            torch.from_numpy(feats.copy()).float()
        )
def collate_fn(batch):
    # Pad sequences to max length in batch
    max_len = max(s[0].shape[0] for s in batch)
    B = len(batch)
    stride = batch[0][1].shape[1]
    
    padded_tokens = torch.zeros(B, max_len, dtype=torch.long)
    padded_feats  = torch.zeros(B, max_len, stride, dtype=torch.float32)
    mask          = torch.zeros(B, max_len, dtype=torch.bool)
    
    for i, (tok, feat) in enumerate(batch):
        L = tok.shape[0]
        padded_tokens[i, :L] = tok
        padded_feats[i, :L]  = feat
        mask[i, :L] = True
        
    return padded_tokens, padded_feats, mask

def train(features_bin, pretrained_st, output_st, epochs=2, lr=1e-4, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    dataset = BonsaiFeatureDataset(features_bin)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    model = DFlashDraftHead().to(device=device, dtype=torch.bfloat16)
    
    # Load warm-start weights from RadixArk
    if os.path.exists(pretrained_st):
        print(f"Warm-starting from {pretrained_st}...")
        with safe_open(pretrained_st, framework="pt") as f:
            sd = {}
            for k in f.keys():
                t = f.get_tensor(k)
                # Map names to model parameters
                if k == "markov_head.markov_w1.weight": sd["markov_w1"] = t
                elif k == "markov_head.markov_w2.weight": sd["markov_w2"] = t
                elif k == "confidence_head.proj.weight": sd["conf_proj_w"] = t
                elif k == "confidence_head.proj.bias": sd["conf_proj_b"] = t
                else: sd[k] = t
            model.load_state_dict(sd, strict=False)
            print("Warm-start weights loaded successfully!")

    # Separate lr: higher lr for fc and heads, lower for transformer blocks
    optimizer = torch.optim.AdamW([
        {"params": [model.fc.weight, model.hidden_norm.weight], "lr": lr * 2},
        {"params": model.layers.parameters(), "lr": lr},
        {"params": [model.norm.weight], "lr": lr},
        {"params": [model.conf_proj_w, model.conf_proj_b], "lr": lr * 2}
    ], weight_decay=0.01)

    total_steps = len(dataloader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    print(f"=== Starting Multi-Epoch Training on GB10 GPU ({total_steps} steps, {epochs} epochs) ===")
    step = 0
    start_time = time.time()
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for b_idx, (tokens, feats, mask) in enumerate(dataloader):
            tokens = tokens.to(device)
            feats = feats.to(device, dtype=torch.bfloat16)
            mask = mask.to(device)
            
            optimizer.zero_grad()
            
            # Predict hidden state representation
            h_pred = model(feats) # [B, S, 5120]
            target_h61 = feats[:, :, 4*5120:5*5120] # target layer 61
            
            # Cosine similarity + MSE loss
            cos_sim = F.cosine_similarity(h_pred, target_h61, dim=-1)
            cos_loss = 1.0 - cos_sim
            mse_loss = F.mse_loss(h_pred, target_h61, reduction="none").mean(dim=-1)
            loss_recon = (cos_loss + 0.5 * mse_loss)[mask].mean()
            
            # Confidence head calibration loss:
            # conf_inp: [B, S-1, 5120 + 256]
            prev_toks = torch.clamp(tokens[:, :-1], 0, model.markov_w1.shape[0] - 1)
            w1_prev = F.embedding(prev_toks, model.markov_w1) # [B, S-1, 256]
            conf_feat = torch.cat([h_pred[:, :-1], w1_prev], dim=-1) # [B, S-1, 5376]
            conf_logits = F.linear(conf_feat, model.conf_proj_w, model.conf_proj_b).squeeze(-1) # [B, S-1]
            
            # Ground truth: target is 1.0 if cosine similarity >= 0.85 (high accuracy token), else 0.0
            conf_target = (cos_sim[:, :-1] >= 0.85).float()
            mask_sub = mask[:, :-1]
            loss_conf = F.binary_cross_entropy_with_logits(conf_logits[mask_sub], conf_target[mask_sub])
            
            loss = loss_recon + 0.25 * loss_conf
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            step += 1
            
            if step % 50 == 0 or step == total_steps:
                elapsed = time.time() - start_time
                eta_min = (total_steps - step) * (elapsed / max(1, step)) / 60.0
                vram_used = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1}/{epochs} | Step {step}/{total_steps} | Loss: {loss.item():.4f} (Recon: {loss_recon.item():.4f}, Conf: {loss_conf.item():.4f}) | LR: {current_lr:.2e} | Elapsed: {elapsed/60.0:.1f}m | ETA: {eta_min:.1f}m")
                
        # Save checkpoint after each epoch
        print(f"=== Epoch {epoch+1}/{epochs} Complete. Saving checkpoint... ===")
        os.makedirs(os.path.dirname(output_st), exist_ok=True)
        out_sd = {}
        for k, v in model.state_dict().items():
            if k == "markov_w1": out_sd["markov_head.markov_w1.weight"] = v.cpu()
            elif k == "markov_w2": out_sd["markov_head.markov_w2.weight"] = v.cpu()
            elif k == "conf_proj_w": out_sd["confidence_head.proj.weight"] = v.cpu()
            elif k == "conf_proj_b": out_sd["confidence_head.proj.bias"] = v.cpu()
            else: out_sd[k] = v.cpu()
        save_file(out_sd, output_st)
        save_file(out_sd, output_st.replace('.safetensors', f'_epoch{epoch+1}.safetensors'))
        print(f"Checkpoint saved to {output_st} and epoch_{epoch+1}.safetensors")

    print("=== All Training Epochs Complete ===")
if __name__ == "__main__":
    part1 = "/home/usman/Bonsai-demo/dflash-training/bonsai2_features_part1.bin"
    part2 = "/home/usman/Bonsai-demo/dflash-training/bonsai2_features_part2.bin"
    part1k = "/home/usman/Bonsai-demo/dflash-training/bonsai2_features_1k.bin"
    
    features_bin = []
    if os.path.exists(part1): features_bin.append(part1)
    if os.path.exists(part2): features_bin.append(part2)
    if not features_bin and os.path.exists(part1k): features_bin.append(part1k)
    
    pretrained_st = "/home/usman/Bonsai-demo/models/qwen38-dspark/model.safetensors"
    output_st = "/home/usman/Bonsai-demo/models/bonsai2-dspark/bonsai2_dspark_trained.safetensors"
    
    epochs = 3
    if len(sys.argv) > 1: epochs = int(sys.argv[1])
    train(features_bin, pretrained_st, output_st, epochs=epochs)
