#!/usr/bin/env python3
"""
Convert RadixArk/Qwen3.8-27B-DSpark (SpecForge PyTorch safetensors)
into a native GGUF DFlash/DSpark drafter compatible with llama.cpp.
"""

import sys, os, struct, json
import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, '/home/usman/Bonsai-demo/llama.cpp/gguf-py')
from gguf import GGUFWriter, GGUFReader

def convert(safetensors_path, donor_gguf_path, output_gguf_path):
    print(f"Reading donor tokenizer from {donor_gguf_path}...")
    donor = GGUFReader(donor_gguf_path)
    
    writer = GGUFWriter(output_gguf_path, "dflash")
    
    # 1. Architecture & General metadata
    writer.add_architecture()
    writer.add_name("Qwen3.8-27B-DSpark")
    writer.add_type("model")
    writer.add_size_label("1.86B")
    writer.add_quantization_version(2)
    writer.add_file_type(32) # BF16
    
    # 2. DFlash / DSpark hyperparameters
    # Target layers: RadixArk used [5, 19, 33, 47, 61]
    # dflash.cpp adds +1 because runtime taps layer input: [6, 20, 34, 48, 62]
    writer.add_uint32("dflash.block_count", 5)
    writer.add_uint32("dflash.context_length", 262144)
    writer.add_uint32("dflash.embedding_length", 5120)
    writer.add_uint32("dflash.feed_forward_length", 17408)
    writer.add_uint32("dflash.attention.head_count", 32)
    writer.add_uint32("dflash.attention.head_count_kv", 8)
    writer.add_float32("dflash.rope.freq_base", 10000000.0)
    writer.add_float32("dflash.attention.layer_norm_rms_epsilon", 1e-06)
    writer.add_uint32("dflash.attention.key_length", 128)
    writer.add_uint32("dflash.attention.value_length", 128)
    writer.add_uint32("dflash.block_size", 7)
    writer.add_array("dflash.target_layers", [6, 20, 34, 48, 62])
    writer.add_uint32("dflash.markov_rank", 256)
    writer.add_bool("dflash.confidence_head", True)
    writer.add_bool("dflash.confidence_head_with_markov", True)
    writer.add_uint32("dflash.vocab_size", 248320)
    
    # 3. Copy Tokenizer metadata from donor GGUF
    print("Injecting donor tokenizer metadata...")
    for field in donor.fields.values():
        if field.name.startswith("tokenizer."):
            writer.add_key_value(field.name, field.parts[field.data[0]] if len(field.data) == 1 else [field.parts[d] for d in field.data], field.types)

    # 4. Map and write tensors from safetensors
    print(f"Loading weights from {safetensors_path}...")
    name_map = {
        "fc.weight": "fc.weight",
        "hidden_norm.weight": "enc.output_norm.weight",
        "norm.weight": "output_norm.weight",
        "markov_head.markov_w1.weight": "markov_w1.weight",
        "markov_head.markov_w2.weight": "markov_w2.weight",
        "confidence_head.proj.weight": "conf_proj.weight",
        "confidence_head.proj.bias": "conf_proj.bias",
    }
    
    for i in range(5):
        name_map[f"layers.{i}.input_layernorm.weight"] = f"blk.{i}.attn_norm.weight"
        name_map[f"layers.{i}.self_attn.q_proj.weight"] = f"blk.{i}.attn_q.weight"
        name_map[f"layers.{i}.self_attn.k_proj.weight"] = f"blk.{i}.attn_k.weight"
        name_map[f"layers.{i}.self_attn.v_proj.weight"] = f"blk.{i}.attn_v.weight"
        name_map[f"layers.{i}.self_attn.o_proj.weight"] = f"blk.{i}.attn_output.weight"
        name_map[f"layers.{i}.self_attn.q_norm.weight"] = f"blk.{i}.attn_q_norm.weight"
        name_map[f"layers.{i}.self_attn.k_norm.weight"] = f"blk.{i}.attn_k_norm.weight"
        name_map[f"layers.{i}.post_attention_layernorm.weight"] = f"blk.{i}.ffn_norm.weight"
        name_map[f"layers.{i}.mlp.gate_proj.weight"] = f"blk.{i}.ffn_gate.weight"
        name_map[f"layers.{i}.mlp.up_proj.weight"] = f"blk.{i}.ffn_up.weight"
        name_map[f"layers.{i}.mlp.down_proj.weight"] = f"blk.{i}.ffn_down.weight"

    with safe_open(safetensors_path, framework="pt") as f:
        st_keys = set(f.keys())
        print(f"Converting {len(name_map)} tensors...")
        for st_name, gguf_name in name_map.items():
            if st_name not in st_keys:
                print(f"WARNING: {st_name} missing from safetensors!")
                continue
            t = f.get_tensor(st_name)
            t_np = t.contiguous().to(dtype=t.dtype if t.dtype != torch.bfloat16 else torch.float32).cpu().numpy()
            writer.add_tensor(gguf_name, t_np)
            
    print(f"Writing GGUF output to {output_gguf_path}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("Conversion complete!")

if __name__ == "__main__":
    safetensors_path = "/home/usman/Bonsai-demo/models/qwen38-dspark/model.safetensors"
    donor_gguf_path = "/home/usman/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf"
    output_gguf_path = "/home/usman/Bonsai-demo/models/qwen38-dspark/Qwen3.8-27B-DSpark-raw.gguf"
    convert(safetensors_path, donor_gguf_path, output_gguf_path)
