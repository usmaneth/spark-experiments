#!/usr/bin/env python3
"""
eagle3_to_gguf.py - write an arch=eagle3 GGUF from a train_eagle3.py checkpoint.

The checkpoint already uses the GGML tensor names. This script adds the
eagle3.* metadata, copies every tokenizer.* key from the donor GGUF and writes
the tensors. It then prints the KV list and the tensor list of the new file.

Tensor types: 1-D tensors F32, d2t I64, 2-D matrices F16, token_embd.weight per
--embd-type (f16 default). Quantize the result with llama-quantize (Q8_0).

Usage:
    eagle3_to_gguf.py <ckpt.safetensors> <donor.gguf> <out.gguf> [--embd-type f16|f32]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

LLAMA_CPP = "/home/usman/Bonsai-demo/llama.cpp"
sys.path.insert(0, os.path.join(LLAMA_CPP, "gguf-py"))
import gguf  # noqa: E402
from gguf import GGUFReader, GGUFValueType, GGUFWriter  # noqa: E402

ARCH = "eagle3"

# metadata per DESIGN.md
HPARAMS = {
    "block_count": 1,
    "context_length": 8192,
    "embedding_length": 5120,
    "feed_forward_length": 17408,
    "head_count": 32,
    "head_count_kv": 8,
    "key_length": 128,
    "value_length": 128,
    "rope_dimension_count": 128,
    "rope_freq_base": 1.0e7,
    "rms_eps": 1e-6,
    "vocab_size": 248320,
    "target_layers": [6, 34, 62],
    "target_hidden_size": 5120,
    "norm_before_fc": True,
    "norm_before_residual": False,
}

# expected numpy shapes (torch order); the writer reverses the dims
EXPECTED_SHAPES = {
    "fc.weight": (5120, 15360),
    "enc.output_norm.weight": (15360,),
    "blk.0.attn_norm.weight": (5120,),
    "blk.0.attn_norm_2.weight": (5120,),
    "blk.0.attn_q.weight": (4096, 10240),
    "blk.0.attn_k.weight": (1024, 10240),
    "blk.0.attn_v.weight": (1024, 10240),
    "blk.0.attn_output.weight": (5120, 4096),
    "blk.0.ffn_norm.weight": (5120,),
    "blk.0.ffn_gate.weight": (17408, 5120),
    "blk.0.ffn_up.weight": (17408, 5120),
    "blk.0.ffn_down.weight": (5120, 17408),
    "output_norm.weight": (5120,),
    "output.weight": (32768, 5120),
    "d2t": (32768,),
    "token_embd.weight": (248320, 5120),
}


def copy_tokenizer_keys(writer: GGUFWriter, donor_path: str) -> int:
    """Copy every tokenizer.* key of the donor with explicit value types."""
    donor = GGUFReader(donor_path)
    n = 0
    for name, field in donor.fields.items():
        if not name.startswith("tokenizer."):
            continue
        vtype = field.types[0]
        if vtype == GGUFValueType.ARRAY:
            sub = field.types[1]
            if sub == GGUFValueType.STRING:
                val = [bytes(field.parts[i]) for i in field.data]
            else:
                val = [field.parts[i][0].item() for i in field.data]
            writer.add_key_value(name, val, GGUFValueType.ARRAY, sub_type=sub)
        elif vtype == GGUFValueType.STRING:
            writer.add_key_value(name, bytes(field.parts[field.data[0]]), GGUFValueType.STRING)
        else:
            val = field.parts[field.data[0]][0].item()
            if vtype == GGUFValueType.BOOL:
                val = bool(val)
            writer.add_key_value(name, val, vtype)
        n += 1
    return n


def load_tensors(ckpt_path: str, embd_dtype):
    """Return (tensors, metadata). metadata is the safetensors header dict (may be empty)."""
    from safetensors import safe_open

    out = {}
    with safe_open(ckpt_path, framework="pt") as f:
        meta = dict(f.metadata() or {})
        for name in f.keys():
            t = f.get_tensor(name)
            if name == "d2t":
                arr = t.to(torch.int64).numpy()
            elif name == "token_embd.weight":
                arr = t.to(torch.float32).numpy().astype(embd_dtype)
            elif t.ndim == 1:
                arr = t.to(torch.float32).numpy()
            else:
                arr = t.to(torch.float32).numpy().astype(np.float16)
            out[name] = np.ascontiguousarray(arr)
    return out, meta


def resolve_target_layers(meta: dict, override: str) -> list:
    """--target-layers wins; then the checkpoint metadata; then the DESIGN default."""
    if override:
        layers = [int(x) for x in override.split(",")]
        src = "--target-layers"
    elif meta.get("target_layers"):
        layers = [int(x) for x in meta["target_layers"].split(",")]
        src = "checkpoint metadata"
    else:
        layers = list(HPARAMS["target_layers"])
        src = "default (no metadata)"
    if len(layers) != 3:
        sys.exit(f"[gguf] ERROR: target_layers must have 3 entries, got {layers} from {src}")
    print(f"[gguf] target_layers = {layers} ({src})")
    return layers


def main():
    ap = argparse.ArgumentParser(description="EAGLE-3 checkpoint -> GGUF (Bonsai 2 27B)")
    ap.add_argument("ckpt")
    ap.add_argument("donor")
    ap.add_argument("out")
    ap.add_argument("--embd-type", choices=["f16", "f32"], default="f16")
    ap.add_argument("--name", default="Bonsai-2-27B-EAGLE3")
    ap.add_argument("--target-layers", default="", help="override, e.g. 20,48,62 (default: checkpoint metadata, else 6,34,62)")
    args = ap.parse_args()

    embd_dtype = np.float16 if args.embd_type == "f16" else np.float32
    print(f"[gguf] reading {args.ckpt}")
    tensors, meta = load_tensors(args.ckpt, embd_dtype)
    if meta:
        print(f"[gguf] checkpoint metadata: {meta}")
    target_layers = resolve_target_layers(meta, args.target_layers)
    missing = [k for k in EXPECTED_SHAPES if k not in tensors]
    if missing:
        sys.exit(f"[gguf] ERROR: checkpoint lacks tensors: {missing}")
    for k, shape in EXPECTED_SHAPES.items():
        if tuple(tensors[k].shape) != shape:
            sys.exit(f"[gguf] ERROR: {k} has shape {tuple(tensors[k].shape)}, expected {shape}")
    extra = [k for k in tensors if k not in EXPECTED_SHAPES]
    if extra:
        print(f"[gguf] note: extra tensors ignored: {extra}")

    writer = GGUFWriter(args.out, ARCH)
    writer.add_architecture()
    writer.add_name(args.name)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    writer.add_block_count(HPARAMS["block_count"])
    writer.add_context_length(HPARAMS["context_length"])
    writer.add_embedding_length(HPARAMS["embedding_length"])
    writer.add_feed_forward_length(HPARAMS["feed_forward_length"])
    writer.add_head_count(HPARAMS["head_count"])
    writer.add_head_count_kv(HPARAMS["head_count_kv"])
    writer.add_key_length(HPARAMS["key_length"])
    writer.add_value_length(HPARAMS["value_length"])
    writer.add_rope_dimension_count(HPARAMS["rope_dimension_count"])
    writer.add_rope_freq_base(HPARAMS["rope_freq_base"])
    writer.add_layer_norm_rms_eps(HPARAMS["rms_eps"])
    writer.add_vocab_size(HPARAMS["vocab_size"])
    writer.add_key_value(f"{ARCH}.target_layers", target_layers, GGUFValueType.ARRAY, sub_type=GGUFValueType.INT32)
    writer.add_uint32(f"{ARCH}.target_hidden_size", HPARAMS["target_hidden_size"])
    writer.add_bool(f"{ARCH}.norm_before_fc", HPARAMS["norm_before_fc"])
    writer.add_bool(f"{ARCH}.norm_before_residual", HPARAMS["norm_before_residual"])
    n_tok = copy_tokenizer_keys(writer, args.donor)
    print(f"[gguf] copied {n_tok} tokenizer.* keys from {os.path.basename(args.donor)}")

    for k in EXPECTED_SHAPES:
        writer.add_tensor(k, tensors[k])
        print(f"[gguf] + {k:28s} {str(tensors[k].dtype):8s} {tuple(tensors[k].shape)}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    print(f"[gguf] wrote {args.out} ({os.path.getsize(args.out) / 1024**3:.2f} GiB)")

    # verification: read the file back
    r = GGUFReader(args.out)
    print("[gguf] KV:")
    for name, field in r.fields.items():
        vt = field.types[0]
        if vt == GGUFValueType.ARRAY:
            sub = field.types[1].name
            if field.types[1] == GGUFValueType.STRING:
                head = [bytes(field.parts[i]).decode("utf-8", "replace") for i in field.data[:3]]
            else:
                head = [field.parts[i][0].item() for i in field.data[:8]]
            print(f"  {name} = ARRAY[{sub}] n={len(field.data)} {head}{' ...' if len(field.data) > len(head) else ''}")
        elif vt == GGUFValueType.STRING:
            s = bytes(field.parts[field.data[0]]).decode("utf-8", "replace")
            if len(s) > 60:
                s = s[:57] + "..."
            print(f"  {name} = {s!r}")
        else:
            print(f"  {name} = {field.parts[field.data[0]][0].item()} ({vt.name})")
    print("[gguf] tensors:")
    for t in r.tensors:
        print(f"  {t.name:28s} {t.tensor_type.name:5s} ne={list(int(x) for x in t.shape)} n={t.n_elements}")


if __name__ == "__main__":
    main()
