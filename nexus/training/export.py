"""
Nexus-Titans: Weight Exporter for Mojo Inference Engine
Packs ternary weights {-1, 0, 1} into 2-bit binary representations and exports
to .nexus binary file readable by Mojo with zero overhead.
"""

import os
import struct
import torch
from typing import Dict, Any

MAGIC_HEADER = b"NEXUS01\x00"

def pack_ternary_weights(w_int8: torch.Tensor) -> bytes:
    """
    Packs a 2D int8 tensor with values in {-1, 0, 1} into 2-bit packed bytes.
    4 weights per byte:
    0b00 = 0
    0b01 = +1
    0b10 = -1
    """
    flat = w_int8.flatten().cpu().tolist()
    num_weights = len(flat)
    num_bytes = (num_weights + 3) // 4
    byte_arr = bytearray(num_bytes)

    for i, val in enumerate(flat):
        byte_idx = i // 4
        shift = (i % 4) * 2
        code = 0
        if val == 1:
            code = 1
        elif val == -1:
            code = 2
        byte_arr[byte_idx] |= (code << shift)

    return bytes(byte_arr)

def extract_ternary_weights(layer: torch.nn.Module):
    if hasattr(layer, "get_ternary_weights"):
        return layer.get_ternary_weights()
    w = layer.weight.detach()
    w_scale = w.abs().mean().clamp(min=1e-5)
    w_norm = w / w_scale
    w_quant = torch.clamp(torch.round(w_norm), -1.0, 1.0).to(torch.int8)
    return w_quant, float(w_scale.item())

def export_model_to_nexus_binary(model: torch.nn.Module, output_path: str):
    """
    Exports a trained NexusTitansLM model to a portable .nexus binary file.
    """
    print(f"[Exporter] Packing and exporting Nexus-Titans model to {output_path}...")
    
    linear_layers = {
        "titans.q_proj": model.titans_memory.q_proj,
        "titans.k_proj": model.titans_memory.k_proj,
        "titans.v_proj": model.titans_memory.v_proj,
        "titans.out_proj": model.titans_memory.out_proj,
        "ffn.gate_proj": model.ffn.gate_proj,
        "ffn.up_proj": model.ffn.up_proj,
        "ffn.down_proj": model.ffn.down_proj,
        "lm_head": model.lm_head
    }

    with open(output_path, "wb") as f:
        # 1. Header
        f.write(MAGIC_HEADER)
        f.write(struct.pack("<IIII", model.vocab_size, model.d_model, model.titans_memory.d_mem, model.ffn.gate_proj.out_features))

        # 2. Token Embeddings (FP32)
        emb_data = model.token_embeddings.weight.detach().cpu().float().numpy().tobytes()
        f.write(struct.pack("<I", len(emb_data)))
        f.write(emb_data)

        # 3. Packed BitNet Linear Layers
        f.write(struct.pack("<I", len(linear_layers)))
        total_orig_bytes = 0
        total_packed_bytes = 0

        for name, layer in linear_layers.items():
            w_quant, scale = extract_ternary_weights(layer)
            packed_bytes = pack_ternary_weights(w_quant)
            out_dim, in_dim = w_quant.shape

            total_orig_bytes += out_dim * in_dim * 4
            total_packed_bytes += len(packed_bytes)

            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<H", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<IIf", in_dim, out_dim, scale))
            f.write(struct.pack("<I", len(packed_bytes)))
            f.write(packed_bytes)

    compression_ratio = total_orig_bytes / max(1, total_packed_bytes)
    file_size_kb = os.path.getsize(output_path) / 1024
    print(f"[Exporter] Successfully exported to {output_path} ({file_size_kb:.2f} KB)")
    print(f"[Exporter] Weight Compression: {total_orig_bytes/1024:.1f} KB -> {total_packed_bytes/1024:.1f} KB ({compression_ratio:.1f}x reduction!)")
