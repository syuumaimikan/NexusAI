"""
Nexus-Titans: Multi-Domain Japanese Pre-training & Conversational SFT
Trains Nexus-Titans model on Japanese Wikipedia, Aozora Bunko literature,
2channel web dialogues, and rich agent instruction trajectories.
"""

import os
import sys
import math
import random
import json
import torch
import torch.optim as optim
from typing import List, Dict

workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from nexus.tokenizer.japanese_bpe import JapaneseBPETokenizer
from nexus.training.model_pt import NexusTitansLM
from nexus.training.export import export_model_to_nexus_binary
from nexus.data.curate_datasets import CONVERSATIONAL_SFT_DATA

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Using compute device: {device}")

    # 1. Load Tokenizer
    model_prefix = os.path.join(workspace_root, "nexus", "tokenizer", "nexus_ja_bpe")
    tok = JapaneseBPETokenizer(model_prefix=model_prefix)
    tok.load()
    vocab_size = tok.sp.get_piece_size()
    print(f"[Training] Loaded Tokenizer with vocab_size={vocab_size}")

    # 2. Prepare Multi-Domain Training Samples
    data_dir = os.path.join(workspace_root, "nexus", "data")
    sft_jsonl = os.path.join(data_dir, "nexus_dialogue_sft.jsonl")
    
    sft_samples: List[List[int]] = []
    
    # 2.1 High-priority agent instruction trajectories (Chit-chat, Persona, Coding, Literature, Tool-use)
    for _ in range(25):
        for item in CONVERSATIONAL_SFT_DATA:
            u = item["user"].strip()
            a = item["assistant"].strip()
            text = f"<user>{u}</user>\n<assistant>{a}</assistant>"
            ids = tok.encode(text)
            if len(ids) > 2:
                sft_samples.append(ids)

    # 2.2 Add curated multi-domain dialogue pairs (2channel, Wikipedia, Web)
    if os.path.exists(sft_jsonl):
        count = 0
        with open(sft_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    u = obj.get("user", "").strip()
                    a = obj.get("assistant", "").strip()
                    if u and a and not any(k in u for k in ["<user>", "<assistant>"]):
                        thought = "質問や対話の文脈を理解し、適切に回答します。"
                        text = f"<user>{u}</user>\n<assistant><think>{thought}</think>\n{a}</assistant>"
                        ids = tok.encode(text)
                        if 10 <= len(ids) <= 140:
                            sft_samples.append(ids)
                            count += 1
                        if count >= 100:
                            break
                except Exception:
                    pass
        print(f"[Training] Loaded {count} additional multi-domain dialogue pairs from {sft_jsonl}", flush=True)

    print(f"[Training] Total prepared SFT samples: {len(sft_samples)}", flush=True)

    # 3. Model setup (BitNet b1.58 + Google Titans LTM Multi-Layer Architecture)
    d_model = 256
    d_mem = 128
    d_ffn = 512
    n_layers = 2

    model = NexusTitansLM(
        vocab_size=vocab_size,
        d_model=d_model,
        d_mem=d_mem,
        d_ffn=d_ffn,
        n_layers=n_layers
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Training] Model initialized with {total_params:,} parameters ({total_params * 4 / (1024*1024):.2f} MB in FP32).", flush=True)

    # 4. Optimizer & Schedule
    optimizer = optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-2)
    epochs = 12
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    # 5. Training loop
    print(f"\n[Training] Starting BitNet b1.58 + Titans LTM Optimization for {epochs} epochs...", flush=True)

    model.train()
    step = 0

    for epoch in range(1, epochs + 1):
        random.shuffle(sft_samples)
        epoch_loss = 0.0
        epoch_surprise = 0.0

        for seq in sft_samples:
            step += 1
            x_tensor = torch.tensor([seq[:-1]], dtype=torch.long, device=device)
            y_tensor = torch.tensor([seq[1:]], dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits, total_loss, surprise_loss = model(x_tensor, targets=y_tensor)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_surprise += surprise_loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(sft_samples)
        avg_surprise = epoch_surprise / len(sft_samples)
        ppl = math.exp(min(10.0, avg_loss))

        print(f"Epoch {epoch:2d}/{epochs} | Avg Loss: {avg_loss:.4f} | Surprise Loss: {avg_surprise:.4f} | Perplexity: {ppl:.2f} | LR: {scheduler.get_last_lr()[0]:.6f}", flush=True)

    # 6. Save Checkpoint
    checkpoint_dir = os.path.join(workspace_root, "nexus", "training")
    ckpt_path = os.path.join(checkpoint_dir, "nexus_checkpoint.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "d_model": d_model,
        "d_mem": d_mem,
        "d_ffn": d_ffn,
        "n_layers": n_layers
    }, ckpt_path)
    print(f"\n[Training] Saved PyTorch SFT checkpoint to {ckpt_path}", flush=True)

    # 7. Export to Mojo Binary Format (.nexus)
    engine_dir = os.path.join(workspace_root, "nexus", "engine")
    os.makedirs(engine_dir, exist_ok=True)
    binary_path = os.path.join(engine_dir, "nexus_titans.nexus")
    export_model_to_nexus_binary(model, binary_path)

if __name__ == "__main__":
    train()
