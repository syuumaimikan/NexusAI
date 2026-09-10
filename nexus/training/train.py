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
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from nexus.tokenizer.japanese_bpe import JapaneseBPETokenizer
from nexus.training.model_pt import NexusTitansLM
from nexus.training.export import export_model_to_nexus_binary
from nexus.data.curate_datasets import CONVERSATIONAL_SFT_DATA

def prepare_epoch_batches(samples: List[Tuple[List[int], List[int]]], batch_size: int, pad_id: int = 3) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Clusters samples by similar sequence lengths to minimize padding overhead,
    pads targets with -100 (ignored by cross entropy), and shuffles batches.
    Each sample is a tuple: (input_ids, target_ids_with_mask).
    """
    sorted_samples = sorted(samples, key=lambda s: len(s[0]) + random.randint(-2, 2))
    batches = []
    for i in range(0, len(sorted_samples), batch_size):
        chunk = sorted_samples[i:i + batch_size]
        max_len = max(len(s[0]) for s in chunk)
        batch_x = []
        batch_y = []
        for x, y in chunk:
            pad_len = max_len - len(x)
            batch_x.append(x + [pad_id] * pad_len)
            batch_y.append(y + [-100] * pad_len)
        batches.append((
            torch.tensor(batch_x, dtype=torch.long),
            torch.tensor(batch_y, dtype=torch.long)
        ))
    random.shuffle(batches)
    return batches

def train():
    print("[DEBUG] Entered train() function", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        print(f"[Training] Using compute device: {device} ({gpu_name}, {vram_mb:.0f} MB VRAM)", flush=True)
    else:
        print(f"[Training] Using compute device: {device}", flush=True)

    # 1. Load Tokenizer
    model_prefix = os.path.join(workspace_root, "nexus", "tokenizer", "nexus_ja_bpe")
    tok = JapaneseBPETokenizer(model_prefix=model_prefix)
    tok.load()
    vocab_size = tok.sp.get_piece_size()
    print(f"[Training] Loaded Tokenizer with vocab_size={vocab_size}", flush=True)

    # 2. Prepare Multi-Domain Training Samples with Prompt Loss Masking
    data_dir = os.path.join(workspace_root, "nexus", "data")
    sft_jsonl = os.path.join(data_dir, "nexus_dialogue_sft.jsonl")
    
    sft_samples: List[Tuple[List[int], List[int]]] = []
    
    def create_masked_sample(u_str: str, a_str: str) -> Optional[Tuple[List[int], List[int]]]:
        prompt_text = f"<user>{u_str}</user>\n<assistant>"
        full_text = f"<user>{u_str}</user>\n<assistant>{a_str}</assistant>"
        prompt_ids = tok.encode(prompt_text)
        full_ids = tok.encode(full_text)
        if len(full_ids) <= len(prompt_ids):
            return None
        x = full_ids[:-1]
        y = full_ids[1:]
        # Mask prompt tokens with -100 so loss is only computed on assistant answer tokens
        prompt_len = len(prompt_ids)
        mask_len = min(prompt_len - 1, len(y))
        y_masked = [-100] * mask_len + y[mask_len:]
        return (x, y_masked)

    # 2.1 High-priority agent instruction trajectories (Chit-chat, History, Coding, Tools)
    for _ in range(12):
        for item in CONVERSATIONAL_SFT_DATA:
            sample = create_masked_sample(item["user"].strip(), item["assistant"].strip())
            if sample is not None:
                sft_samples.append(sample)

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
                        thought = "文脈と質問の意図を把握し、的確に回答します。"
                        answer_text = f"<think>{thought}</think>\n{a}"
                        sample = create_masked_sample(u, answer_text)
                        if sample is not None and 10 <= len(sample[0]) <= 180:
                            sft_samples.append(sample)
                            count += 1
                        if count >= 300:
                            break
                except Exception:
                    pass
        print(f"[Training] Loaded {count} additional multi-domain dialogue pairs from {sft_jsonl}", flush=True)

    print(f"[Training] Total prepared SFT samples with Prompt Loss Masking: {len(sft_samples)}", flush=True)

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
    
    # 3.1 Load Pre-trained weights if available
    pretrain_ckpt = os.path.join(workspace_root, "nexus", "training", "nexus_pretrained.pt")
    if os.path.exists(pretrain_ckpt):
        print(f"[Training] Loading pre-trained Wikipedia weights from {pretrain_ckpt}...", flush=True)
        try:
            ckpt_data = torch.load(pretrain_ckpt, map_location=device)
            model.load_state_dict(ckpt_data["model_state_dict"])
            print("[Training] Pre-trained weights successfully loaded!", flush=True)
        except Exception as e:
            print(f"[Warning: Failed to load pre-trained checkpoint]: {e}", flush=True)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Training] Model ready with {total_params:,} parameters ({total_params * 4 / (1024*1024):.2f} MB in FP32).", flush=True)

    # 4. Optimizer & Schedule
    optimizer = optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1e-2)
    epochs = 8
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-5)

    # 5. Training loop
    batch_size = 12
    accum_steps = 2
    pad_id = tok.sp.pad_id() if hasattr(tok, "sp") and tok.sp else 3
    print(f"\n[Training] Starting Prompt-Masked SFT for {epochs} epochs (BatchSize={batch_size}, Accumulation={accum_steps})...", flush=True)

    model.train()

    for epoch in range(1, epochs + 1):
        batches = prepare_epoch_batches(sft_samples, batch_size=batch_size, pad_id=pad_id)
        epoch_loss = 0.0
        epoch_surprise = 0.0
        optimizer.zero_grad()

        pbar = tqdm(batches, desc=f"Epoch {epoch:2d}/{epochs}", unit="batch", dynamic_ncols=True)
        for step, (x_tensor, y_tensor) in enumerate(pbar, 1):
            x_tensor = x_tensor.to(device)
            y_tensor = y_tensor.to(device)

            logits, total_loss, surprise_loss = model(x_tensor, targets=y_tensor)
            scaled_loss = total_loss / accum_steps
            scaled_loss.backward()

            if step % accum_steps == 0 or step == len(batches):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            loss_val = total_loss.item()
            surprise_val = surprise_loss.item()
            epoch_loss += loss_val
            epoch_surprise += surprise_val

            running_avg = epoch_loss / step
            pbar.set_postfix({
                "loss": f"{loss_val:.3f}",
                "avg": f"{running_avg:.3f}",
                "ppl": f"{math.exp(min(10.0, running_avg)):.2f}",
                "lr": f"{scheduler.get_last_lr()[0]:.5f}"
            })

        scheduler.step()
        avg_loss = epoch_loss / len(batches)
        avg_surprise = epoch_surprise / len(batches)
        ppl = math.exp(min(10.0, avg_loss))

        print(f"Epoch {epoch:2d}/{epochs} Complete | Avg Loss: {avg_loss:.4f} | Surprise Loss: {avg_surprise:.4f} | Perplexity: {ppl:.2f} | LR: {scheduler.get_last_lr()[0]:.6f}", flush=True)

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
