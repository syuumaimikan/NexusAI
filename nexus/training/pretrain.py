"""
Nexus-Titans: Scalable Japanese Wikipedia Causal LM Pre-training
Trains BitNet b1.58 + Google Titans LTM on Japanese Wikipedia, literature, and encyclopedic knowledge.
"""

import os
import sys
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
from tqdm import tqdm

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from nexus.tokenizer.japanese_bpe import JapaneseBPETokenizer
from nexus.training.model_pt import NexusTitansLM

def load_and_tokenize_corpus(tok: JapaneseBPETokenizer, max_chars: int = 5_000_000, seq_len: int = 128) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Loads Wikipedia and multi-domain texts, tokenizes, and packs into causal LM (x, y) chunks.
    Prioritizes structured Wikipedia encyclopedia articles (history, science, geography).
    """
    data_dir = os.path.join(WORKSPACE_ROOT, "nexus", "data")
    wiki_corpus_path = os.path.join(data_dir, "ja_wikipedia_corpus.txt")
    massive_pretrain_path = os.path.join(data_dir, "nexus_massive_pretrain.txt")

    all_tokens: List[int] = []

    # 1. High-priority Wikipedia core articles (History, eras, science)
    if os.path.exists(wiki_corpus_path):
        print(f"[Pretrain] Loading high-priority Wikipedia encyclopedia articles from {wiki_corpus_path}...")
        with open(wiki_corpus_path, "r", encoding="utf-8") as f:
            wiki_text = f.read()
        tokens = tok.encode(wiki_text)
        print(f"[Pretrain] Wikipedia core tokens: {len(tokens):,}")
        # Repeat key knowledge articles 3 times to ensure strong embedding
        for _ in range(3):
            all_tokens.extend(tokens)

    # 2. General massive corpus (Wikipedia 10,000 articles + Aozora Bunko)
    if os.path.exists(massive_pretrain_path):
        print(f"[Pretrain] Streaming general corpus from {massive_pretrain_path} (up to {max_chars:,} chars)...")
        read_chars = 0
        with open(massive_pretrain_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                t = tok.encode(line.strip())
                all_tokens.extend(t)
                read_chars += len(line)
                if read_chars >= max_chars:
                    break
        print(f"[Pretrain] Loaded {read_chars:,} characters from massive corpus.")

    print(f"[Pretrain] Total token stream length: {len(all_tokens):,} tokens.")

    # 3. Pack into causal LM chunks: length = seq_len + 1 (x: 0..seq_len-1, y: 1..seq_len)
    step = seq_len # Non-overlapping sequential chunks for efficient, full-coverage pre-training
    samples = []
    chunk_size = seq_len + 1
    for i in range(0, len(all_tokens) - chunk_size, step):
        chunk = all_tokens[i:i + chunk_size]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        samples.append((x, y))

    print(f"[Pretrain] Generated {len(samples):,} causal sequence chunks (seq_len={seq_len}).")
    return samples

def pretrain():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        print(f"[Pretrain] Hardware accelerator: {device} ({gpu_name}, {vram_mb:.0f} MB VRAM)")
    else:
        print(f"[Pretrain] Compute device: {device}")

    # 1. Tokenizer
    model_prefix = os.path.join(WORKSPACE_ROOT, "nexus", "tokenizer", "nexus_ja_bpe")
    tok = JapaneseBPETokenizer(model_prefix=model_prefix)
    tok.load()
    vocab_size = tok.sp.get_piece_size()
    print(f"[Pretrain] Tokenizer loaded: vocab_size={vocab_size}")

    # 2. Data Preparation
    seq_len = 128
    samples = load_and_tokenize_corpus(tok, max_chars=1_500_000, seq_len=seq_len)
    random.shuffle(samples)

    # 3. Initialize Model Architecture (BitNet b1.58 + Google Titans LTM)
    d_model = 256
    d_mem = 128
    d_ffn = 512
    n_layers = 2

    model = NexusTitansLM(
        vocab_size=vocab_size,
        d_model=d_model,
        d_mem=d_mem,
        d_ffn=d_ffn,
        n_layers=n_layers,
        max_seq_len=2048
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Pretrain] Model initialized with {total_params:,} parameters ({total_params * 4 / (1024*1024):.2f} MB FP32).")

    # 4. Training Hyperparameters
    batch_size = 16
    accum_steps = 2
    epochs = 2
    optimizer = optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    checkpoint_dir = os.path.join(WORKSPACE_ROOT, "nexus", "training")
    os.makedirs(checkpoint_dir, exist_ok=True)
    pretrain_ckpt_path = os.path.join(checkpoint_dir, "nexus_pretrained.pt")

    print(f"\n[Pretrain] Starting Wikipedia Causal LM Pre-training for {epochs} epochs (Batch={batch_size}, Accum={accum_steps})...")

    model.train()
    total_samples = len(samples)

    for epoch in range(1, epochs + 1):
        random.shuffle(samples)
        epoch_loss = 0.0
        epoch_surprise = 0.0
        optimizer.zero_grad()

        num_batches = math.ceil(total_samples / batch_size)
        pbar = tqdm(range(num_batches), desc=f"Pretrain Epoch {epoch:2d}/{epochs}", unit="batch", dynamic_ncols=True)

        for b_idx in pbar:
            start_idx = b_idx * batch_size
            end_idx = min(start_idx + batch_size, total_samples)
            batch_slice = samples[start_idx:end_idx]

            x_batch = torch.stack([s[0] for s in batch_slice]).to(device)
            y_batch = torch.stack([s[1] for s in batch_slice]).to(device)

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits, total_loss, surprise_loss = model(x_batch, targets=y_batch)
                scaled_loss = total_loss / accum_steps

            scaler.scale(scaled_loss).backward()

            if (b_idx + 1) % accum_steps == 0 or (b_idx + 1) == num_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            loss_val = total_loss.item()
            surprise_val = surprise_loss.item()
            epoch_loss += loss_val
            epoch_surprise += surprise_val

            running_loss = epoch_loss / (b_idx + 1)
            pbar.set_postfix({
                "loss": f"{loss_val:.3f}",
                "avg": f"{running_loss:.3f}",
                "ppl": f"{math.exp(min(8.0, running_loss)):.1f}",
                "lr": f"{scheduler.get_last_lr()[0]:.5f}"
            })

        scheduler.step()
        avg_loss = epoch_loss / num_batches
        avg_surprise = epoch_surprise / num_batches
        ppl = math.exp(min(8.0, avg_loss))
        print(f"Pretrain Epoch {epoch:2d}/{epochs} Complete | Avg Loss: {avg_loss:.4f} | Surprise: {avg_surprise:.4f} | PPL: {ppl:.2f}")

        # Save checkpoint per epoch
        torch.save({
            "model_state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "d_model": d_model,
            "d_mem": d_mem,
            "d_ffn": d_ffn,
            "n_layers": n_layers,
            "epoch": epoch
        }, pretrain_ckpt_path)
        print(f"[Pretrain] Saved checkpoint for epoch {epoch} to {pretrain_ckpt_path}")

if __name__ == "__main__":
    pretrain()
