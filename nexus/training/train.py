"""
Nexus-Titans: Japanese Pre-training & Quantization-Aware Fine-Tuning
Trains Nexus-Titans model on Japanese Wikipedia and reasoning corpora,
then exports to packed BitNet binary format.
"""

import os
import sys
import math
import torch
import torch.optim as optim
from typing import List

workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from nexus.tokenizer.japanese_bpe import JapaneseBPETokenizer
from nexus.training.model_pt import NexusTitansLM
from nexus.training.export import export_model_to_nexus_binary

def load_training_corpus() -> str:
    data_dir = os.path.join(workspace_root, "nexus", "data")
    tok_dir = os.path.join(workspace_root, "nexus", "tokenizer")
    
    texts = []
    # 1. Japanese Wikipedia corpus
    wiki_path = os.path.join(data_dir, "ja_wikipedia_corpus.txt")
    if os.path.exists(wiki_path):
        with open(wiki_path, "r", encoding="utf-8") as f:
            texts.append(f.read())
            
    # 2. Seed corpus
    seed_path = os.path.join(tok_dir, "seed_corpus.txt")
    if os.path.exists(seed_path):
        with open(seed_path, "r", encoding="utf-8") as f:
            texts.append(f.read())
            
    combined = "\n\n".join(texts)
    print(f"[Training] Loaded Japanese corpus with {len(combined):,} total characters.")
    return combined

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Using compute device: {device}")

    # 1. Tokenizer
    model_prefix = os.path.join(workspace_root, "nexus", "tokenizer", "nexus_ja_bpe")
    tok = JapaneseBPETokenizer(model_prefix=model_prefix)
    tok.load()
    vocab_size = tok.sp.get_piece_size()
    print(f"[Training] Loaded Tokenizer with vocab_size={vocab_size}")

    # 2. Tokenize text
    raw_text = load_training_corpus()
    all_tokens = tok.encode(raw_text)
    print(f"[Training] Encoded text into {len(all_tokens):,} tokens.")

    # 3. Model setup
    d_model = 128
    d_mem = 64
    d_ffn = 256
    seq_len = 64
    batch_size = 4

    model = NexusTitansLM(vocab_size=vocab_size, d_model=d_model, d_mem=d_mem, d_ffn=d_ffn).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Training] Model initialized with {total_params:,} parameters.")

    # 4. Optimizer & Schedule
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    
    # 5. Training loop
    num_steps = 40
    print(f"\n[Training] Starting QAT & Titans LTM Optimization for {num_steps} steps...")
    
    model.train()
    for step in range(1, num_steps + 1):
        # Prepare batch
        batch_inputs = []
        batch_targets = []
        for _ in range(batch_size):
            start_idx = torch.randint(0, len(all_tokens) - seq_len - 1, (1,)).item()
            chunk = all_tokens[start_idx : start_idx + seq_len + 1]
            batch_inputs.append(chunk[:-1])
            batch_targets.append(chunk[1:])

        x = torch.tensor(batch_inputs, dtype=torch.long, device=device)
        y = torch.tensor(batch_targets, dtype=torch.long, device=device)

        optimizer.zero_grad()
        logits, total_loss, surprise_loss = model(x, targets=y)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % 10 == 0:
            ppl = math.exp(min(10.0, total_loss.item()))
            print(f"Step {step:2d}/{num_steps} | Total Loss: {total_loss.item():.4f} | Surprise Loss: {surprise_loss.item():.4f} | Perplexity: {ppl:.2f}")

    # 6. Save Checkpoint
    checkpoint_dir = os.path.join(workspace_root, "nexus", "training")
    ckpt_path = os.path.join(checkpoint_dir, "nexus_checkpoint.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "d_model": d_model,
        "d_mem": d_mem,
        "d_ffn": d_ffn
    }, ckpt_path)
    print(f"\n[Training] Saved PyTorch checkpoint to {ckpt_path}")

    # 7. Export to Mojo Binary Format (.nexus)
    engine_dir = os.path.join(workspace_root, "nexus", "engine")
    os.makedirs(engine_dir, exist_ok=True)
    binary_path = os.path.join(engine_dir, "nexus_titans.nexus")
    export_model_to_nexus_binary(model, binary_path)

if __name__ == "__main__":
    train()
