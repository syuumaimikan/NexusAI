"""
Nexus-Titans: PyTorch Training Architecture
Implements BitNet b1.58 ternary quantization with Straight-Through Estimator (STE)
and Google Titans Neural Long-Term Memory (LTM) with test-time surprise learning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class BitLinear(nn.Linear):
    """
    BitNet b1.58 Ternary Linear Layer with Straight-Through Estimator (STE).
    Weights are quantized to {-1, 0, 1} during forward pass, allowing
    floating point gradients to flow back during backward pass.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Weight quantization to ternary {-1, 0, 1}
        w_scale = self.weight.abs().mean().clamp(min=1e-5)
        w_normalized = self.weight / w_scale
        w_quant = torch.clamp(torch.round(w_normalized), -1.0, 1.0)
        # STE: forward uses quantized weights, backward gradients pass through to continuous weights
        w_ste = self.weight + (w_quant - self.weight).detach()

        # 2. Linear projection
        y = F.linear(x, w_ste) * w_scale
        if self.bias is not None:
            y = y + self.bias
        return y

    def get_ternary_weights(self) -> torch.Tensor:
        """Returns discretized {-1, 0, 1} integer tensor and scale factor."""
        w_scale = self.weight.abs().mean().clamp(min=1e-5)
        w_normalized = self.weight / w_scale
        w_quant = torch.clamp(torch.round(w_normalized), -1.0, 1.0).to(torch.int8)
        return w_quant, float(w_scale.item())

class TitansNeuralMemory(nn.Module):
    """
    Google Titans: Neural Long-Term Memory (LTM) Module.
    Maintains a persistent/adaptive neural memory matrix M updated via surprise gradient descent.
    """
    def __init__(self, d_model: int, d_mem: int, lr: float = 0.05, decay: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.d_mem = d_mem
        self.lr = lr
        self.decay = decay

        # Projections
        self.q_proj = BitLinear(d_model, d_mem)
        self.k_proj = BitLinear(d_model, d_mem)
        self.v_proj = BitLinear(d_model, d_mem)
        self.out_proj = BitLinear(d_mem, d_model)
        self.alpha_proj = nn.Linear(d_model, 1) # Dynamic forgetting gate

        # Initial memory parameter
        self.register_buffer("M", torch.zeros(d_mem, d_mem))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, seq_len, d_model]
        B, T, D = x.shape
        q = self.q_proj(x) # [B, T, d_mem]
        k = self.k_proj(x) # [B, T, d_mem]
        v = self.v_proj(x) # [B, T, d_mem]

        outputs = []
        total_surprise_loss = 0.0
        current_M = self.M.clone()

        for t in range(T):
            q_t = q[:, t] # [B, d_mem]
            k_t = k[:, t] # [B, d_mem]
            v_t = v[:, t] # [B, d_mem]
            x_t = x[:, t]

            # 1. Retrieve from memory: y = M * q
            y_mem = torch.matmul(q_t, current_M.T) # [B, d_mem]
            out_t = self.out_proj(y_mem) # [B, d_model]
            outputs.append(out_t)

            # 2. Associative memory prediction and surprise metric
            v_hat = torch.matmul(k_t, current_M.T) # [B, d_mem]
            surprise_t = F.mse_loss(v_hat, v_t)
            total_surprise_loss += surprise_t

            # 3. Dynamic forgetting gate
            alpha_t = torch.sigmoid(self.alpha_proj(x_t)).mean()

            # 4. Online update of memory matrix
            err_t = (v_t - v_hat).mean(dim=0, keepdim=True) # [1, d_mem]
            k_mean = k_t.mean(dim=0, keepdim=True) # [1, d_mem]
            grad_step = self.lr * torch.matmul(err_t.T, k_mean) # [d_mem, d_mem]
            current_M = (1.0 - alpha_t * self.decay) * current_M + grad_step

        y_all = torch.stack(outputs, dim=1) # [B, T, d_model]
        avg_surprise = total_surprise_loss / max(1, T)
        return y_all, avg_surprise

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.gate_proj = BitLinear(d_model, d_ffn)
        self.up_proj = BitLinear(d_model, d_ffn)
        self.down_proj = BitLinear(d_ffn, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class NexusTitansLM(nn.Module):
    """
    Complete Nexus-Titans Causal Language Model.
    """
    def __init__(self, vocab_size: int, d_model: int = 128, d_mem: int = 64, d_ffn: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.titans_memory = TitansNeuralMemory(d_model, d_mem)
        self.ffn = SwiGLUFFN(d_model, d_ffn)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.lm_head = BitLinear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        x = self.token_embeddings(input_ids) # [B, T, d_model]
        
        # Titans LTM block
        h_norm1 = self.norm1(x)
        mem_out, surprise_loss = self.titans_memory(h_norm1)
        x = x + mem_out

        # SwiGLU FFN block
        h_norm2 = self.norm2(x)
        x = x + self.ffn(h_norm2)

        logits = self.lm_head(x) # [B, T, vocab_size]

        loss = None
        if targets is not None:
            # Language modeling cross-entropy loss + surprise regularization
            ce_loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            loss = ce_loss + 0.1 * surprise_loss

        return logits, loss, surprise_loss
