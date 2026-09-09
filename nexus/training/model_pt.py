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

    def get_ternary_weights(self) -> Tuple[torch.Tensor, float]:
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

        # Projections (BitNet 1.58 ternary linear)
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
    """SwiGLU Feed-Forward Network using BitNet b1.58 ternary projections."""
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.gate_proj = BitLinear(d_model, d_ffn)
        self.up_proj = BitLinear(d_model, d_ffn)
        self.down_proj = BitLinear(d_ffn, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class CausalConv1d(nn.Module):
    """
    Local Short-Term Memory via Causal Depthwise 1D Convolution.
    Implements Google Titans MAC/MAG design principle: combines local n-gram context
    with Neural Long-Term Memory (LTM) without requiring KV-cache.
    """
    def __init__(self, d_model: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, groups=d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] -> transpose for Conv1d -> [B, D, T]
        x_trans = x.transpose(1, 2)
        x_pad = F.pad(x_trans, (self.kernel_size - 1, 0)) # Causal left-padding
        y = self.conv(x_pad).transpose(1, 2)
        return y

class NexusBlock(nn.Module):
    """
    Unified Nexus Block combining:
    1. Short-Term Memory: Causal Depthwise Conv1d (O(1) local syntax context)
    2. Long-Term Memory: Google Titans Neural LTM (O(1) online test-time memory)
    3. BitNet b1.58 SwiGLU FFN: Pure ternary {-1, 0, 1} feedforward computation
    """
    def __init__(self, d_model: int, d_mem: int, d_ffn: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.local_conv = CausalConv1d(d_model, kernel_size=4)
        self.titans_memory = TitansNeuralMemory(d_model, d_mem)
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ffn)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_x = self.norm1(x)
        h_conv = self.local_conv(norm_x)
        h_mem, s = self.titans_memory(norm_x)
        x = x + h_conv + h_mem
        x = x + self.ffn(self.norm2(x))
        return x, s

class NexusTitansLM(nn.Module):
    """
    Complete Nexus-Titans Causal Language Model.
    Adheres strictly to the BitNet b1.58 design standard (Microsoft Research):
    - All hidden layer linear projections (QKV, Out, FFN Gate/Up/Down) are 1.58-bit ternary.
    - Token Embedding and LM Output Head maintain full floating-point precision for vocabulary discrimination.
    - Infinite context retention through Google Titans Test-Time Learning.
    - Local syntax and indentation retention through Causal Conv1D short-term memory.
    """
    def __init__(self, vocab_size: int, d_model: int = 256, d_mem: int = 128, d_ffn: int = 512, n_layers: int = 2, max_seq_len: int = 2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_mem = d_mem
        self.d_ffn = d_ffn
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([NexusBlock(d_model, d_mem, d_ffn) for _ in range(n_layers)])
        self.norm_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Backwards-compatible convenience references to layer 0
        self.titans_memory = self.layers[0].titans_memory
        self.ffn = self.layers[0].ffn

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = input_ids.shape
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        x = self.token_embeddings(input_ids) + self.pos_embeddings(pos)
        total_surprise = 0.0

        for layer in self.layers:
            x, s = layer(x)
            total_surprise = total_surprise + s

        x = self.norm_f(x)
        logits = self.lm_head(x) # [B, T, vocab_size]

        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            loss = ce_loss + 0.05 * total_surprise

        return logits, loss, total_surprise
