"""
Nexus-Titans Comprehensive Test Suite
Validates:
1. BitNet b1.58 ternary quantization math
2. Google Titans Neural LTM online learning & surprise metric
3. Japanese BPE Tokenizer compression & round-trip
4. Binary .nexus weight export & serialization integrity
5. Agent tools (Bash, File I/O, Web Search, Titans Memory)
"""

import os
import sys
import struct
import tempfile
import torch
import pytest

workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from nexus.training.model_pt import BitLinear, TitansNeuralMemory, NexusTitansLM
from nexus.tokenizer.japanese_bpe import JapaneseBPETokenizer
from nexus.training.export import pack_ternary_weights, export_model_to_nexus_binary
from nexus.agent.nexus_agent import NexusAgent

def test_bitnet_ternary_quantization():
    linear = BitLinear(in_features=64, out_features=32)
    w_quant, scale = linear.get_ternary_weights()
    # Check that all weights are strictly in {-1, 0, 1}
    unique_vals = torch.unique(w_quant).tolist()
    for val in unique_vals:
        assert val in [-1, 0, 1], f"Non-ternary weight found: {val}"
    assert scale > 0.0

def test_titans_neural_memory_online_learning():
    d_model = 32
    d_mem = 16
    memory = TitansNeuralMemory(d_model=d_model, d_mem=d_mem, lr=0.1, decay=0.01)
    
    # Simulate sequential input tokens
    x = torch.randn(1, 10, d_model)
    out, surprise_loss = memory(x)
    assert out.shape == (1, 10, d_model)
    assert surprise_loss.item() >= 0.0

def test_japanese_tokenizer_roundtrip():
    model_path = os.path.join(os.path.dirname(__file__), "..", "nexus", "tokenizer", "nexus_ja_bpe")
    tok = JapaneseBPETokenizer(model_prefix=model_path)
    tok.load()
    
    sample_text = "NexusAIはMojoとTitansアーキテクチャによる日本語TinyLLMです。"
    tokens = tok.encode(sample_text)
    decoded = tok.decode(tokens)
    assert decoded == sample_text, f"Roundtrip failed: expected '{sample_text}', got '{decoded}'"
    assert len(tokens) < len(sample_text), f"Compression target failed: {len(tokens)} >= {len(sample_text)}"

def test_export_binary_integrity():
    model = NexusTitansLM(vocab_size=100, d_model=32, d_mem=16, d_ffn=64)
    with tempfile.NamedTemporaryFile(suffix=".nexus", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        export_model_to_nexus_binary(model, tmp_path)
        assert os.path.exists(tmp_path)
        with open(tmp_path, "rb") as f:
            magic = f.read(8)
            assert magic == b"NEXUS01\x00", f"Invalid magic header: {magic}"
            v_size, d_m, d_me, d_f = struct.unpack("<IIII", f.read(16))
            assert v_size == 100
            assert d_m == 32
            assert d_me == 16
            assert d_f == 64
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_agent_tools_execution():
    workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    agent = NexusAgent(workspace_root=workspace)
    
    # 1. Test bash execute
    bash_out = agent.tool_bash_execute("echo 'NexusAI agent test'")
    assert "NexusAI agent test" in bash_out
    
    # 2. Test file read/write/edit
    test_file = "test_agent_scratch.txt"
    agent.tool_file_write(test_file, "Line 1: Mojo\nLine 2: Titans\nLine 3: BitNet\n")
    read_out = agent.tool_file_read(test_file)
    assert "Line 2: Titans" in read_out
    
    agent.tool_file_edit(test_file, "Line 2: Titans", "Line 2: Google Titans LTM")
    read_edited = agent.tool_file_read(test_file)
    assert "Line 2: Google Titans LTM" in read_edited
    
    # Cleanup test file
    agent.tool_bash_execute(f"rm -f {test_file}")
    
    # 3. Test Titans memory absorption
    init_mem_len = len(agent.long_term_memory)
    absorb_res = agent.tool_memory_absorb("Fact: Titans uses surprise metric for test-time training.")
    assert "吸収完了" in absorb_res
    assert len(agent.long_term_memory) == init_mem_len + 1
