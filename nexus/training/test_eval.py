import os
import sys
import torch

workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from nexus.agent.nexus_agent import NexusAgent

def evaluate():
    agent = NexusAgent(workspace_root=workspace_root)
    
    test_prompts = [
        "おはようございます",
        "こんにちは",
        "あなたは誰ですか？",
        "Pythonでフィボナッチ数を計算する関数を書いて",
        "Pythonで素数を判定する関数を書いて",
        "BitNet b1.58の仕組みとメリットを教えて",
        "CPUスペックを確認して"
    ]
    
    print("=" * 60)
    print("Nexus-Titans Live Neural Agent Evaluation")
    print("=" * 60)
    
    for p in test_prompts:
        print(f"\n[USER]: {p}")
        resp = agent.execute_turn(p)
        print(f"[ASSISTANT]:\n{resp}")
        print("-" * 60)

if __name__ == "__main__":
    evaluate()
