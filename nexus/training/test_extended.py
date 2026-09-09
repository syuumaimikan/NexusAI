import os
import sys

workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from nexus.agent.nexus_agent import NexusAgent

def evaluate_extended():
    agent = NexusAgent(workspace_root=workspace_root)
    
    test_prompts = [
        "吾輩は猫であるの有名な冒頭の一節を教えて",
        "自作PCで一番予算をかけるべきパーツはどこ？",
        "Titansをgrepして",
        "人工知能についてWeb検索して"
    ]
    
    print("=" * 60)
    print("Nexus-Titans Extended Capabilities Evaluation")
    print("=" * 60)
    
    for p in test_prompts:
        print(f"\n[USER]: {p}")
        resp = agent.execute_turn(p)
        print(f"[ASSISTANT]:\n{resp}")
        print("-" * 60)

if __name__ == "__main__":
    evaluate_extended()
