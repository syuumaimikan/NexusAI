#!/usr/bin/env python3
"""
Nexus-Titans: Interactive CLI (Claude Code & ChatGPT Style Terminal UI)
Provides Chat, Autonomous Coding, and Research capabilities powered by Mojo and Titans LTM.
"""

import sys
import os

workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from nexus.agent.nexus_agent import NexusAgent

# Terminal ANSI colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

def print_banner():
    banner = f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════════════════════════════╗
║                       ⚡ Nexus-Titans Terminal ⚡                       ║
║        Next-Gen Japanese TinyLLM (Google Titans LTM + BitNet b1.58)      ║
║        Powered by Mojo 1.0 & NVIDIA RTX 5060 Hardware Acceleration       ║
╚══════════════════════════════════════════════════════════════════════════╝{RESET}
{DIM}Commands: /chat (対話) | /code (コーディング) | /research (調査) | /memory (記憶確認) | /exit{RESET}
"""
    print(banner)

def main():
    workspace = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    agent = NexusAgent(workspace_root=workspace)
    current_mode = "code"

    if len(sys.argv) > 1:
        # Non-interactive one-shot command execution
        user_query = " ".join(sys.argv[1:])
        print(f"{BOLD}[Query]:{RESET} {user_query}\n")
        resp = agent.execute_turn(user_query, mode=current_mode)
        print(resp)
        return

    print_banner()
    while True:
        try:
            mode_badge = f"{GREEN}[{current_mode.upper()}]{RESET}" if current_mode == "code" else f"{BLUE}[{current_mode.upper()}]{RESET}"
            prompt_str = f"\n{mode_badge} {BOLD}Nexus >>{RESET} "
            user_input = input(prompt_str).strip()

            if not user_input:
                continue

            if user_input in ["/exit", "exit", "quit"]:
                print(f"{YELLOW}Nexus-Titans を終了します。お疲れ様でした。{RESET}")
                break

            elif user_input == "/chat":
                current_mode = "chat"
                print(f"{CYAN}💬 対話モードに切り替えました (ChatGPT equivalent){RESET}")
                continue

            elif user_input.startswith("/chat "):
                current_mode = "chat"
                user_input = user_input[6:].strip()

            elif user_input == "/code":
                current_mode = "code"
                print(f"{GREEN}💻 コーディング・エージェントモードに切り替えました (Claude Code equivalent){RESET}")
                continue

            elif user_input.startswith("/code "):
                current_mode = "code"
                user_input = user_input[6:].strip()

            elif user_input == "/research":
                current_mode = "research"
                print(f"{MAGENTA}🔍 Web・知識リサーチモードに切り替えました{RESET}")
                continue

            elif user_input.startswith("/research "):
                current_mode = "research"
                user_input = user_input[10:].strip()

            elif user_input == "/memory":
                mem_count = len(agent.long_term_memory)
                print(f"{YELLOW}🧠 [Titans LTM Status]{RESET}")
                print(f" - 固定メモリスロットサイズ: 64 x 64 Float32 (16 KB)")
                print(f" - 保持記憶エントリ数: {mem_count}")
                for i, m in enumerate(agent.long_term_memory[-3:]):
                    print(f"   [{i+1}] {m[:100]}...")
                continue

            elif user_input == "/help":
                print(f"""
{BOLD}利用可能なコマンド:{RESET}
  /code      - Claude Code風のコーディング＆シェル実行モード
  /chat      - ChatGPT風の自然な日本語対話モード
  /research  - Web検索・Wiki知識探索モード
  /memory    - Titans長期記憶のステータス確認
  /exit      - 終了
""")
                continue

            # Execute agent turn
            print(f"{DIM}Thinking & executing with Mojo/Titans...{RESET}")
            response = agent.execute_turn(user_input, mode=current_mode)
            print(f"\n{response}")

        except (KeyboardInterrupt, EOFError):
            print(f"\n{YELLOW}終了します。{RESET}")
            break

if __name__ == "__main__":
    main()
