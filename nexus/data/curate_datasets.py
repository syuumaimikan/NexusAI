"""
Nexus-Titans: Reasoning & Agent Dataset Generator
Curates Japanese Chain-of-Thought (CoT) reasoning, code tasks, and Claude Code-style ReAct trajectories.
"""

import json
import os
from typing import List, Dict

AGENT_REASONING_TRAJECTORIES = [
    {
        "instruction": "このプロジェクトにMojoのベンチマークスクリプトを追加して、実行結果を確認してください。",
        "trajectory": (
            "<think>ユーザーはMojoによるベンチマークスクリプトの追加と実行を求めています。\n"
            "1. まずscratchディレクトリにベンチマークファイルを作成する。\n"
            "2. pixi run mojo run コマンドで実行する。\n"
            "3. 出力結果をパースしてユーザーに要約を報告する。</think>\n"
            "<tool_call>bash_execute(command='pixi run mojo run nexus/core/bitnet.mojo')</tool_call>\n"
            "<tool_response>Average Forward Time: 0.12 ms | Memory Compression: ~16x</tool_response>\n"
            "<think>ベンチマークが正常に実行され、0.12msの実行速度と16倍のメモリ圧縮が確認できました。結果を報告します。</think>\n"
            "MojoによるBitNet b1.58加減算カーネルのベンチマークを実行しました。結果は以下の通りです：\n"
            "- 平均順伝播時間: 0.12 ms\n"
            "- メモリ圧縮率: FP32比で約16倍（2ビットパッキング）\n"
            "浮動小数点乗算器を完全排除した超高速な加減算推論が確認できました。"
        )
    },
    {
        "instruction": "Google Titansのサプライズ度計算式と更新式を日本語で解説してください。",
        "trajectory": (
            "<think>Google Titans (Learning to Memorize at Test Time, 2024年12月) の数学的定式化を整理する。\n"
            "- サプライズ度: キーに対する記憶の予測誤差 || M_{t-1} k_t - v_t ||\n"
            "- オンライン更新: 動的忘却ゲート (1 - alpha_t) と学習率 eta_t による勾配降下法\n"
            "日本語でわかりやすく構造化して回答する。</think>\n"
            "Google Titansのコアとなる「サプライズ度」と「記憶更新」の数式は以下の通りです：\n\n"
            "1. **サプライズ度（Surprise Metric）**:\n"
            "入力キー $k_t$ に対する過去の記憶 $M_{t-1}$ の予測誤差を計算します。\n"
            "$$S_t = \\| M_{t-1} k_t - v_t \\|$$\n"
            "すでに記憶されている既知の情報であればサプライズ度は小さく、未知・新奇な情報であればサプライズ度は大きくなります。\n\n"
            "2. **オンライン記憶更新式（Test-Time Gradient Step）**:\n"
            "$$M_t = (1 - \\alpha_t) M_{t-1} + \\eta_t (v_t - M_{t-1} k_t) k_t^T$$\n"
            "- $\\alpha_t$: 動的忘却率（不要になった過去の情報を減衰）\n"
            "- $\\eta_t$: 適応的学習率（驚きが大きいほど強く書き込む）\n\n"
            "これにより、文脈が100万トークンを超えても、メモリサイズ $O(1)$ のまま過去の重要情報を半永久的に保持できます。"
        )
    },
    {
        "instruction": "ローカル環境のGPUスペックとMojoのバージョンを確認して。",
        "trajectory": (
            "<think>GPUスペックとMojoのバージョンを調査する。\n"
            "nvidia-smi と pixi run mojo --version を実行して情報を収集する。</think>\n"
            "<tool_call>bash_execute(command='nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && pixi run mojo --version')</tool_call>\n"
            "<tool_response>NVIDIA GeForce RTX 5060, 8151 MiB\nMojo 1.0.0 (ed45d567)</tool_response>\n"
            "<think>情報を正確に取得できたので、整理して報告する。</think>\n"
            "システムスペックの確認結果です：\n"
            "- **GPU**: NVIDIA GeForce RTX 5060 (VRAM: 8GB, Blackwell世代 / Compute 10.x)\n"
            "- **Mojoバージョン**: Mojo 1.0.0 (ed45d567)\n"
            "RTX 5060の8GB VRAM上で、BitNet b1.58 3値量子化モデル（約0.3GB〜1GB消費）が極めて高速に動作可能な状態です。"
        )
    },
    {
        "instruction": "最新の日本のAI技術動向についてWeb検索して要約して。",
        "trajectory": (
            "<think>最新の日本のAI技術動向をリサーチする。\n"
            "web_searchツールを用いて日本のLLM、オープンソースモデル、産学連携の動向を検索する。</think>\n"
            "<tool_call>web_search(query='日本 国産LLM 最新動向 2025 2026')</tool_call>\n"
            "<tool_response>Swallow, PLaMo, Sarashina, 富岳LLMなどの国産モデル、エッジAI、省電力モデルの研究が加速中。</tool_response>\n"
            "<think>検索結果を基に要約を作成する。</think>\n"
            "日本の最新AI動向のまとめです：\n"
            "1. **高品質な日本語特化モデル**: SwallowやSarashina等、日本語コーパスの品質にこだわったオープンモデルの発展。\n"
            "2. **省電力・エッジAIへのシフト**: 巨大なデータセンター型モデルだけでなく、ローカル端末や組み込みで動作するTinyLLM・量子化モデルの需要が急増。\n"
            "3. **エージェント化（自律推論・ツール連携）**: 従来のチャットボットから、コード実行やリサーチを自律的にこなすClaude Code風エージェントの現場導入が進展。"
        )
    }
]

def save_reasoning_dataset(output_path: str):
    print(f"[Dataset] Saving {len(AGENT_REASONING_TRAJECTORIES)} reasoning and agent trajectories to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in AGENT_REASONING_TRAJECTORIES:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[Dataset] Reasoning dataset successfully prepared!")

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_file = os.path.join(out_dir, "nexus_reasoning_sft.jsonl")
    save_reasoning_dataset(dataset_file)
