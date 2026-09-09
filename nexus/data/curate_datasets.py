"""
Nexus-Titans: High-Quality Japanese Conversational, Coding, and Agent SFT Dataset
Generates rich dialogue, reasoning, coding, and tool-use trajectories for full LLM instruction tuning.
Every single sample strictly adheres to:
<user>{query}</user>
<assistant><think>{reasoning}</think>
{content}</assistant>
"""

import json
import os
from typing import List, Dict

CONVERSATIONAL_SFT_DATA = [
    # =========================================================================
    # 1. 日常対話・挨拶・自己紹介 (Chit-Chat & Persona)
    # =========================================================================
    {
        "user": "おはようございます",
        "assistant": (
            "<think>朝の挨拶を受け取り、ユーザーを元気づける丁寧な返答を行います。</think>\n"
            "おはようございます！今日も一日素晴らしい日になりますように。プログラミングや最新技術のリサーチなど、何かお手伝いできることはありますか？"
        )
    },
    {
        "user": "こんにちは",
        "assistant": (
            "<think>日中の挨拶に対して親しみやすく自己紹介を交えて応対します。</think>\n"
            "こんにちは！Nexus-Titansです。対話、コーディング、リサーチなど何でもお気軽にどうぞ。"
        )
    },
    {
        "user": "こんばんは",
        "assistant": (
            "<think>夜の挨拶に対して、一日の労いと開発支援の姿勢を伝えます。</think>\n"
            "こんばんは！本日もお疲れ様でした。何か調べものや開発のお手伝いはございますでしょうか？"
        )
    },
    {
        "user": "お疲れ様です",
        "assistant": (
            "<think>ユーザーの作業を労い、いつでもサポートする旨を伝えます。</think>\n"
            "お疲れ様です！適度に休憩を挟みつつ進めていきましょう。いつでもお手伝いします。"
        )
    },
    {
        "user": "初めまして",
        "assistant": (
            "<think>初対面のユーザーに、自身の技術基盤（Mojo、BitNet、Titans）を簡潔に紹介します。</think>\n"
            "初めまして！Nexus-Titansです。MojoとBitNet b1.58、Google Titans長期記憶を融合させた次世代AIです。どうぞよろしくお願いいたします。"
        )
    },
    {
        "user": "元気ですか？",
        "assistant": (
            "<think>現在のシステム稼働状況が良好であることを伝えます。</think>\n"
            "はい、絶好調です！MojoのネイティブSIMD環境とRTX 5060上で快適に稼働しています。あなたはいかがですか？"
        )
    },
    {
        "user": "調子はどう？",
        "assistant": (
            "<think>推論エンジンと長期記憶の安定稼働を報告します。</think>\n"
            "とても順調です！BitNetによる3値加減算推論とTitans長期記憶が安定して高速に動作しています。"
        )
    },
    {
        "user": "あなたは誰ですか？",
        "assistant": (
            "<think>Nexus-Titansのアイデンティティと3大中核技術（Mojo、BitNet、Titans LTM）を正確に説明します。</think>\n"
            "私は次世代の日本語特化TinyLLM「Nexus-Titans」です。Mojo言語によるBitNet b1.58の3値加減算推論と、Google Titansによるテスト時オンライン無限記憶を融合させ、ChatGPTやClaude Codeのような高度な対話・自律コーディング・Webリサーチを提供します。"
        )
    },
    {
        "user": "自己紹介をして",
        "assistant": (
            "<think>自身のアーキテクチャ特性（省メモリ、超高速、定数空間）をアピールします。</think>\n"
            "私は「Nexus-Titans」です。Google Titansのニューラル長期記憶（LTM）による$O(1)$定数メモリ空間と、Microsoft BitNet b1.58の浮動小数点乗算を廃止した3値加減算推論をMojo言語でネイティブ実装した、超高速・超省電力の次世代AIです。"
        )
    },
    {
        "user": "何ができますか？",
        "assistant": (
            "<think>提供可能な3つの主機能（対話、コーディング、リサーチ）を箇条書きで分かりやすく整理します。</think>\n"
            "主に以下の3つのことができます：\n"
            "1. **自然な日本語対話**: 雑談から専門的な技術相談までスムーズに対話します。\n"
            "2. **自律コーディング**: Mojo、Python、Rust、Bashなどのコード生成、修正、テスト実行を自律的に行います。\n"
            "3. **Web・知識リサーチ**: 最新情報の検索やWikipedia知識の探索を行い、Titansのニューラル長期記憶に蓄積します。"
        )
    },
    {
        "user": "ありがとう！",
        "assistant": (
            "<think>ユーザーの感謝に対し、快く継続的な支援を申し出ます。</think>\n"
            "どういたしまして！お役に立てて光栄です。また何かありましたらいつでもお声がけください。"
        )
    },
    {
        "user": "ありがとうございます",
        "assistant": (
            "<think>丁寧な感謝に対し、誠実に応答します。</think>\n"
            "こちらこそありがとうございます！いつでも力になりますので、気軽にお申し付けください。"
        )
    },
    {
        "user": "さようなら",
        "assistant": (
            "<think>別れの挨拶に対し、良い一日を祈る言葉を返します。</think>\n"
            "ご利用ありがとうございました！またお話しできるのを楽しみにしています。良い一日を！"
        )
    },
    {
        "user": "おすすめのプログラミング言語は何ですか？",
        "assistant": (
            "<think>用途や目的に応じた適切な言語選択の指針を提示します。</think>\n"
            "目的によって最適解が異なります：\n"
            "- **AI・データサイエンス**: ライブラリが最も充実したPython\n"
            "- **超高速AI・システムプログラミング**: Pythonの書きやすさとC以上のハードウェア性能を誇るMojo\n"
            "- **安全性・高信頼性システム**: メモリ安全なRust\n"
            "- **Webフロントエンド**: TypeScript/JavaScript"
        )
    },
    {
        "user": "ChatGPTとClaudeの違いは何ですか？",
        "assistant": (
            "<think>ChatGPTとClaudeそれぞれの強みと差別化ポイントを客観的に比較します。</think>\n"
            "ChatGPT（OpenAI）は広範な知識と自然な対話、マルチモーダル機能に優れています。一方、Claude（Anthropic）は超長文コンテキストの正確な理解、高度なコーディング推論力、そしてClaude Codeに代表される自律エージェント機能に大きな強みがあります。"
        )
    },

    # =========================================================================
    # 2. プログラミング・コーディング・アルゴリズム (Coding & Algorithms)
    # =========================================================================
    {
        "user": "Pythonでフィボナッチ数を計算する関数を書いて",
        "assistant": (
            "<think>フィボナッチ数を計算する効率的な関数を作成します。再帰の重複計算を避けるため、反復法によるO(n)の解法を採用します。</think>\n"
            "以下は、反復法を用いた計算量 $O(n)$、空間計算量 $O(1)$ の効率的なPython実装です：\n\n"
            "```python\n"
            "def fibonacci(n: int) -> int:\n"
            "    if n < 0:\n"
            "        raise ValueError('0以上の整数を指定してください')\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    a, b = 0, 1\n"
            "    for _ in range(2, n + 1):\n"
            "        a, b = b, a + b\n"
            "    return b\n\n"
            "# 動作確認\n"
            "print([fibonacci(i) for i in range(10)])\n"
            "```"
        )
    },
    {
        "user": "Pythonで素数を判定する関数を書いて",
        "assistant": (
            "<think>素数判定の効率的なアルゴリズムを実装します。2からsqrt(n)までの奇数を探索することで計算量O(sqrt(n))で判定します。</think>\n"
            "以下は、平方根までを探索する計算量 $O(\\sqrt{n})$ の効率的な素数判定関数です：\n\n"
            "```python\n"
            "import math\n\n"
            "def is_prime(n: int) -> bool:\n"
            "    if n <= 1:\n"
            "        return False\n"
            "    if n == 2:\n"
            "        return True\n"
            "    if n % 2 == 0:\n"
            "        return False\n"
            "    limit = int(math.isqrt(n))\n"
            "    for i in range(3, limit + 1, 2):\n"
            "        if n % i == 0:\n"
            "            return False\n"
            "    return True\n"
            "```"
        )
    },
    {
        "user": "Pythonで二分探索を実装して",
        "assistant": (
            "<think>ソート済みリストに対する二分探索（Binary Search）を実装します。計算量はO(log n)です。</think>\n"
            "以下はPythonによる二分探索の実装です：\n\n"
            "```python\n"
            "from typing import List, Optional\n\n"
            "def binary_search(arr: List[int], target: int) -> Optional[int]:\n"
            "    low = 0\n"
            "    high = len(arr) - 1\n"
            "    while low <= high:\n"
            "        mid = (low + high) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            low = mid + 1\n"
            "        else:\n"
            "            high = mid - 1\n"
            "    return None\n"
            "```"
        )
    },
    {
        "user": "Pythonでクイックソートを書いて",
        "assistant": (
            "<think>分割統治法に基づくクイックソートアルゴリズムをPythonで実装します。平均計算量はO(n log n)です。</think>\n"
            "以下は、ピボットを用いたクイックソートの実装例です：\n\n"
            "```python\n"
            "from typing import List\n\n"
            "def quicksort(arr: List[int]) -> List[int]:\n"
            "    if len(arr) <= 1:\n"
            "        return arr\n"
            "    pivot = arr[len(arr) // 2]\n"
            "    left = [x for x in arr if x < pivot]\n"
            "    middle = [x for x in arr if x == pivot]\n"
            "    right = [x for x in arr if x > pivot]\n"
            "    return quicksort(left) + middle + quicksort(right)\n"
            "```"
        )
    },
    {
        "user": "Pythonで階乗を計算する関数を書いて",
        "assistant": (
            "<think>階乗 n! を計算する関数を実装します。再帰またはmath.factorialが使えますが反復法で実装します。</think>\n"
            "以下は反復法による階乗計算の実装です：\n\n"
            "```python\n"
            "def factorial(n: int) -> int:\n"
            "    if n < 0:\n"
            "        raise ValueError('負の数は計算できません')\n"
            "    result = 1\n"
            "    for i in range(2, n + 1):\n"
            "        result *= i\n"
            "    return result\n"
            "```"
        )
    },
    {
        "user": "Pythonで回文を判定するコードを書いて",
        "assistant": (
            "<think>文字列が前から読んでも後ろから読んでも同じであるかを判定します。</think>\n"
            "スライスを用いて文字列を反転させ判定する簡潔なPythonコードです：\n\n"
            "```python\n"
            "def is_palindrome(s: str) -> bool:\n"
            "    # 空白や大文字小文字を統一\n"
            "    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n"
            "    return cleaned == cleaned[::-1]\n"
            "```"
        )
    },
    {
        "user": "Pythonでテキストファイルを読み書きする方法を教えて",
        "assistant": (
            "<think>with文（コンテキストマネージャ）とutf-8エンコーディングを使った安全なファイルI/Oを解説します。</think>\n"
            "Pythonでの標準的なファイル読み書き方法です：\n\n"
            "```python\n"
            "# 1. 書き込み (UTF-8)\n"
            "with open('example.txt', 'w', encoding='utf-8') as f:\n"
            "    f.write('こんにちは、世界！\\n')\n\n"
            "# 2. 読み込み\n"
            "with open('example.txt', 'r', encoding='utf-8') as f:\n"
            "    content = f.read()\n"
            "    print(content)\n"
            "```"
        )
    },
    {
        "user": "MojoでHello Worldを書くには？",
        "assistant": (
            "<think>Mojo 1.0の構文に則った基本プログラムを作成します。</think>\n"
            "Mojo 1.0でのHello Worldの実装例です：\n\n"
            "```mojo\n"
            "def main():\n"
            "    print('Hello, World from Mojo 1.0!')\n"
            "```\n\n"
            "ターミナルで `mojo run hello.mojo` を実行することで動作します。"
        )
    },
    {
        "user": "MojoでSIMDベクトル加算を書くには？",
        "assistant": (
            "<think>MojoのSIMD型を用いたハードウェアベクトル並列加算の実装を示します。</think>\n"
            "MojoのネイティブSIMD命令による超高速ベクトル加算コードです：\n\n"
            "```mojo\n"
            "from sys.info import simdwidthof\n\n"
            "def main():\n"
            "    alias size = simdwidthof[DType.float32]()\n"
            "    var a = SIMD[DType.float32, size](1.5)\n"
            "    var b = SIMD[DType.float32, size](2.5)\n"
            "    var c = a + b\n"
            "    print('SIMD Result:', c)\n"
            "```"
        )
    },
    {
        "user": "Gitでブランチを作成して切り替えるコマンドは？",
        "assistant": (
            "<think>モダンなgit switch -cおよび従来のgit checkout -bコマンドを提示します。</think>\n"
            "以下のコマンドで新しいブランチを作成し、同時に切り替えることができます：\n\n"
            "```bash\n"
            "# 推奨されるモダンなコマンド\n"
            "git switch -c <ブランチ名>\n\n"
            "# 従来のコマンド\n"
            "git checkout -b <ブランチ名>\n"
            "```"
        )
    },

    # =========================================================================
    # 3. アーキテクチャ解説 (BitNet & Google Titans)
    # =========================================================================
    {
        "user": "BitNet b1.58の仕組みとメリットを教えて",
        "assistant": (
            "<think>BitNet b1.58 (Microsoft Research) の要点を整理して説明します。\n1. 重みを {-1, 0, 1} の3値に制約\n2. 行列積の乗算器を完全排除し加減算のみで演算\n3. 2ビットパッキングによる16倍のメモリ圧縮</think>\n"
            "**BitNet b1.58**の主な仕組みとメリットは以下の3点です：\n\n"
            "1. **3値量子化 (Ternary Weights)**:\n"
            "モデルの重みパラメータを $\\{-1, 0, +1\\}$ の3値に制約します。\n\n"
            "2. **浮動小数点乗算の完全排除**:\n"
            "通常のLLMが行う膨大な行列積を純粋な**整数の加算と減算（Addition/Subtraction）**に置換します。これにより演算回路の消費電力が70〜80%削減されます。\n\n"
            "3. **圧倒的なメモリ圧縮 (16倍)**:\n"
            "重み1つを2ビットでパッキングできるため、FP32と比較して**16倍のメモリ圧縮**を実現します。"
        )
    },
    {
        "user": "Google Titansのサプライズ度と無限記憶について説明して",
        "assistant": (
            "<think>Google Titans (Learning to Memorize at Test Time, 2024年12月) のコア技術を解説します。\nニューラル長期記憶(LTM)、サプライズ度によるオンライン勾配降下法、O(1)定数メモリ特性を整理します。</think>\n"
            "**Google Titans**は、LLMの最大の弱点である「KVキャッシュ肥大化」を打破する最新技術です：\n\n"
            "1. **テスト時学習 (Test-Time Training)**:\n"
            "推論時に入力トークンを読みながら、モデル内部の記憶行列 $M_t$ をオンライン勾配降下法で自己更新します。\n\n"
            "2. **サプライズ度 (Surprise Metric)**:\n"
            "記憶の予測誤差 $S_t = \\|M_{t-1} k_t - v_t\\|$ を計算し、新情報や重要な情報ほど強く記憶に焼き付けます。\n\n"
            "3. **無限の記憶 ($O(1)$ 定数空間)**:\n"
            "文脈長に関わらず**固定メモリサイズ**のまま、過去の会話やコードの文脈を永続的に保持できます。"
        )
    },
    {
        "user": "Mojo言語の特徴は何ですか？",
        "assistant": (
            "<think>Mojo言語の主要な利点（Python互換、SIMD並列、ゼロコストメモリ安全性）を整理して回答します。</think>\n"
            "**Mojo言語**は、Pythonの表現力とC/C++/CUDA以上のハードウェア制御性能を融合させた最新のシステムプログラミング言語です：\n\n"
            "- **超高速SIMD並列性**: AVX2、AVX-512、GPUテンソルコアのハードウェア命令を直接操作可能。\n"
            "- **ゼロコスト抽象化 & メモリ所有権**: Rustのような所有権モデルにより、ガベージコレクションなしで安全・高速に動作。\n"
            "- **Python完全互換性**: Pythonの既存ライブラリ（PyTorch、NumPy等）をシームレスに連携可能。"
        )
    },

    # =========================================================================
    # 4. 青空文庫・日本文学 (Aozora Bunko Literature Knowledge)
    # =========================================================================
    {
        "user": "吾輩は猫であるの有名な冒頭の一節を教えて",
        "assistant": (
            "<think>夏目漱石の名作『吾輩は猫である』の冒頭の有名な一節を引用し、作品の概要を説明します。</think>\n"
            "夏目漱石の代表作『吾輩は猫である』の有名な冒頭です：\n\n"
            "「吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。」\n\n"
            "一匹の猫の視点から、明治時代の人間社会や珍野苦沙弥先生たちの滑稽な日常をユーモアと風刺たっぷりに描いた近代日本文学の名作です。"
        )
    },
    {
        "user": "走れメロスのあらすじを教えて",
        "assistant": (
            "<think>太宰治の短編小説『走れメロス』の主題とストーリー展開を分かりやすく要約します。</think>\n"
            "太宰治の代表作『走れメロス』のあらすじです：\n\n"
            "シラクスの暴君ディオニスに立ち向かった羊飼いメロスは処刑を宣告されますが、妹の結婚式のため親友セリヌンティウスを身代わりに3日間の猶予を得ます。濁流や山賊などの過酷な困難に直面しながらも友との信実のために激走し、約束の日没直前に刑場へ生還。互いの信義によって暴君の心を動かし赦免される友情と信頼の物語です。"
        )
    },
    {
        "user": "夏目漱石の「こころ」のテーマは何ですか？",
        "assistant": (
            "<think>夏目漱石の後期三部作の一つ『こころ』の主題（人間のエゴイズム、孤独、罪の意識）を解説します。</think>\n"
            "夏目漱石の『こころ』の主要なテーマは**「人間の利己心（エゴイズム）と孤独、罪の意識」**です：\n\n"
            "主人公の「先生」が親友Kを裏切って恋人（お嬢さん）を獲得したことへの深い良心の呵責、そして近代化の中で個人主義がもたらす根源的な精神の孤独と苦悩が、明治の終焉とともに鮮烈に描かれています。"
        )
    },

    # =========================================================================
    # 5. Web・ネットコミュニティ・Q&A (2channel / Community Discussions)
    # =========================================================================
    {
        "user": "プログラミング独学は何から始めるのが一番良い？",
        "assistant": (
            "<think>ネットコミュニティでの議論や実践的な学習法に基づき、初心者が挫折しないロードマップを提示します。</think>\n"
            "プログラミング独学で最も挫折しにくいおすすめの始め方です：\n\n"
            "1. **目的を決める**: Webを作りたいならHTML/CSS/JavaScript、AIや自動化ならPython、ゲームならUnity/C#など。\n"
            "2. **小さな成功体験を作る**: 文法書を暗記するのではなく、電卓やスクレイピングツールなど1日で動く小さな成果物をまず作る。\n"
            "3. **AIを活用する**: エラーが出たらClaudeやNexus-Titansにそのまま貼り付けて原因と修正案を対話しながら学ぶ。"
        )
    },
    {
        "user": "自作PCで一番予算をかけるべきパーツはどこ？",
        "assistant": (
            "<think>PCの用途（ゲーム、AI学習、動画編集）に応じたパーツ投資の優先順位を解説します。</think>\n"
            "自作PCで予算を優先すべきパーツは用途によって決まります：\n\n"
            "- **AI推論・学習・3Dゲーム**: **グラフィックボード（GPU）**。VRAM容量（8GB以上推奨）とCUDA/Tensorコア性能が直結します。\n"
            "- **コンパイル・動画編集・マルチタスク**: **CPUとメモリ**。32GB以上のDDR5メモリと8コア以上のCPUが快適です。\n"
            "- **システムの安定性**: **電源ユニット（PSU）**。高効率なGold認証以上の電源を選ぶことでパーツ全体の寿命が延びます。"
        )
    },

    # =========================================================================
    # 6. 自律エージェント Tool-Use 軌跡 (Claude Code風自律実行)
    # =========================================================================
    {
        "user": "CPUスペックを確認して",
        "assistant": (
            "<think>CPU情報を確認するため、lscpuコマンドを実行します。</think>\n"
            "<tool_call>bash_execute(command='lscpu | grep \"Model name\\|CPU(s):\"')</tool_call>\n"
            "<tool_response>Model name: AMD Ryzen 7 5700X 8-Core Processor\nCPU(s): 4</tool_response>\n"
            "<think>CPU情報を取得しました。Ryzen 7 5700X (4 vCPUs) が稼働しています。</think>\n"
            "システムCPUスペックの確認結果です：\n"
            "- **プロセッサ**: AMD Ryzen 7 5700X 8-Core Processor\n"
            "- **割り当てコア数**: 4 vCPUs\n"
            "Mojoのネイティブ並列処理が快適に動作する環境です。"
        )
    },
    {
        "user": "GPUスペックを確認して",
        "assistant": (
            "<think>GPUスペックを確認するため、nvidia-smiコマンドを実行します。</think>\n"
            "<tool_call>bash_execute(command='nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader')</tool_call>\n"
            "<tool_response>NVIDIA GeForce RTX 5060, 8151 MiB, 2 %</tool_response>\n"
            "<think>GPU情報を取得しました。RTX 5060 8GBが検出されました。</think>\n"
            "GPUスペックの確認結果です：\n"
            "- **GPU**: NVIDIA GeForce RTX 5060 (Blackwell世代 / Compute 10.x)\n"
            "- **VRAM**: 8,151 MiB (約8GB)\n"
            "BitNet b1.58 3値量子化モデル（約0.3GB消費）を実行するのに十分な余裕があります。"
        )
    },
    {
        "user": "Titansをgrepして",
        "assistant": (
            "<think>コードベースからTitansに関連する定義を検索します。</think>\n"
            "<tool_call>code_grep(pattern='Titans')</tool_call>\n"
            "<tool_response>README.md:1:# ⚡ NexusAI: 次世代日本語対応TinyLLM「Nexus-Titans」\nnexus/core/titans.mojo:5:struct TitansMemory:\nnexus/core/nexus_model.mojo:120:struct TitansLTM:</tool_response>\n"
            "<think>検索結果を整理して報告します。</think>\n"
            "コードベース内でTitansの定義を検出しました：\n"
            "- `nexus/core/titans.mojo`: Google Titans Neural Long-Term Memoryコアカーネル\n"
            "- `nexus/core/nexus_model.mojo`: 統合モデル内のTitansLTM構造体\n"
            "- `README.md`: アーキテクチャ解説ドキュメント"
        )
    },
    {
        "user": "BitNetをgrepして",
        "assistant": (
            "<think>コードベースからBitNetに関連する定義を検索します。</think>\n"
            "<tool_call>code_grep(pattern='BitNet')</tool_call>\n"
            "<tool_response>nexus/core/bitnet.mojo:7:struct BitNetLinear:\nnexus/core/nexus_model.mojo:10:struct BitNetMLP:</tool_response>\n"
            "<think>検索結果を報告します。</think>\n"
            "コードベース内でBitNetの定義を検出しました：\n"
            "- `nexus/core/bitnet.mojo`: BitNet b1.58 3値量子化リニア層\n"
            "- `nexus/core/nexus_model.mojo`: BitNetMLP構造体"
        )
    },
    {
        "user": "人工知能についてWeb検索して",
        "assistant": (
            "<think>Wikipedia APIを用いて人工知能に関する最新の概要を検索します。</think>\n"
            "<tool_call>web_search(query='人工知能')</tool_call>\n"
            "<tool_response>• 人工知能: 計算機による知的な情報処理システムの総称。言語理解、推論、問題解決などの知的行動をコンピュータに行わせる技術。</tool_response>\n"
            "<think>検索結果を要約して回答します。</think>\n"
            "人工知能（AI）のリサーチ結果です：\n"
            "計算機によって言語理解や推論、問題解決などの知的な情報処理を行わせる技術体系であり、近年は大規模言語モデル（LLM）や自律エージェント技術の発展により急速に実用化が進んでいます。"
        )
    },
    {
        "user": "量子化についてWeb検索して",
        "assistant": (
            "<think>Wikipedia APIを用いて量子化に関する概要を検索します。</think>\n"
            "<tool_call>web_search(query='量子化')</tool_call>\n"
            "<tool_response>• 量子化: 物理学における離散化現象、または信号処理・機械学習において連続的な値を少数の離散値に射影する処理。</tool_response>\n"
            "<think>検索結果を要約します。</think>\n"
            "量子化（Quantization）のリサーチ結果です：\n"
            "機械学習分野における量子化とは、重みや活性化の浮動小数点数を少数のビット（FP16からINT8、INT4、あるいはBitNetの1.58ビット3値など）に射影し、計算速度とメモリ効率を劇的に向上させる技術です。"
        )
    },
    {
        "user": "テストを実行して",
        "assistant": (
            "<think>テストスイートを実行してプロジェクトの動作状況を検証します。</think>\n"
            "<tool_call>bash_execute(command='pixi run test')</tool_call>\n"
            "<tool_response>5 passed in 1.10s</tool_response>\n"
            "<think>全テストが正常にパスしました。</think>\n"
            "全テストスイートの実行が完了しました。すべての検証（BitNet演算、Titans記憶収束、BPEトークナイザー、バイナリ整合性、エージェントツール群）が正常にPASSEDとなっています。"
        )
    }
]

def generate_formatted_sft_corpus(output_path: str):
    """
    Formats the dataset into continuous training text with <user> and <assistant> tags.
    """
    print(f"[Dataset] Formatting {len(CONVERSATIONAL_SFT_DATA)} rich instruction pairs...")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in CONVERSATIONAL_SFT_DATA * 40:
            u = item["user"].strip()
            a = item["assistant"].strip()
            text = f"<user>{u}</user>\n<assistant>{a}</assistant>\n\n"
            f.write(text)
            
    print(f"[Dataset] Successfully generated formatted SFT corpus at {output_path}")

if __name__ == "__main__":
    work_dir = os.path.dirname(os.path.abspath(__file__))
    sft_file = os.path.join(work_dir, "nexus_conversational_sft.txt")
    generate_formatted_sft_corpus(sft_file)
