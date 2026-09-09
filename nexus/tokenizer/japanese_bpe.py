"""
Nexus-Titans: Japanese High-Compression BPE Tokenizer (Nexus-BPE 48k)
Optimized for Japanese kanji, hiragana/katakana compounds, and programming languages (Mojo, Python, Rust, Bash).
"""

import os
import sys
import sentencepiece as spm
from typing import List, Dict, Tuple

class JapaneseBPETokenizer:
    def __init__(self, model_prefix: str = "nexus_ja_bpe", vocab_size: int = 8000):
        self.model_prefix = model_prefix
        self.vocab_size = vocab_size
        self.sp = None
        self.model_file = f"{model_prefix}.model"
        self.vocab_file = f"{model_prefix}.vocab"

    def train_from_corpus(self, corpus_text_path: str):
        """
        Trains a SentencePiece BPE model with byte fallback and Japanese character normalization.
        """
        print(f"[Tokenizer] Training SentencePiece BPE model with vocab_size={self.vocab_size}...")
        spm.SentencePieceTrainer.train(
            input=corpus_text_path,
            model_prefix=self.model_prefix,
            vocab_size=self.vocab_size,
            character_coverage=0.9995,
            model_type="bpe",
            byte_fallback=True,
            pad_id=3,
            user_defined_symbols=[
                "<user>", "</user>",
                "<assistant>", "</assistant>",
                "<tool_call>", "</tool_call>",
                "<tool_response>", "</tool_response>",
                "<think>", "</think>",
                "<memory>", "</memory>"
            ],
            split_digits=True,
            normalization_rule_name="nfkc"
        )
        self.load()
        print(f"[Tokenizer] Model saved to {self.model_file}")

    def load(self, model_file: str = None):
        if model_file is None:
            model_file = self.model_file
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_file)
        self.model_file = model_file

    def encode(self, text: str) -> List[int]:
        if self.sp is None:
            self.load()
        return self.sp.encode_as_ids(text)

    def decode(self, ids: List[int]) -> str:
        if self.sp is None:
            self.load()
        return self.sp.decode_ids(ids)

    def get_compression_ratio(self, text: str) -> Dict[str, float]:
        """
        Calculates compression ratio: number of tokens vs characters and UTF-8 bytes.
        """
        tokens = self.encode(text)
        num_chars = len(text)
        num_bytes = len(text.encode("utf-8"))
        num_tokens = len(tokens)
        
        return {
            "num_chars": num_chars,
            "num_bytes": num_bytes,
            "num_tokens": num_tokens,
            "tokens_per_char": num_tokens / max(1, num_chars),
            "bytes_per_token": num_bytes / max(1, num_tokens)
        }

def prepare_japanese_seed_corpus(output_path: str):
    """
    Creates a high-density Japanese & code training corpus seed for the tokenizer.
    """
    samples = [
        # 百科事典・科学・技術・社会
        "人工知能（じんこうちのう、英: Artificial Intelligence、AI）とは、計算機による知的な情報処理システムの総称である。",
        "東京は日本の首都であり、政治・経済・文化の中心都市である。千代田区、中央区、港区など23の特別区から成る。",
        "量子コンピュータは量子力学の重ね合わせと量子もつれを利用して超並列計算を行う次世代計算機である。",
        "機械学習におけるニューラルネットワークは、多層の重みパラメータと活性化関数によって非線形な関数近似を行う。",
        "自然言語処理（NLP）では、大規模言語モデル（LLM）がTransformerアーキテクチャを基盤として発展してきた。",
        "BitNet b1.58は重みを{-1, 0, +1}の3値に制約することで、乗算器を不要にし、加減算のみで超省電力を実現する。",
        "Google Titansは推論時にサプライズ誤差をオンライン勾配降下法で学習し、無限の文脈を定数メモリで記憶する。",
        # プログラミング・コード構文（Mojo, Python, Rust, Bash）
        "def main():\n    var x: Int = 10\n    var y: Float32 = 3.14\n    print('NexusAI is running with Mojo!')",
        "struct TitansMemory:\n    var d_model: Int\n    var d_mem: Int\n    def retrieve(self, q): return M * q",
        "cargo build --release && ./target/release/nexus-engine --prompt 'こんにちは'",
        "git clone https://github.com/syuum/NexusAI.git && cd NexusAI && pixi run mojo run main.mojo",
        "curl -s https://ja.wikipedia.org/wiki/メインページ | grep -o '今日の出来事'",
        # 対話・エージェント（Claude Code風コマンド・敬語・プロンプト）
        "ユーザーの質問：このプロジェクトのバグを調査して修正コードを提案してください。",
        "<think>現在のコードベースを調査するため、まず関連ファイルを検索します。</think>",
        "<tool_call>bash_execute(command='pytest tests/')</tool_call>",
        "<tool_response>2 passed, 0 failed</tool_response>",
        "テストが正常に通過しました。変更内容をコミットしてもよろしいでしょうか？",
        "ありがとうございます。ご不明な点がございましたらいつでもお申し付けください。"
    ] * 200  # Repeat to give sufficient frequency counts for BPE merge operations

    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(s + "\n")

if __name__ == "__main__":
    work_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(work_dir), "data")
    corpus_file = os.path.join(work_dir, "seed_corpus.txt")
    model_prefix = os.path.join(work_dir, "nexus_ja_bpe")
    
    print("Step 1: Generating Japanese Seed Corpus...")
    prepare_japanese_seed_corpus(corpus_file)
    
    train_files = [corpus_file]
    massive_corpus = os.path.join(data_dir, "nexus_massive_pretrain.txt")
    if os.path.exists(massive_corpus):
        train_files.append(massive_corpus)
    sft_corpus = os.path.join(data_dir, "nexus_conversational_sft.txt")
    if os.path.exists(sft_corpus):
        train_files.append(sft_corpus)
    wiki_corpus = os.path.join(data_dir, "ja_wikipedia_corpus.txt")
    if os.path.exists(wiki_corpus):
        train_files.append(wiki_corpus)

    merged_input = ",".join(train_files)
    print(f"Step 2: Training Japanese BPE Tokenizer on {len(train_files)} corpora (vocab_size=8000)...")
    tok = JapaneseBPETokenizer(model_prefix=model_prefix, vocab_size=8000)
    tok.train_from_corpus(merged_input)
    
    test_text = "Mojo言語とTitansアーキテクチャを用いた次世代日本語AIの開発。"
    stats = tok.get_compression_ratio(test_text)
    print("\n=== Tokenizer Evaluation ===")
    print("Input Text:", test_text)
    print(f"Characters: {stats['num_chars']} | UTF-8 Bytes: {stats['num_bytes']} | Tokens: {stats['num_tokens']}")
    print(f"Tokens/Char: {stats['tokens_per_char']:.2f} (Target < 1.0 for high compression)")
    print(f"Decoded: {tok.decode(tok.encode(test_text))}")
