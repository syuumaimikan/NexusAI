"""
Nexus-Titans: Multi-Domain Japanese Corpus Harvester & Preprocessor
Extracts and unifies massive datasets across:
1. Japanese Wikipedia (100,000 articles)
2. Aozora Bunko (Classic Japanese Literature)
3. 2channel / Open2ch (Conversational Internet Forums & Q&A)
4. High-Quality Agent SFT Dialogues (Chat, Coding, & Autonomous Tools)
"""

import os
import sys
import gzip
import json
import re
from typing import List, Dict, Generator
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tqdm import tqdm

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from nexus.data.curate_datasets import CONVERSATIONAL_SFT_DATA
DATA_DIR = os.path.join(WORKSPACE_ROOT, "nexus", "data")

def clean_japanese_text(text: str) -> str:
    """Cleans raw text by removing excessive whitespace and junk artifacts."""
    if not text:
        return ""
    # Remove HTML tags if present
    text = re.sub(r"<[^>]+>", "", text)
    # Normalize multiple whitespace and newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def harvest_wikipedia(max_articles: int = 15000) -> List[str]:
    """Extracts articles from Japanese Wikipedia parquet dump."""
    print(f"[Harvester] Loading Japanese Wikipedia (target: {max_articles:,} articles)...")
    try:
        wiki_path = hf_hub_download(
            repo_id="mmnga/wikipedia-ja-20230720-100k",
            filename="data/train-00000-of-00001-7dc295ea415020ef.parquet",
            repo_type="dataset"
        )
        table = pq.read_table(wiki_path)
        texts = table.column("text").to_pylist()
        
        extracted = []
        for i in range(min(max_articles, len(texts))):
            raw = texts[i]
            cleaned = clean_japanese_text(raw)
            # Filter short stubs or empty articles
            if len(cleaned) >= 100:
                extracted.append(cleaned[:2000]) # Cap each article chunk
                
        print(f"[Harvester] Successfully extracted {len(extracted):,} Wikipedia articles.")
        return extracted
    except Exception as e:
        print(f"[Warning] Failed to harvest Wikipedia: {e}")
        return []

def harvest_aozora_bunko(max_works: int = 5000) -> List[str]:
    """Extracts literary works from cleaned Aozora Bunko archive."""
    print(f"[Harvester] Loading Aozora Bunko Japanese literature (target: {max_works:,} works)...")
    try:
        aozora_path = hf_hub_download(
            repo_id="globis-university/aozorabunko-clean",
            filename="aozorabunko-dedupe-clean.jsonl.gz",
            repo_type="dataset"
        )
        extracted = []
        with gzip.open(aozora_path, "rt", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                txt = item.get("text", "")
                cleaned = clean_japanese_text(txt)
                if len(cleaned) >= 150:
                    extracted.append(cleaned[:2500])
                if len(extracted) >= max_works:
                    break
        print(f"[Harvester] Successfully extracted {len(extracted):,} Aozora Bunko texts.")
        return extracted
    except Exception as e:
        print(f"[Warning] Failed to harvest Aozora Bunko: {e}")
        return []

def harvest_2channel_qa(max_dialogues: int = 15000) -> List[Dict[str, str]]:
    """Extracts conversational Q&A pairs from Open2ch LiveJupiter."""
    print(f"[Harvester] Loading 2channel / Open2ch dialogue (target: {max_dialogues:,} dialogues)...")
    try:
        open2ch_path = hf_hub_download(
            repo_id="yuiseki/open2ch-livejupiter-qa",
            filename="data/train-00000-of-00001.parquet",
            repo_type="dataset"
        )
        table = pq.read_table(open2ch_path)
        questions = table.column("question").to_pylist()
        answers = table.column("answer").to_pylist()
        
        pairs = []
        for q, a in zip(questions, answers):
            clean_q = clean_japanese_text(q)
            clean_a = clean_japanese_text(a)
            # Filter low quality or extremely short replies
            if len(clean_q) >= 4 and len(clean_a) >= 4:
                # Basic profanity/spam filter
                if not any(bad in clean_q or bad in clean_a for bad in ["死ね", "殺す", "アフィ"]):
                    pairs.append({"user": clean_q, "assistant": clean_a})
            if len(pairs) >= max_dialogues:
                break
                
        print(f"[Harvester] Successfully extracted {len(pairs):,} 2channel conversational dialogue pairs.")
        return pairs
    except Exception as e:
        print(f"[Warning] Failed to harvest 2channel dialogue: {e}")
        return []

def build_unified_dataset():
    """Builds pretraining corpus and rich multi-turn SFT dataset."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 1. Harvest each domain
    wiki_texts = harvest_wikipedia(max_articles=10000)
    aozora_texts = harvest_aozora_bunko(max_works=4000)
    two_ch_pairs = harvest_2channel_qa(max_dialogues=10000)
    
    # 2. Build Massive Pretrain Text Corpus
    pretrain_path = os.path.join(DATA_DIR, "nexus_massive_pretrain.txt")
    total_chars = 0
    with open(pretrain_path, "w", encoding="utf-8") as f:
        print(f"[Pipeline] Compiling pretrain corpus into {pretrain_path}...")
        for t in wiki_texts:
            f.write(t + "\n\n")
            total_chars += len(t)
        for t in aozora_texts:
            f.write(t + "\n\n")
            total_chars += len(t)
        for pair in two_ch_pairs:
            line = f"質問: {pair['user']}\n回答: {pair['assistant']}\n\n"
            f.write(line)
            total_chars += len(line)
            
    print(f"[Pipeline] Pretrain corpus ready: {os.path.getsize(pretrain_path)/(1024*1024):.2f} MB ({total_chars:,} characters)")
    
    # 3. Build Multi-Domain SFT Instruction Corpus
    sft_path = os.path.join(DATA_DIR, "nexus_dialogue_sft.jsonl")
    all_sft_pairs: List[Dict[str, str]] = []
    
    # High-priority Agent instructions (repeat to establish strong agent persona)
    all_sft_pairs.extend(CONVERSATIONAL_SFT_DATA * 20)
    
    # Add curated 2channel conversational threads
    all_sft_pairs.extend(two_ch_pairs[:5000])
    
    with open(sft_path, "w", encoding="utf-8") as f:
        print(f"[Pipeline] Compiling {len(all_sft_pairs):,} SFT dialogue turns into {sft_path}...")
        for item in all_sft_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"[Pipeline] SFT dataset ready: {len(all_sft_pairs):,} pairs at {sft_path}")
    print("\n[Summary] Multi-domain Japanese corpus creation complete:")
    print(f" - Wikipedia Articles: {len(wiki_texts):,}")
    print(f" - Aozora Bunko Works: {len(aozora_texts):,}")
    print(f" - 2channel / Forum Dialogues: {len(two_ch_pairs):,}")
    print(f" - Agent SFT Trajectories: {len(CONVERSATIONAL_SFT_DATA)} unique patterns")

if __name__ == "__main__":
    build_unified_dataset()
