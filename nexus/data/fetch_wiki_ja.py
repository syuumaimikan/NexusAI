"""
Nexus-Titans: Japanese Wikipedia Knowledge Harvester
Fetches clean, encyclopedic Japanese articles across science, technology, programming, history, and geography.
"""

import os
import time
import requests
from bs4 import BeautifulSoup
from typing import List, Dict

SAMPLE_TOPICS = [
    "人工知能", "機械学習", "深層学習", "自然言語処理", "大規模言語モデル",
    "量子コンピュータ", "オペレーティングシステム", "コンパイラ",
    "Python", "プログラミング言語", "Linux", "分散コンピューティング",
    "日本の地理", "日本の歴史", "東京", "経済学", "物理学"
]

def fetch_wikipedia_summary_and_content(title: str) -> Dict[str, str]:
    """
    Fetches article text from Japanese Wikipedia API.
    """
    url = "https://ja.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "exintro": False
    }
    headers = {
        "User-Agent": "NexusAI-Researcher/1.0 (https://github.com/syuum/NexusAI; syuum@example.com)"
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if page_id != "-1":
                return {
                    "title": title,
                    "content": page_data.get("extract", "")
                }
    except Exception as e:
        print(f"[Error fetching {title}]: {e}")
    return {"title": title, "content": ""}

def harvest_wikipedia_corpus(output_dir: str, topics: List[str] = None):
    if topics is None:
        topics = SAMPLE_TOPICS
    os.makedirs(output_dir, exist_ok=True)
    all_articles_path = os.path.join(output_dir, "ja_wikipedia_corpus.txt")
    
    print(f"[Wiki-Harvest] Fetching {len(topics)} representative Japanese Wikipedia articles...")
    total_chars = 0
    with open(all_articles_path, "w", encoding="utf-8") as out_f:
        for i, topic in enumerate(topics):
            print(f"[{i+1}/{len(topics)}] Fetching: {topic}...")
            art = fetch_wikipedia_summary_and_content(topic)
            content = art["content"]
            if content:
                # Clean up whitespace and section markers
                clean_lines = [l.strip() for l in content.split("\n") if l.strip() and not l.startswith("==")]
                clean_text = "\n".join(clean_lines[:50]) # First 50 paragraphs
                out_f.write(f"# {art['title']}\n{clean_text}\n\n")
                total_chars += len(clean_text)
            time.sleep(0.5) # Polite rate limiting
            
    print(f"[Wiki-Harvest] Complete! Saved {total_chars:,} characters to {all_articles_path}")
    return all_articles_path

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    harvest_wikipedia_corpus(out_dir, topics=SAMPLE_TOPICS[:5]) # Fetch 5 key topics for validation
