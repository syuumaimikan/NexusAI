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
    # 1. コンピュータ・情報科学
    "人工知能", "機械学習", "深層学習", "自然言語処理", "大規模言語モデル",
    "量子コンピュータ", "オペレーティングシステム", "コンパイラ",
    "Python", "プログラミング言語", "Linux", "分散コンピューティング",
    "インターネット", "公開鍵暗号", "アルゴリズム", "半導体",
    # 2. 日本の歴史・文化
    "日本の歴史", "日本の地理", "東京",
    "縄文時代", "弥生時代", "古墳時代", "飛鳥時代", "奈良時代", "平安時代",
    "鎌倉時代", "室町時代", "戦国時代 (日本)", "江戸時代", "明治", "大正", "昭和", "平成", "令和",
    # 3. 世界史・国際社会
    "世界史", "産業革命", "ルネサンス", "大航海時代", "フランス革命", "国際連合", "古代エジプト", "メソポタミア",
    # 4. 自然科学・物理・化学・生物
    "物理学", "相対性理論", "量子力学", "原子", "周期表", "化学", "DNA", "進化論", "細胞",
    # 5. 宇宙・地球・地理
    "宇宙", "太陽系", "地球", "ビッグバン", "ブラックホール", "気候", "海洋",
    # 6. 政治・経済・社会
    "経済学", "市場経済", "インフレーション", "三権分立", "日本国憲法", "資本主義", "民主主義"
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
                clean_text = "\n".join(clean_lines[:150]) # Extract up to 150 paragraphs
                out_f.write(f"# {art['title']}\n{clean_text}\n\n")
                total_chars += len(clean_text)
            time.sleep(0.15) # Polite rate limiting
            
    print(f"[Wiki-Harvest] Complete! Saved {total_chars:,} characters to {all_articles_path}")
    return all_articles_path

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    harvest_wikipedia_corpus(out_dir, topics=SAMPLE_TOPICS)
