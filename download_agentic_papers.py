#!/usr/bin/env python3
"""
Download papers from the Agentic RL synthesis document.
"""

import os
import re
import time
import requests

PAPERS = [
    # AgentNet (NeurIPS 2025)
    ("AgentNet Decentralized Multi-Agent", "https://arxiv.org/abs/2505.08881"),

    # DigiRL - already have the paper but let's ensure
    ("DigiRL Device Control Agents", "https://arxiv.org/abs/2406.11896"),

    # FireAct
    ("FireAct Language Agent Fine-tuning", "https://arxiv.org/abs/2310.05915"),

    # MasRouter
    ("MasRouter Multi-Agent Routing", "https://arxiv.org/abs/2502.11133"),

    # OSWorld
    ("OSWorld Multimodal Agents Benchmark", "https://arxiv.org/abs/2404.07972"),

    # DeepSeek-OCR / Janus-Pro with OCR
    ("DeepSeek Janus Pro Multimodal", "https://arxiv.org/abs/2501.02707"),

    # OS-Harm safety benchmark
    ("OS-Harm Computer Use Safety", "https://arxiv.org/abs/2506.14866"),
]

def sanitize_filename(title):
    safe = re.sub(r'[^\w\s-]', '', title)
    safe = re.sub(r'\s+', '_', safe.strip())
    return safe[:80]

def get_arxiv_pdf_url(url):
    if 'arxiv.org/abs/' in url:
        arxiv_id = url.split('/abs/')[-1]
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    return None

def download_paper(title, url, output_dir):
    filename = sanitize_filename(title) + ".pdf"
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath) and os.path.getsize(filepath) > 10000:
        print(f"  [SKIP] Already exists: {filename}")
        return True

    pdf_url = get_arxiv_pdf_url(url)
    if not pdf_url:
        print(f"  [SKIP] Not an arxiv paper")
        return False

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    }

    try:
        print(f"  [DOWNLOADING] {title}...")
        response = requests.get(pdf_url, headers=headers, timeout=60)
        response.raise_for_status()

        if not response.content[:4] == b'%PDF':
            print(f"  [ERROR] Not a PDF")
            return False

        with open(filepath, 'wb') as f:
            f.write(response.content)

        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  [SUCCESS] {filename} ({size_mb:.1f} MB)")
        return True

    except Exception as e:
        print(f"  [ERROR] {str(e)[:50]}")
        return False

def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\n{'='*60}")
    print("AGENTIC SYNTHESIS PAPERS")
    print(f"{'='*60}\n")

    success = 0
    for i, (title, url) in enumerate(PAPERS, 1):
        print(f"[{i}/{len(PAPERS)}] {title}")
        if download_paper(title, url, output_dir):
            success += 1
        time.sleep(1)

    print(f"\n{'='*60}")
    print(f"Downloaded: {success}/{len(PAPERS)}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
