#!/usr/bin/env python3
"""
Download papers from agentic-reasoning-reinforcement-fine-tuning repo
and related GRPO/RL fine-tuning papers.
"""

import os
import re
import time
import requests

# Papers from agentic-reasoning repo + related papers
PAPERS = [
    # From redhat-et/agentic-reasoning-reinforcement-fine-tuning
    ("DeepSeekMath GRPO", "https://arxiv.org/abs/2402.03300"),
    ("WebAgent-R1 Multi-Turn RL", "https://arxiv.org/abs/2505.16421"),
    ("ARTIST Agentic Reasoning Tool Integration", "https://arxiv.org/abs/2505.01441"),

    # DeepSeek R1 series
    ("DeepSeek-R1 Reasoning via RL", "https://arxiv.org/abs/2501.12948"),
    ("DeepSeek R1 Implications for AI", "https://arxiv.org/abs/2502.02523"),
    ("DeepSeek R1 Reasoning Models Faithful", "https://arxiv.org/abs/2501.08156"),

    # RL Fine-tuning for Reasoning
    ("R-Search Multi-Step Reasoning", "https://arxiv.org/abs/2506.08352"),
    ("RL Fine-tuning Instruction Following", "https://arxiv.org/abs/2506.21560"),
    ("RFT Powers Multimodal Reasoning", "https://arxiv.org/abs/2505.18536"),
    ("Guided GRPO Adaptive Guidance", "https://arxiv.org/abs/2508.13023"),

    # Additional reasoning papers
    ("OpenAI O1 Replication Journey", "https://arxiv.org/abs/2410.18982"),
    ("Qwen QwQ Reasoning Model", "https://arxiv.org/abs/2412.15115"),
    ("Sky-T1 Training Small Reasoning LLMs", "https://arxiv.org/abs/2501.09606"),
    ("STILL-2 Distilling Reasoning", "https://arxiv.org/abs/2502.05171"),
    ("s1 Simple Test-Time Scaling", "https://arxiv.org/abs/2501.19393"),
]

def sanitize_filename(title):
    safe = re.sub(r'[^\w\s-]', '', title)
    safe = re.sub(r'\s+', '_', safe.strip())
    return safe[:80]

def get_arxiv_pdf_url(url):
    if 'arxiv.org/abs/' in url:
        arxiv_id = url.split('/abs/')[-1]
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    elif 'arxiv.org/pdf/' in url:
        if not url.endswith('.pdf'):
            return url + '.pdf'
        return url
    return None

def download_paper(title, url, output_dir):
    filename = sanitize_filename(title) + ".pdf"
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath) and os.path.getsize(filepath) > 10000:
        print(f"  [SKIP] Already exists: {filename}")
        return True

    pdf_url = get_arxiv_pdf_url(url)
    if not pdf_url:
        print(f"  [SKIP] Not an arxiv paper: {url}")
        return False

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'application/pdf,*/*',
    }

    try:
        print(f"  [DOWNLOADING] {title}...")
        response = requests.get(pdf_url, headers=headers, timeout=60, stream=True)
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
    print("AGENTIC REASONING & RL FINE-TUNING PAPERS")
    print(f"{'='*60}")
    print(f"Papers to download: {len(PAPERS)}")
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
