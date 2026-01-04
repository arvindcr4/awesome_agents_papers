#!/usr/bin/env python3
"""
Download slides for RL fine-tuning and reasoning papers.
"""

import os
import re
import time
import requests

SLIDES = [
    # DeepSeek R1 slides
    ("DeepSeek_R1_Introduction", "https://anoopkunchukuttan.gitlab.io/publications/presentations/DeepSeek-OSSProjects-Intro-Feb2025.pdf"),
    ("DeepSeek_R1_Toronto", "https://www.cs.toronto.edu/~cmaddis/courses/csc2541_w25/presentations/ivanov_farhat_deepseek.pdf"),
    ("DeepSeek_R1_CMU_Reasoning", "http://www.cs.cmu.edu/~mgormley/courses/10423-s25//slides/lecture26-reasoning.pdf"),
    ("DeepSeek_R1_Seoul_National", "https://idea.snu.ac.kr/wp-content/uploads/sites/6/2025/02/DeepSeek_r1%EB%B0%95%EC%84%B8%ED%98%84.pdf"),

    # RL for Reasoning
    ("Stanford_RL_for_LLM_Reasoning", "https://cs224r.stanford.edu/slides/10_cs224r-rl_for_reasoning_lecture.pdf"),
    ("PTA_GRPO_Planning_Reasoning", "https://arxiv.org/pdf/2510.01833"),
]

def sanitize_filename(name):
    safe = re.sub(r'[^\w\s-]', '', name)
    safe = re.sub(r'\s+', '_', safe.strip())
    return safe

def download_slide(name, url, output_dir):
    filename = sanitize_filename(name) + ".pdf"
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath) and os.path.getsize(filepath) > 5000:
        print(f"  [SKIP] Already exists")
        return True

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'application/pdf,*/*',
    }

    try:
        print(f"  [DOWNLOADING] {name}...")
        response = requests.get(url, headers=headers, timeout=60, allow_redirects=True)

        if response.status_code != 200:
            print(f"  [ERROR] HTTP {response.status_code}")
            return False

        if not response.content[:4] == b'%PDF':
            print(f"  [SKIP] Not a PDF")
            return False

        with open(filepath, 'wb') as f:
            f.write(response.content)

        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  [SUCCESS] ({size_mb:.1f} MB)")
        return True

    except Exception as e:
        print(f"  [ERROR] {str(e)[:50]}")
        return False

def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "slides")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("RL REASONING SLIDES DOWNLOADER")
    print(f"{'='*60}\n")

    success = 0
    for i, (name, url) in enumerate(SLIDES, 1):
        print(f"[{i}/{len(SLIDES)}] {name}")
        if download_slide(name, url, output_dir):
            success += 1
        time.sleep(0.5)

    print(f"\n{'='*60}")
    print(f"Downloaded: {success}/{len(SLIDES)}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
