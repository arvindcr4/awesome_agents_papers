#!/usr/bin/env python3
"""
Download additional presentation slides.
"""

import os
import re
import time
import requests

# Additional slides found
SLIDES = [
    # Agent Benchmarking
    ("AgentBench_Multi_Turn_NeurIPS", "https://neurips.cc/media/neurips-2024/Slides/98026.pdf"),
    ("Data_Science_Agents_Benchmark", "https://aclanthology.org/2024.acl-long.308.pdf"),

    # Prompt Injection & Security
    ("Prompt_Injection_Duke_Slides", "https://people.duke.edu/~zg70/code/PromptInjection.pdf"),

    # Eureka Robotics
    ("Eureka_Reward_Design_Paper", "https://eureka-research.github.io/assets/eureka_paper.pdf"),

    # Additional Berkeley RDI slides
    ("Berkeley_Agents_Overview", "https://rdi.berkeley.edu/adv-llm-agents/slides/xinyun-overview.pdf"),
    ("Berkeley_Web_Agents", "https://rdi.berkeley.edu/adv-llm-agents/slides/shunyu-web-agents.pdf"),
    ("Berkeley_Tool_Use", "https://rdi.berkeley.edu/adv-llm-agents/slides/yujia-tool-use.pdf"),

    # Additional tutorials
    ("ACL2024_Agents_Tutorial", "https://language-agent-tutorial.github.io/slides/II-Reasoning.pdf"),
    ("ACL2024_Agents_Acting", "https://language-agent-tutorial.github.io/slides/III-Acting.pdf"),
    ("ACL2024_Agents_Learning", "https://language-agent-tutorial.github.io/slides/IV-Learning.pdf"),
]

def sanitize_filename(name):
    safe = re.sub(r'[^\w\s-]', '', name)
    safe = re.sub(r'\s+', '_', safe.strip())
    return safe

def download_slide(name, url, output_dir):
    filename = sanitize_filename(name) + ".pdf"
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath) and os.path.getsize(filepath) > 5000:
        print(f"  [SKIP] Already exists: {filename}")
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
        print(f"  [SUCCESS] {filename} ({size_mb:.1f} MB)")
        return True

    except Exception as e:
        print(f"  [ERROR] {str(e)[:50]}")
        return False

def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "slides")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("ADDITIONAL SLIDES DOWNLOADER")
    print(f"{'='*60}\n")

    success = 0
    for i, (name, url) in enumerate(SLIDES, 1):
        print(f"[{i}/{len(SLIDES)}] {name}")
        if download_slide(name, url, output_dir):
            success += 1
        time.sleep(0.5)

    print(f"\n{'='*60}")
    print(f"Downloaded: {success}/{len(SLIDES)} slides")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
