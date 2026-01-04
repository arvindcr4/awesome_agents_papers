#!/usr/bin/env python3
"""
Download remaining papers - finding arXiv preprints and alternative sources.
"""

import os
import re
import time
import requests

# Additional papers to try - found arXiv versions or alternative sources
REMAINING_PAPERS = [
    # Gran Turismo - arXiv preprint
    ("Gran Turismo Deep Reinforcement Learning", "https://arxiv.org/pdf/2008.07971.pdf"),

    # AlphaProof related - formal reasoning papers
    ("AlphaGeometry Solving Olympiad Geometry", "https://arxiv.org/pdf/2401.04890.pdf"),

    # DSPy framework paper
    ("DSPy Compiling Declarative Language Model", "https://arxiv.org/pdf/2310.03714.pdf"),

    # BrowseComp - try to find related benchmark paper
    ("BrowseComp Web Browsing Benchmark", "https://arxiv.org/pdf/2504.09474.pdf"),

    # SLAC robotics paper
    ("SLAC Simulation-Pretrained Latent Action", "https://arxiv.org/pdf/2410.09816.pdf"),

    # Project GR00T related - humanoid robot papers
    ("GR00T N1 Foundation Model Humanoid", "https://arxiv.org/pdf/2503.14734.pdf"),

    # Naptime/Big Sleep security paper
    ("Big Sleep LLM Vulnerabilities Real-World", "https://arxiv.org/pdf/2411.00176.pdf"),
]

def sanitize_filename(title):
    safe = re.sub(r'[^\w\s-]', '', title)
    safe = re.sub(r'\s+', '_', safe.strip())
    return safe[:80]

def download_paper(title, url, output_dir):
    filename = sanitize_filename(title) + ".pdf"
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath) and os.path.getsize(filepath) > 10000:
        print(f"  [SKIP] Already exists: {filename}")
        return True

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'application/pdf,*/*',
    }

    try:
        print(f"  [DOWNLOADING] {title[:50]}...")
        response = requests.get(url, headers=headers, timeout=60, stream=True)

        if response.status_code != 200:
            print(f"  [ERROR] HTTP {response.status_code}")
            return False

        first_bytes = response.content[:4] if len(response.content) >= 4 else b''
        if first_bytes != b'%PDF':
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
    output_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\n{'='*60}")
    print("REMAINING PAPERS DOWNLOADER")
    print(f"{'='*60}\n")

    success = 0
    for i, (title, url) in enumerate(REMAINING_PAPERS, 1):
        print(f"[{i}/{len(REMAINING_PAPERS)}] {title}")
        if download_paper(title, url, output_dir):
            success += 1
        time.sleep(1)

    print(f"\n{'='*60}")
    print(f"Downloaded: {success}/{len(REMAINING_PAPERS)}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
