#!/usr/bin/env python3
"""
Download non-arXiv papers from the awesome-agents repository.
"""

import os
import re
import time
import requests
from urllib.parse import urlparse

# Non-arXiv papers with direct PDF links or findable PDFs
NON_ARXIV_PAPERS = [
    # Direct PDF links
    ("TapeAgents Holistic Framework", "https://rdi.berkeley.edu/llm-agents-mooc/assets/tapeagents.pdf"),
    ("Privtrans Privilege Separation", "https://dawnsong.io/papers/privtrans.pdf"),
    ("Extracting Training Data from LLMs", "https://www.usenix.org/system/files/sec21-carlini-extracting.pdf"),
    ("The Secret Sharer Unintended Memorization", "https://www.usenix.org/system/files/sec19-carlini.pdf"),

    # Nature papers - try sci-hub or open access versions
    ("Gran Turismo Deep RL", "https://www.nature.com/articles/s41586-021-04357-7.pdf"),
    ("Virtual Lab AI agents SARS-CoV-2", "https://www.nature.com/articles/s41586-025-09442-9.pdf"),

    # Try to find arxiv versions of some papers
    ("Tree Search Language Model Agents", "https://arxiv.org/pdf/2407.01476.pdf"),  # Related paper
    ("VisualWebArena", "https://arxiv.org/pdf/2401.13649.pdf"),

    # OpenAI blog posts - try to find related papers
    ("SWE-bench Verified", "https://arxiv.org/pdf/2310.06770.pdf"),  # Original SWE-bench paper
]

def sanitize_filename(title):
    """Convert paper title to safe filename."""
    safe = re.sub(r'[^\w\s-]', '', title)
    safe = re.sub(r'\s+', '_', safe.strip())
    return safe[:80]

def download_paper(title, url, output_dir):
    """Download a single paper."""
    filename = sanitize_filename(title) + ".pdf"
    filepath = os.path.join(output_dir, filename)

    # Skip if already downloaded
    if os.path.exists(filepath) and os.path.getsize(filepath) > 10000:
        print(f"  [SKIP] Already exists: {filename}")
        return True

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/pdf,*/*',
    }

    try:
        print(f"  [DOWNLOADING] {title[:50]}...")
        print(f"    URL: {url}")
        response = requests.get(url, headers=headers, timeout=60, stream=True, allow_redirects=True)

        if response.status_code == 403:
            print(f"  [BLOCKED] Access forbidden (paywall): {title}")
            return False
        elif response.status_code == 404:
            print(f"  [NOT FOUND] {title}")
            return False

        response.raise_for_status()

        # Check if it's actually a PDF
        content_type = response.headers.get('content-type', '')
        first_bytes = response.content[:4] if len(response.content) >= 4 else b''

        if first_bytes != b'%PDF':
            print(f"  [SKIP] Not a PDF (got {content_type[:30]})")
            return False

        with open(filepath, 'wb') as f:
            f.write(response.content)

        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  [SUCCESS] {filename} ({size_mb:.1f} MB)")
        return True

    except requests.exceptions.RequestException as e:
        print(f"  [ERROR] {title}: {str(e)[:50]}")
        return False

def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\n{'='*60}")
    print("NON-ARXIV PAPER DOWNLOADER")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Papers to try: {len(NON_ARXIV_PAPERS)}")
    print(f"{'='*60}\n")

    success_count = 0
    fail_count = 0

    for i, (title, url) in enumerate(NON_ARXIV_PAPERS, 1):
        print(f"\n[{i}/{len(NON_ARXIV_PAPERS)}] {title}")
        result = download_paper(title, url, output_dir)
        if result:
            success_count += 1
        else:
            fail_count += 1
        time.sleep(1)

    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully downloaded: {success_count}")
    print(f"Failed/Blocked: {fail_count}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
