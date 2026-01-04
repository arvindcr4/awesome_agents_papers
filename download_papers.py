#!/usr/bin/env python3
"""
Download all papers from the awesome-agents repository.
Uses Groq LLM for smart filename generation from paper titles.
"""

import os
import re
import time
import requests
from urllib.parse import urlparse
from pathlib import Path

# Paper list from awesome-agents repository
PAPERS = [
    # Inference-Time Techniques
    ("Large Language Models as Optimizers", "https://arxiv.org/abs/2309.03409"),
    ("Large Language Models Cannot Self-Correct Reasoning Yet", "https://arxiv.org/abs/2310.01798"),
    ("Teaching Large Language Models to Self-Debug", "https://arxiv.org/abs/2304.05128"),
    ("Chain-of-Thought Reasoning Without Prompting", "https://arxiv.org/abs/2402.10200"),
    ("Premise Order Matters in Reasoning with Large Language Models", "https://arxiv.org/abs/2402.08939"),
    ("Chain-of-Thought Empowers Transformers to Solve Inherently Serial Problems", "https://arxiv.org/abs/2402.12875"),

    # Post-Training & Alignment
    ("Direct Preference Optimization", "https://arxiv.org/abs/2305.18290"),
    ("Iterative Reasoning Preference Optimization", "https://arxiv.org/abs/2404.19733"),
    ("Chain-of-Verification Reduces Hallucination", "https://arxiv.org/abs/2309.11495"),
    ("Unpacking DPO and PPO", "https://arxiv.org/abs/2406.09279"),

    # Memory & Planning
    ("Grokked Transformers are Implicit Reasoners", "https://arxiv.org/abs/2405.15071"),
    ("HippoRAG Neurobiologically Inspired Long-Term Memory", "https://arxiv.org/abs/2405.14831"),
    ("Is Your LLM Secretly a World Model of the Internet", "https://arxiv.org/abs/2411.06559"),
    ("Tree Search for Language Model Agents", "https://arxiv.org/abs/2407.07614"),

    # Agent Frameworks
    ("ReAct Synergizing Reasoning and Acting", "https://arxiv.org/abs/2210.03629"),
    ("AutoGen Multi-Agent Conversation", "https://arxiv.org/abs/2308.08155"),
    ("StateFlow Enhancing LLM Task-Solving", "https://arxiv.org/abs/2403.11322"),

    # Code Generation & Software Agents
    ("SWE-agent Agent-Computer Interfaces", "https://arxiv.org/abs/2405.15793"),
    ("OpenHands AI Software Developers", "https://arxiv.org/abs/2407.16741"),
    ("Interactive Tools Assist LM Agents Security Vulnerabilities", "https://arxiv.org/abs/2409.16165"),

    # Web & Multimodal Agents
    ("WebShop Scalable Real-World Web Interaction", "https://arxiv.org/abs/2207.01206"),
    ("Mind2Web Generalist Agent for the Web", "https://arxiv.org/abs/2306.06070"),
    ("WebArena Realistic Web Environment", "https://arxiv.org/abs/2307.13854"),
    ("OSWORLD Benchmarking Multimodal Agents", "https://arxiv.org/abs/2404.07972"),
    ("AGUVIS Unified Pure Vision Agents GUI", "https://arxiv.org/abs/2412.04454"),

    # Enterprise & Workflow Agents
    ("WorkArena Common Knowledge Work Tasks", "https://arxiv.org/abs/2403.07718"),
    ("WorkArena++ Compositional Planning", "https://arxiv.org/abs/2407.05291"),
    ("TapeAgents Holistic Framework Agent Development", "https://arxiv.org/abs/2410.02536"),

    # Mathematics & Theorem Proving
    ("LeanDojo Theorem Proving Retrieval-Augmented", "https://arxiv.org/abs/2306.15626"),
    ("Autoformalization with Large Language Models", "https://arxiv.org/abs/2205.12615"),
    ("Autoformalizing Euclidean Geometry", "https://arxiv.org/abs/2405.17216"),
    ("Draft Sketch and Prove Formal Theorem Provers", "https://arxiv.org/abs/2210.12283"),
    ("miniCTX Neural Theorem Proving Long-Contexts", "https://arxiv.org/abs/2408.03350"),
    ("Lean-STaR Interleave Thinking and Proving", "https://arxiv.org/abs/2407.10040"),
    ("ImProver Agent-Based Automated Proof Optimization", "https://arxiv.org/abs/2410.04753"),
    ("In-Context Learning Agent Formal Theorem-Proving", "https://arxiv.org/abs/2310.04353"),
    ("Symbolic Regression Learned Concept Library", "https://arxiv.org/abs/2409.09359"),

    # Robotics & Embodied Agents
    ("Voyager Open-Ended Embodied Agent", "https://arxiv.org/abs/2305.16291"),
    ("Eureka Human-Level Reward Design", "https://arxiv.org/abs/2310.12931"),
    ("DrEureka Language Model Guided Sim-To-Real", "https://arxiv.org/abs/2406.01967"),

    # Scientific Discovery
    ("Paper2Agent Research Papers as AI Agents", "https://arxiv.org/abs/2509.06917"),
    ("OpenScholar Synthesizing Scientific Literature", "https://arxiv.org/abs/2411.14199"),

    # Safety & Security
    ("DataSentinel Game-Theoretic Detection Prompt Injection", "https://arxiv.org/abs/2504.11358"),
    ("AgentPoison Red-teaming LLM Agents", "https://arxiv.org/abs/2407.12784"),
    ("Progent Programmable Privilege Control", "https://arxiv.org/abs/2504.11703"),
    ("DecodingTrust Trustworthiness GPT Models", "https://arxiv.org/abs/2306.11698"),
    ("Representation Engineering AI Transparency", "https://arxiv.org/abs/2310.01405"),

    # Evaluation & Benchmarking
    ("Survey Evaluation LLM-based Agents", "https://arxiv.org/abs/2503.16416"),
    ("Adding Error Bars to Evals", "https://arxiv.org/abs/2411.00640"),
    ("Tau2-Bench Conversational Agents Dual-Control", "https://arxiv.org/abs/2506.07982"),

    # Neural & Symbolic Reasoning
    ("Beyond A-Star Better Planning Transformers", "https://arxiv.org/abs/2402.14083"),
    ("Dualformer Controllable Fast and Slow Thinking", "https://arxiv.org/abs/2410.09918"),
    ("Composing Global Optimizers Algebraic Objects", "https://arxiv.org/abs/2410.01779"),
    ("SurCo Learning Linear Surrogates", "https://arxiv.org/abs/2210.12547"),
]

def sanitize_filename(title):
    """Convert paper title to safe filename."""
    # Remove special characters and replace spaces
    safe = re.sub(r'[^\w\s-]', '', title)
    safe = re.sub(r'\s+', '_', safe.strip())
    return safe[:80]  # Limit length

def get_arxiv_pdf_url(url):
    """Convert arxiv abstract URL to PDF URL."""
    if 'arxiv.org/abs/' in url:
        arxiv_id = url.split('/abs/')[-1]
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    elif 'arxiv.org/pdf/' in url:
        if not url.endswith('.pdf'):
            return url + '.pdf'
        return url
    return None

def download_paper(title, url, output_dir):
    """Download a single paper."""
    filename = sanitize_filename(title) + ".pdf"
    filepath = os.path.join(output_dir, filename)

    # Skip if already downloaded
    if os.path.exists(filepath) and os.path.getsize(filepath) > 10000:
        print(f"  [SKIP] Already exists: {filename}")
        return True

    # Get PDF URL for arxiv
    pdf_url = get_arxiv_pdf_url(url)
    if not pdf_url:
        print(f"  [SKIP] Not an arxiv paper: {url}")
        return False

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/pdf,*/*',
    }

    try:
        print(f"  [DOWNLOADING] {title[:50]}...")
        response = requests.get(pdf_url, headers=headers, timeout=60, stream=True)
        response.raise_for_status()

        # Check if it's actually a PDF
        content_type = response.headers.get('content-type', '')
        if 'pdf' not in content_type.lower() and not response.content[:4] == b'%PDF':
            print(f"  [ERROR] Not a PDF: {content_type}")
            return False

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  [SUCCESS] {filename} ({size_mb:.1f} MB)")
        return True

    except requests.exceptions.RequestException as e:
        print(f"  [ERROR] {title}: {str(e)[:50]}")
        return False

def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\n{'='*60}")
    print("AWESOME AGENTS PAPER DOWNLOADER")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Total papers to download: {len(PAPERS)}")
    print(f"{'='*60}\n")

    success_count = 0
    skip_count = 0
    fail_count = 0

    for i, (title, url) in enumerate(PAPERS, 1):
        print(f"\n[{i}/{len(PAPERS)}] {title}")
        result = download_paper(title, url, output_dir)
        if result:
            if os.path.exists(os.path.join(output_dir, sanitize_filename(title) + ".pdf")):
                success_count += 1
        else:
            if 'arxiv.org' not in url:
                skip_count += 1
            else:
                fail_count += 1

        # Rate limiting - be nice to arxiv
        time.sleep(1)

    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully downloaded: {success_count}")
    print(f"Skipped (non-arxiv): {skip_count}")
    print(f"Failed: {fail_count}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
