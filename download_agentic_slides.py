#!/usr/bin/env python3
"""
Download slides for Agentic RL papers from the synthesis document.
"""

import os
import re
import time
import requests

SLIDES = [
    # Stanford RL for Agents (excellent!)
    ("Stanford_RL_for_Agents_2025", "https://web.stanford.edu/class/cs329t/slides/Lecture%208%20-%20[10_23_2025]%20Reinforcement%20Learning%20%20for%20Agents.pdf"),

    # LLM Agents Tool Learning Tutorial
    ("LLM_Agents_Tool_Learning_Tutorial", "https://llmagenttutorial.github.io/files/background_slides.pdf"),

    # CMU Language Models as Agents
    ("CMU_Language_Models_as_Agents", "https://phontron.com/class/anlp2024/assets/slides/anlp-20-agents.pdf"),

    # Mannheim LLM Agents Tool Use
    ("Mannheim_LLM_Agents_Tool_Use", "https://www.uni-mannheim.de/media/Einrichtungen/dws/Files_Teaching/Large_Language_Models_and_Agents/FSS2025/IE686_LA_04_LLMAgentsAndToolUse.pdf"),

    # Multi-Agent RL Introduction (Edinburgh)
    ("Edinburgh_Multi_Agent_RL_Intro", "https://opencourse.inf.ed.ac.uk/sites/default/files/https/opencourse.inf.ed.ac.uk/rl/2025/marlintro2025_0.pdf"),

    # ARTIST paper (Microsoft)
    ("ARTIST_Agentic_Reasoning_Microsoft", "https://www.microsoft.com/en-us/research/wp-content/uploads/2025/04/AgenticReasoning.pdf"),

    # Intel AI Agents Architecture Guide
    ("Intel_AI_Agents_Architecture", "https://cdrdv2-public.intel.com/853582/AI%20Agents%20and%20Architecture-Resource-Guide.pdf"),

    # Cisco Agentic Frameworks Overview
    ("Cisco_Agentic_Frameworks_Overview", "https://www.ciscolive.com/c/dam/r/ciscolive/emea/docs/2025/pdf/AIHUB-2170.pdf"),

    # MasRouter ACL paper
    ("MasRouter_ACL_2025", "https://aclanthology.org/2025.acl-long.757.pdf"),

    # DigiRL NeurIPS paper
    ("DigiRL_NeurIPS_2024", "https://proceedings.neurips.cc/paper_files/paper/2024/file/1704ddd0bb89f159dfe609b32c889995-Paper-Conference.pdf"),

    # OS-Harm Benchmark
    ("OS_Harm_Benchmark", "https://arxiv.org/pdf/2506.14866"),

    # PTA-GRPO Planning paper
    ("PTA_GRPO_High_Level_Planning", "https://arxiv.org/pdf/2510.01833"),
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
        response = requests.get(url, headers=headers, timeout=90, allow_redirects=True)

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
    print("AGENTIC RL SLIDES DOWNLOADER")
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
