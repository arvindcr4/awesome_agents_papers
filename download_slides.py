#!/usr/bin/env python3
"""
Download presentation slides for awesome-agents papers.
"""

import os
import re
import time
import requests

# Slide PDFs found via Firecrawl search
SLIDES = [
    # ReAct
    ("ReAct_UVA_Lecture", "https://yumeng5.github.io/teaching/2024-spring-cs6501/agent.pdf"),

    # Chain of Thought
    ("CoT_Interpretable_ML_Lecture", "https://interpretable-ml-class.github.io/slides/Lecture_23_CoT.pdf"),
    ("CoT_Toronto_Presentation", "https://www.cs.toronto.edu/~cmaddis/courses/csc2541_w25/presentations/dai_chen_cot.pdf"),
    ("CoT_Princeton_Lecture", "https://www.cs.princeton.edu/courses/archive/fall22/cos597G/lectures/lec09.pdf"),
    ("CoT_SJTU_Slides", "https://bcmi.sjtu.edu.cn/home/zhangzs/slides/CoT-zhuosheng.pdf"),
    ("Concise_CoT_Benefits", "https://matthewrenze.com/research/the-benefits-of-a-concise-chain-of-thought/slides.pdf"),

    # SWE-agent and Code Agents
    ("Software_Agents_Neubig", "http://www.phontron.com/slides/neubig24softwareagents.pdf"),
    ("Code_Agents_Vulnerability_Detection_Berkeley", "https://rdi.berkeley.edu/adv-llm-agents/slides/Code%20Agents%20and%20AI%20for%20Vulnerability%20Detection.pdf"),

    # WebArena
    ("Multimodal_Agents_Berkeley", "https://rdi.berkeley.edu/adv-llm-agents/slides/ruslan-multimodal.pdf"),
    ("Web_Agent_Evaluation_Refinement", "https://marworkshop.github.io/cvpr24/pdf/18.pdf"),

    # DPO - Direct Preference Optimization
    ("DPO_Stanford_Lecture", "https://web.stanford.edu/class/cs234/CS234Spr2024/slides/dpo_slides.pdf"),
    ("DPO_CMU_Lecture", "https://www.cs.cmu.edu/~mgormley/courses/10423//slides/lecture12-dpo-text2img.pdf"),
    ("DPO_UT_Austin", "https://ut.philkr.net/advances_in_deeplearning/large_language_models/dpo/slides.pdf"),
    ("DPO_Toronto_Presentation", "https://www.cs.toronto.edu/~cmaddis/courses/csc2541_w25/presentations/mu_cao_dpo.pdf"),
    ("DPO_Jinen_Slides", "https://jinen.setpal.net/slides/dpo.pdf"),

    # LLM Agents Tutorials
    ("EMNLP2024_Language_Agents_Tutorial", "https://language-agent-tutorial.github.io/slides/I-Introduction.pdf"),
    ("WWW2024_LLM_Agents_Tutorial", "https://www2024.thewebconf.org/docs/tutorial-slides/large-language-model-powered-agents.pdf"),
    ("Berkeley_LLM_Training_Agents", "https://rdi.berkeley.edu/agentic-ai/slides/lecture1.pdf"),

    # Voyager
    ("Voyager_UT_Austin_Presentation", "https://www.cs.utexas.edu/~yukez/cs391r_fall2023/slides/pre_10-31_Yuqi.pdf"),

    # HippoRAG
    ("HippoRAG_NeurIPS_Slides", "https://nips.cc/media/neurips-2024/Slides/94043.pdf"),

    # LeanDojo and Theorem Proving
    ("LeanDojo_AITP_Slides", "http://aitp-conference.org/2023/slides/KY.pdf"),
    ("LeanDojo_NeurIPS_Slides", "https://neurips.cc/media/neurips-2023/Slides/73738.pdf"),
    ("Theorem_Proving_ML_Slides", "https://lftcm2023.github.io/slides/Kaiyu_Yang_TheoremProvingViaMachineLearning.pdf"),

    # RLHF
    ("RLHF_UT_Austin_Slides", "https://ut.philkr.net/advances_in_deeplearning/large_language_models/rlhf/slides.pdf"),
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

        # Check if PDF
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
    print("SLIDES DOWNLOADER")
    print(f"{'='*60}")
    print(f"Output: {output_dir}")
    print(f"Slides to download: {len(SLIDES)}")
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
