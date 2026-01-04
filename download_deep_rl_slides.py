#!/usr/bin/env python3
"""
Download slides for Deep RL papers and algorithms.
"""

import os
import re
import time
import requests

SLIDES = [
    # DQN & Value-Based Methods
    ("DQN_CMU_Deep_Q_Learning", "https://andrew.cmu.edu/course/10-403/slides/S19_lecture9_DQL.pdf"),
    ("DQN_CVUT_Q_Learning", "https://cw.fel.cvut.cz/b222/_media/courses/zui/slides-l6-2023.pdf"),
    ("DQN_NTHU_Deep_RL", "https://nthu-datalab.github.io/ml/slides/15_Deep-Reinforcement-Learning.pdf"),
    ("DQN_Waterloo_CS885", "https://cs.uwaterloo.ca/~ppoupart/teaching/cs885-winter25/slides/cs885-lecture3b-annotated.pdf"),
    ("Dueling_DQN_PER_Buffalo", "https://cse.buffalo.edu/~avereshc/rl_fall19/lecture_14_1_Dueling_DQN_and_PER.pdf"),

    # Rainbow & Advanced DQN
    ("Rainbow_Prague_NPFL122", "https://ufal.mff.cuni.cz/~straka/courses/npfl122/2021/slides.pdf/npfl122-2021-05.pdf"),
    ("Rainbow_Berkeley_Off_Policy", "https://people.eecs.berkeley.edu/~pabbeel/cs287-fa19/slides/Lec19-RL-II-off-policy.pdf"),
    ("Rainbow_Wisconsin_CS760", "https://pages.cs.wisc.edu/~jphanna/teaching/2023fall_cs760/slides/lec26-rl-IV.pdf"),

    # PPO & TRPO
    ("PPO_Waterloo_CS885", "https://cs.uwaterloo.ca/~ppoupart/teaching/cs885-spring18/slides/cs885-lecture15b.pdf"),
    ("PPO_NTU_Taiwan", "https://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2018/Lecture/PPO%20(v3).pdf"),
    ("PPO_Stanford_CS234", "https://web.stanford.edu/class/cs234/slides/lecture6post.pdf"),
    ("TRPO_FAU_Mutschler", "http://download.cmutschler.de/lectures/FAU_RL_2021/slides/7-02_TRPO.pdf"),
    ("TRPO_UT_Austin", "https://www.cs.utexas.edu/~yukez/cs391r_fall2023/slides/pre_09-26_Jonathan.pdf"),
    ("TRPO_CMU_Natural_PG", "https://www.andrew.cmu.edu/course/10-703/slides/Lecture_NaturalPolicyGradientsTRPOPPO.pdf"),
    ("TRPO_Toronto_PAIR", "https://www.pair.toronto.edu/csc2621-w20/assets/slides/lec3_trpo.pdf"),

    # A3C & Actor-Critic
    ("A3C_WPI_DS595", "https://users.wpi.edu/~yli15/courses/DS595CS525Fall20/Slides/RL-8_A3C.pdf"),
    ("A3C_Buffalo_Actor_Critic", "https://cse.buffalo.edu/~avereshc/rl_fall19/lecture_20_Actor_Critic_Methods.pdf"),
    ("A3C_NTU_Taiwan", "https://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2018/Lecture/AC.pdf"),
    ("A3C_UIUC_ECE448", "https://courses.grainger.illinois.edu/ece448/sp2022/slides/lec36.pdf"),
    ("A3C_Julien_Vitay", "https://julien-vitay.net/course-deeprl/slides/pdf/3.4-A3C.pdf"),

    # DDPG, TD3, SAC
    ("DDPG_Paderborn_DPG", "https://groups.uni-paderborn.de/lea/share/lehre/reinforcementlearning/lecture_slides/built/Lecture12.pdf"),
    ("DDPG_FAU_Mutschler", "http://download.cmutschler.de/lectures/FAU_RL_2021/slides/7-04_DDPG.pdf"),
    ("DDPG_Julien_Vitay", "https://julien-vitay.net/course-deeprl/slides/pdf/3.5-DDPG.pdf"),
    ("DDPG_Buffalo_DPG", "https://cse.buffalo.edu/~avereshc/rl_fall19/lecture_21_Actor_Critic_DPG_DDPG.pdf"),
    ("TD3_SAC_Prague_NPFL139", "https://ufal.mff.cuni.cz/~straka/courses/npfl139/2324/slides.pdf/npfl139-2324-08.pdf"),
    ("SAC_Toronto_PAIR", "https://www.pair.toronto.edu/csc2621-w20/assets/slides/lec4_sac.pdf"),
    ("SAC_Purdue_RL_Inference", "https://www.stat.purdue.edu/~wang4094/resources/slides/RL_as_inference__RLSC_seminar_slides.pdf"),
    ("SAC_Stanford_CS231n", "https://cs231n.stanford.edu/slides/2020/lecture_17.pdf"),

    # Policy Gradient & REINFORCE
    ("Policy_Gradient_Toronto", "https://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/slides/lec20.pdf"),
    ("Policy_Gradient_Berkeley_CS285", "https://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_4_policy_gradient.pdf"),
    ("REINFORCE_Stanford_CS229", "https://cs229.stanford.edu/notes2020fall/notes2020fall/cs229-notes14.pdf"),
    ("Policy_Gradient_Michigan_EECS498", "https://web.eecs.umich.edu/~justincj/slides/eecs498/FA2020/598_FA2020_lecture21.pdf"),

    # GAE & Advantage Estimation
    ("GAE_Stanford_CS234", "https://web.stanford.edu/class/cs234/slides/lecture7post.pdf"),
    ("GAE_Berkeley_CS285", "https://rail.eecs.berkeley.edu/deeprlcourse-fa20/static/slides/lec-6.pdf"),

    # Temporal Difference & Q-Learning
    ("TD_Learning_CMU", "https://andrew.cmu.edu/course/10-403/slides/S19_lecture5_TD.pdf"),
    ("TD_Methods_Michigan", "https://www.ambujtewari.com/stats701-winter2021/slides/TD-methods.pdf"),
    ("Q_Learning_Northeastern", "https://www.ccs.neu.edu/home/dmklee/cs4910_s22/slides/reinforcement_learning_qlearning.pdf"),
    ("TD_Sutton_Barto", "https://web.stanford.edu/class/cme241/lecture_slides/rich_sutton_slides/11-12-TD.pdf"),
    ("SARSA_Q_Learning_PSU", "https://www.cse.psu.edu/~mzm616/courses/cmpsc448/slides/22.%20Sarsa%20and%20Q-learning.pdf"),

    # Model-Based RL
    ("Model_Based_RL_FAU", "https://download.cmutschler.de/lectures/FAU_RL_2023/slides/09%20Model-based%20RL%201.pdf"),
    ("Model_Based_RL_Toronto", "http://www.cs.toronto.edu/~rgrosse/courses/csc2515_2019/tutorials/tut10/tutorial10-slides.pdf"),
    ("Model_Based_RL_Berkeley", "https://people.eecs.berkeley.edu/~pabbeel/cs287-fa19/slides/Lec20-RL-III-model-based.pdf"),
    ("Model_Based_RL_CMU", "https://andrew.cmu.edu/course/10-403/slides/modelbasedRL_I_2019.pdf"),

    # Imitation & Inverse RL
    ("Imitation_Learning_WPI", "https://users.wpi.edu/~yli15/courses/DS595Spring22/Slides/RL-8_PG-v3.pdf"),
    ("Imitation_Learning_EPFL", "https://www.epfl.ch/labs/lions/wp-content/uploads/2024/08/Lecture-6_-Imitation-Learning.pdf"),
    ("Imitation_Learning_Stanford_CS234", "https://web.stanford.edu/class/cs234/slides/lecture8post.pdf"),
    ("Inverse_RL_TU_Darmstadt", "https://www.ias.informatik.tu-darmstadt.de/uploads/Teaching/Colombia2015/L10.pdf"),
    ("Inverse_RL_Berkeley_CS285", "https://rail.eecs.berkeley.edu/deeprlcourse-fa20/static/slides/lec-20.pdf"),

    # Berkeley CS285 Core Lectures
    ("Berkeley_CS294_Intro", "https://rail.eecs.berkeley.edu/deeprlcourse-fa18/static/slides/lec-1.pdf"),
    ("Berkeley_CS294_Intro_2017", "https://rll.berkeley.edu/deeprlcourse/f17docs/lecture_1_introduction.pdf"),

    # Experience Replay
    ("Experience_Replay_ICML2020", "https://icml.cc/media/icml-2020/Slides/6751.pdf"),
    ("PER_Julien_Vitay", "https://julien-vitay.net/course-deeprl/slides/pdf/3.1-DQN.pdf"),
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
    print("DEEP RL ALGORITHM SLIDES DOWNLOADER")
    print(f"{'='*60}\n")

    success = 0
    for i, (name, url) in enumerate(SLIDES, 1):
        print(f"[{i}/{len(SLIDES)}] {name}")
        if download_slide(name, url, output_dir):
            success += 1
        time.sleep(0.3)

    print(f"\n{'='*60}")
    print(f"Downloaded: {success}/{len(SLIDES)}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
