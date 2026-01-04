# Deep Reinforcement Learning Resources

> Comprehensive collection of Deep RL resources curated from the Udacity Deep RL Nanodegree community.
> Original source: [bit.ly/drlndlinks](http://bit.ly/drlndlinks)

---

## Table of Contents
- [Courses & Tutorials](#courses--tutorials)
- [Books](#books)
- [Key Papers](#key-papers)
- [GitHub Repositories](#github-repositories)
- [OpenAI Resources](#openai-resources)
- [PyTorch & Frameworks](#pytorch--frameworks)
- [Unity ML-Agents](#unity-ml-agents)
- [Blogs](#blogs)
- [Videos & Lectures](#videos--lectures)
- [Cloud Platforms](#cloud-platforms)
- [Competitions](#competitions)
- [Cheatsheets & Glossaries](#cheatsheets--glossaries)
- [Articles](#articles)
- [Lecture Slides](#lecture-slides)

---

## Lecture Slides

> 📂 **92 curated lecture slides** from top universities available in the `slides/` folder (~504 MB)

### Value-Based Methods (DQN Family)
| Algorithm | Source | File |
|-----------|--------|------|
| DQN | CMU 10-403 | `slides/DQN_CMU_Deep_Q_Learning.pdf` |
| DQN | CVUT Q-Learning | `slides/DQN_CVUT_Q_Learning.pdf` |
| DQN | NTHU Deep RL | `slides/DQN_NTHU_Deep_RL.pdf` |
| DQN | Waterloo CS885 | `slides/DQN_Waterloo_CS885.pdf` |
| Dueling DQN + PER | U. Buffalo | `slides/Dueling_DQN_PER_Buffalo.pdf` |
| Rainbow | Prague NPFL122 | `slides/Rainbow_Prague_NPFL122.pdf` |
| Rainbow/Off-Policy | Berkeley CS287 | `slides/Rainbow_Berkeley_Off_Policy.pdf` |
| Rainbow | Wisconsin CS760 | `slides/Rainbow_Wisconsin_CS760.pdf` |
| PER (DQN Extensions) | Julien Vitay | `slides/PER_Julien_Vitay.pdf` |
| Experience Replay | ICML 2020 | `slides/Experience_Replay_ICML2020.pdf` |

### Policy Gradient Methods
| Algorithm | Source | File |
|-----------|--------|------|
| PPO | Waterloo CS885 | `slides/PPO_Waterloo_CS885.pdf` |
| PPO | NTU Taiwan | `slides/PPO_NTU_Taiwan.pdf` |
| TRPO | FAU Mutschler | `slides/TRPO_FAU_Mutschler.pdf` |
| TRPO | UT Austin | `slides/TRPO_UT_Austin.pdf` |
| TRPO/PPO | CMU Natural PG | `slides/TRPO_CMU_Natural_PG.pdf` |
| TRPO | Toronto PAIR | `slides/TRPO_Toronto_PAIR.pdf` |
| Policy Gradient | Toronto | `slides/Policy_Gradient_Toronto.pdf` |
| Policy Gradient | Berkeley CS285 | `slides/Policy_Gradient_Berkeley_CS285.pdf` |
| REINFORCE | Stanford CS229 | `slides/REINFORCE_Stanford_CS229.pdf` |
| GAE | Berkeley CS285 | `slides/GAE_Berkeley_CS285.pdf` |

### Actor-Critic Methods
| Algorithm | Source | File |
|-----------|--------|------|
| A3C | WPI DS595 | `slides/A3C_WPI_DS595.pdf` |
| A3C/Actor-Critic | U. Buffalo | `slides/A3C_Buffalo_Actor_Critic.pdf` |
| A3C | NTU Taiwan | `slides/A3C_NTU_Taiwan.pdf` |
| A3C | UIUC ECE448 | `slides/A3C_UIUC_ECE448.pdf` |
| A3C | Julien Vitay | `slides/A3C_Julien_Vitay.pdf` |
| DDPG | Paderborn DPG | `slides/DDPG_Paderborn_DPG.pdf` |
| DDPG | FAU Mutschler | `slides/DDPG_FAU_Mutschler.pdf` |
| DDPG | Julien Vitay | `slides/DDPG_Julien_Vitay.pdf` |
| DDPG/DPG | U. Buffalo | `slides/DDPG_Buffalo_DPG.pdf` |
| TD3 + SAC | Prague NPFL139 | `slides/TD3_SAC_Prague_NPFL139.pdf` |
| SAC | Toronto PAIR | `slides/SAC_Toronto_PAIR.pdf` |
| SAC/RL as Inference | Purdue | `slides/SAC_Purdue_RL_Inference.pdf` |
| SAC | Stanford CS231n | `slides/SAC_Stanford_CS231n.pdf` |

### Temporal Difference Learning
| Algorithm | Source | File |
|-----------|--------|------|
| TD Learning | CMU 10-403 | `slides/TD_Learning_CMU.pdf` |
| TD Methods | Michigan | `slides/TD_Methods_Michigan.pdf` |
| TD Learning | Sutton & Barto | `slides/TD_Sutton_Barto.pdf` |
| Q-Learning | Northeastern | `slides/Q_Learning_Northeastern.pdf` |

### Model-Based RL
| Topic | Source | File |
|-------|--------|------|
| Model-Based RL | FAU 2023 | `slides/Model_Based_RL_FAU.pdf` |
| Model-Based RL | Toronto | `slides/Model_Based_RL_Toronto.pdf` |
| Model-Based RL | Berkeley CS287 | `slides/Model_Based_RL_Berkeley.pdf` |
| Model-Based RL | CMU 10-403 | `slides/Model_Based_RL_CMU.pdf` |

### Imitation & Inverse RL
| Topic | Source | File |
|-------|--------|------|
| Imitation Learning | WPI DS595 | `slides/Imitation_Learning_WPI.pdf` |
| Imitation Learning | EPFL | `slides/Imitation_Learning_EPFL.pdf` |
| Inverse RL | TU Darmstadt | `slides/Inverse_RL_TU_Darmstadt.pdf` |
| Inverse RL | Berkeley CS285 | `slides/Inverse_RL_Berkeley_CS285.pdf` |

### Introductory Lectures
| Topic | Source | File |
|-------|--------|------|
| Deep RL Intro | Berkeley CS294 | `slides/Berkeley_CS294_Intro.pdf` |
| Deep RL Intro (2017) | Berkeley CS294 | `slides/Berkeley_CS294_Intro_2017.pdf` |

### Agentic RL & LLM Agents
| Topic | Source | File |
|-------|--------|------|
| RL for Agents | Stanford CS329T 2025 | `slides/Stanford_RL_for_Agents_2025.pdf` |
| RL for LLM Reasoning | Stanford | `slides/Stanford_RL_for_LLM_Reasoning.pdf` |
| LLM Agents Tool Learning | Tutorial | `slides/LLM_Agents_Tool_Learning_Tutorial.pdf` |
| Language Models as Agents | CMU | `slides/CMU_Language_Models_as_Agents.pdf` |
| LLM Agents & Tool Use | Mannheim | `slides/Mannheim_LLM_Agents_Tool_Use.pdf` |
| Multi-Agent RL Intro | Edinburgh | `slides/Edinburgh_Multi_Agent_RL_Intro.pdf` |
| DigiRL | NeurIPS 2024 | `slides/DigiRL_NeurIPS_2024.pdf` |
| MasRouter | ACL 2025 | `slides/MasRouter_ACL_2025.pdf` |
| ARTIST Reasoning | Microsoft Research | `slides/ARTIST_Agentic_Reasoning_Microsoft.pdf` |
| OS-Harm Benchmark | Safety | `slides/OS_Harm_Benchmark.pdf` |
| PTA-GRPO Planning | Planning | `slides/PTA_GRPO_High_Level_Planning.pdf` |

### RLHF & Preference Learning
| Topic | Source | File |
|-------|--------|------|
| DPO | CMU Lecture | `slides/DPO_CMU_Lecture.pdf` |
| DPO | Toronto | `slides/DPO_Toronto_Presentation.pdf` |
| DPO | UT Austin | `slides/DPO_UT_Austin.pdf` |
| DPO | Jinen Slides | `slides/DPO_Jinen_Slides.pdf` |
| RLHF | UT Austin | `slides/RLHF_UT_Austin_Slides.pdf` |

### Chain-of-Thought & Reasoning
| Topic | Source | File |
|-------|--------|------|
| DeepSeek-R1 | CMU Reasoning | `slides/DeepSeek_R1_CMU_Reasoning.pdf` |
| DeepSeek-R1 | Introduction | `slides/DeepSeek_R1_Introduction.pdf` |
| DeepSeek-R1 | Seoul National | `slides/DeepSeek_R1_Seoul_National.pdf` |
| DeepSeek-R1 | Toronto | `slides/DeepSeek_R1_Toronto.pdf` |
| CoT | Princeton | `slides/CoT_Princeton_Lecture.pdf` |
| CoT | SJTU | `slides/CoT_SJTU_Slides.pdf` |
| CoT | Toronto | `slides/CoT_Toronto_Presentation.pdf` |
| CoT Interpretable ML | Lecture | `slides/CoT_Interpretable_ML_Lecture.pdf` |
| Concise CoT | Benefits | `slides/Concise_CoT_Benefits.pdf` |
| ReAct | UVA Lecture | `slides/ReAct_UVA_Lecture.pdf` |

### Benchmarks & Evaluation
| Topic | Source | File |
|-------|--------|------|
| AgentBench | NeurIPS | `slides/AgentBench_Multi_Turn_NeurIPS.pdf` |
| Data Science Agents | Benchmark | `slides/Data_Science_Agents_Benchmark.pdf` |
| Web Agent Evaluation | Refinement | `slides/Web_Agent_Evaluation_Refinement.pdf` |

### Additional Topics
| Topic | Source | File |
|-------|--------|------|
| Software Agents | CMU Neubig | `slides/Software_Agents_Neubig.pdf` |
| Multimodal Agents | Berkeley | `slides/Multimodal_Agents_Berkeley.pdf` |
| LLM Training Agents | Berkeley | `slides/Berkeley_LLM_Training_Agents.pdf` |
| Voyager | UT Austin | `slides/Voyager_UT_Austin_Presentation.pdf` |
| LeanDojo | AITP | `slides/LeanDojo_AITP_Slides.pdf` |
| LeanDojo | NeurIPS | `slides/LeanDojo_NeurIPS_Slides.pdf` |
| Theorem Proving ML | Slides | `slides/Theorem_Proving_ML_Slides.pdf` |
| HippoRAG | NeurIPS | `slides/HippoRAG_NeurIPS_Slides.pdf` |
| Eureka Reward Design | Paper | `slides/Eureka_Reward_Design_Paper.pdf` |
| Code Agents Vulnerability | Berkeley | `slides/Code_Agents_Vulnerability_Detection_Berkeley.pdf` |
| Prompt Injection | Duke | `slides/Prompt_Injection_Duke_Slides.pdf` |
| EMNLP 2024 Language Agents | Tutorial | `slides/EMNLP2024_Language_Agents_Tutorial.pdf` |
| WWW 2024 LLM Agents | Tutorial | `slides/WWW2024_LLM_Agents_Tutorial.pdf` |
| AI Agents Architecture | Intel | `slides/Intel_AI_Agents_Architecture.pdf` |
| Agentic Frameworks | Cisco | `slides/Cisco_Agentic_Frameworks_Overview.pdf` |

---

## Courses & Tutorials

| Course | Platform | Description |
|--------|----------|-------------|
| [Deep RL Course](https://huggingface.co/learn/deep-rl-course/unit0/introduction) | Hugging Face | Free, comprehensive Deep RL course |
| [Deep RL Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) | Udacity | Official nanodegree program |
| [CS294-112 Deep RL](https://rail.eecs.berkeley.edu/deeprlcourse/) | UC Berkeley | Sergey Levine's course |
| [Spinning Up in Deep RL](https://spinningup.openai.com/) | OpenAI | Educational resource with code |
| [Thomas Simonini's Course](https://simoninithomas.github.io/Deep_reinforcement_learning_Course/) | Free | Step-by-step Deep RL tutorials |
| [RL in Finance Specialization](https://www.coursera.org/specializations/machine-learning-reinforcement-finance) | Coursera | RL for finance applications |
| [TensorFlow for Deep Learning](https://www.udacity.com/course/intro-to-tensorflow-for-deep-learning--ud187) | Udacity | Free course from Google |
| [Intro to Deep Learning with PyTorch](https://www.udacity.com/course/deep-learning-pytorch--ud188) | Udacity | PyTorch fundamentals |

### Free Tutorials
| Tutorial | Link |
|----------|------|
| Introduction to RL | [FreeCodeCamp](https://medium.freecodecamp.org/an-introduction-to-reinforcement-learning-4339519de419) |
| Diving Deeper with Q-Learning | [FreeCodeCamp](https://medium.freecodecamp.org/diving-deeper-into-reinforcement-learning-with-q-learning-c18d0db58efe) |
| Curiosity-Driven Learning | [Towards Data Science](https://towardsdatascience.com/curiosity-driven-learning-made-easy-part-i-d3e5a2263359) |

---

## Books

| Book | Author | Link |
|------|--------|------|
| Reinforcement Learning: An Introduction (2nd ed) | Sutton & Barto | [PDF](http://incompleteideas.net/book/RLbook2020.pdf) |
| Grokking Deep Reinforcement Learning | Miguel Morales | [Manning](https://www.manning.com/books/grokking-deep-reinforcement-learning), [GitHub](https://github.com/mimoralea/gdrl) |
| Deep Reinforcement Learning Hands-On | Maxim Lapan | [Packt](https://www.packtpub.com/product/deep-reinforcement-learning-hands-on/9781788834247) |
| Multi-Agent Machine Learning: A Reinforcement Approach | H. Schwartz | Academic Press |

---

## Key Papers

### Foundational Papers

| Paper | Year | Link | Notes |
|-------|------|------|-------|
| Playing Atari with Deep RL | 2013 | [arXiv](https://arxiv.org/abs/1312.5602) | Original DQN |
| Human-level Control through Deep RL | 2015 | [Nature](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) | DQN breakthrough |
| Deep RL with Double Q-learning | 2015 | [arXiv](https://arxiv.org/abs/1509.06461) | Double DQN |
| Dueling Network Architectures | 2015 | [arXiv](https://arxiv.org/abs/1511.06581) | Dueling DQN |
| Prioritized Experience Replay | 2015 | [arXiv](https://arxiv.org/abs/1511.05952) | PER |
| Asynchronous Methods for Deep RL | 2016 | [arXiv](https://arxiv.org/abs/1602.01783) | A3C |
| Continuous Control with Deep RL | 2015 | [arXiv](https://arxiv.org/abs/1509.02971) | DDPG |

### Policy Gradient Methods

| Paper | Year | Link | Notes |
|-------|------|------|-------|
| Proximal Policy Optimization | 2017 | [arXiv](https://arxiv.org/abs/1707.06347) | PPO |
| High-dimensional Continuous Control Using GAE | 2015 | [arXiv](https://arxiv.org/abs/1506.02438) | GAE |
| Emergence of Locomotion Behaviours | 2017 | [arXiv](https://arxiv.org/abs/1707.02286) | A2C/A3C for locomotion |
| D4PG | 2018 | [OpenReview](https://openreview.net/pdf?id=SyZipzbCb) | Distributed DDPG |

### Advanced Topics

| Paper | Year | Link | Notes |
|-------|------|------|-------|
| Rainbow: Combining Improvements | 2017 | [arXiv](https://arxiv.org/abs/1710.02298) | Combined DQN improvements |
| Noisy Networks for Exploration | 2017 | [arXiv](https://arxiv.org/abs/1706.10295) | NoisyNet |
| Deep Recurrent Q-Learning | 2015 | [arXiv](https://arxiv.org/abs/1507.06527) | DRQN |
| Distributional Perspective on RL | 2017 | [arXiv](https://arxiv.org/abs/1707.06887) | C51 |
| Meta-Gradient RL | 2018 | [arXiv](https://arxiv.org/abs/1805.09801) | Meta-learning |
| RUDDER: Return Decomposition | 2018 | [arXiv](https://arxiv.org/abs/1806.07857) | Credit assignment |

### Surveys & Overviews

| Paper | Pages | Link |
|-------|-------|------|
| Deep Reinforcement Learning (Yuxi Li) | 150 | [arXiv](https://arxiv.org/abs/1810.06339) |
| Deep RL: An Overview | 70 | [arXiv](https://arxiv.org/abs/1701.07274) |
| A Brief Survey of Deep RL | 16 | [arXiv](https://arxiv.org/abs/1708.05866) |
| OpenAI: Key Papers in Deep RL | - | [Spinning Up](https://spinningup.openai.com/en/latest/spinningup/keypapers.html) |

### Multi-Agent RL

| Paper | Link |
|-------|------|
| Multi-Agent RL: Challenges and Approaches | [arXiv](https://arxiv.org/abs/1807.09427) |
| Is MARL the Answer or the Question? | [arXiv](https://arxiv.org/abs/1810.05587) |
| MARL Papers Collection | [GitHub](https://github.com/LantaoYu/MARL-Papers) |

### Applications

| Paper | Application | Link |
|-------|-------------|------|
| Deep RL in Portfolio Management | Finance | [arXiv](https://arxiv.org/abs/1808.09940) |
| Market Making via RL | Trading | [arXiv](https://arxiv.org/abs/1804.04216) |
| Autonomous Driving with RL | Self-driving | [arXiv](https://arxiv.org/abs/1801.05299) |

---

## GitHub Repositories

### Official Course Repos

| Repository | Description |
|------------|-------------|
| [udacity/deep-reinforcement-learning](https://github.com/udacity/deep-reinforcement-learning) | Udacity Deep RL Nanodegree |
| [udacity/Value-based-methods](https://github.com/udacity/Value-based-methods) | Value-based RL methods |
| [udacity/DL_PyTorch](https://github.com/udacity/DL_PyTorch) | Deep Learning with PyTorch |

### RL Frameworks

| Repository | Description |
|------------|-------------|
| [openai/baselines](https://github.com/openai/baselines) | OpenAI high-quality RL implementations |
| [openai/spinningup](https://github.com/openai/spinningup) | Educational RL implementations |
| [google/dopamine](https://github.com/google/dopamine) | Google's RL research framework |
| [deepmind/trfl](https://github.com/deepmind/trfl) | TensorFlow RL library |
| [reinforceio/tensorforce](https://github.com/reinforceio/tensorforce) | TensorFlow RL library |
| [rll/rllab](https://github.com/rll/rllab) | Berkeley RL lab framework |

### Algorithm Implementations

| Repository | Description |
|------------|-------------|
| [ShangtongZhang/DeepRL](https://github.com/ShangtongZhang/DeepRL) | Modular PyTorch RL implementations |
| [higgsfield/RL-Adventure-2](https://github.com/higgsfield/RL-Adventure-2) | Policy Gradient implementations |
| [suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general) | AlphaZero for any game |
| [openai/evolution-strategies-starter](https://github.com/openai/evolution-strategies-starter) | Distributed evolution strategies |
| [openai/large-scale-curiosity](https://github.com/openai/large-scale-curiosity) | Curiosity-driven learning |

### Environments & Simulators

| Repository | Description |
|------------|-------------|
| [openai/gym](https://github.com/openai/gym) | OpenAI Gym environments |
| [Unity-Technologies/ml-agents](https://github.com/Unity-Technologies/ml-agents) | Unity ML-Agents Toolkit |
| [Unity-Technologies/marathon-envs](https://github.com/Unity-Technologies/marathon-envs) | Unity Marathon environments |
| [StanfordVL/GibsonEnv](https://github.com/StanfordVL/GibsonEnv) | Gibson real-world perception |
| [dusty-nv/jetson-reinforcement](https://github.com/dusty-nv/jetson-reinforcement) | Deep RL for NVIDIA Jetson |

### Projects & Examples

| Repository | Description |
|------------|-------------|
| [worldmodels.github.io](https://worldmodels.github.io/) | World Models implementation |
| [kvfrans/openai-cartpole](https://github.com/kvfrans/openai-cartpole) | CartPole solutions |
| [alirezamika/bipedal-es](https://github.com/alirezamika/bipedal-es) | BipedalWalker with ES |
| [xkiwilabs/DQN_Unity_Keras](https://github.com/xkiwilabs/DQN_Unity_Keras) | DQN for Unity |

---

## OpenAI Resources

| Resource | Link |
|----------|------|
| OpenAI Homepage | [openai.com](https://openai.com) |
| OpenAI Blog | [blog.openai.com](https://blog.openai.com/) |
| Gym Repository | [github.com/openai/gym](https://github.com/openai/gym) |
| Gym Documentation | [gym.openai.com](https://gym.openai.com) |
| Gym Leaderboard | [Leaderboard](https://github.com/openai/gym/wiki/Leaderboard) |
| Baselines | [github.com/openai/baselines](https://github.com/openai/baselines) |
| Spinning Up | [spinningup.openai.com](https://spinningup.openai.com/) |

---

## PyTorch & Frameworks

| Resource | Link |
|----------|------|
| PyTorch | [pytorch.org](https://pytorch.org) |
| PyTorch Tutorials | [pytorch.org/tutorials](https://pytorch.org/tutorials/) |
| PyTorch Neural Networks | [NN Documentation](https://pytorch.org/docs/stable/nn.html) |
| PyTorch GitHub | [github.com/pytorch/pytorch](https://github.com/pytorch/pytorch) |
| PlaidML (any device) | [github.com/plaidml/plaidml](https://github.com/plaidml/plaidml) |

---

## Unity ML-Agents

| Resource | Link |
|----------|------|
| ML-Agents Repository | [GitHub](https://github.com/Unity-Technologies/ml-agents) |
| Getting Started | [Balance Ball Tutorial](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Getting-Started-with-Balance-Ball.md) |
| Unity ML Blog | [Introduction](https://blogs.unity3d.com/2017/09/19/introducing-unity-machine-learning-agents/) |
| Marathon Environments | [GitHub](https://github.com/Unity-Technologies/marathon-envs) |
| Adventures in Unity ML-Agents | [Blog](http://adventuresinunitymlagents.com/) |

---

## Blogs

### Research Labs
| Blog | Organization |
|------|--------------|
| [DeepMind Blog](https://deepmind.com/blog/) | DeepMind |
| [OpenAI Blog](https://blog.openai.com/) | OpenAI |
| [Google AI Blog](https://ai.googleblog.com/) | Google |
| [UC Berkeley AI Research](http://bair.berkeley.edu/blog/) | UC Berkeley |
| [The Gradient](https://thegradient.pub/) | Stanford AI Lab |

### Individual Researchers
| Blog | Author |
|------|--------|
| [Andrej Karpathy (older)](http://karpathy.github.io/) | Andrej Karpathy |
| [Andrej Karpathy (newer)](https://medium.com/@karpathy/) | Andrej Karpathy |
| [Richard S. Sutton](http://incompleteideas.net/) | Richard Sutton |
| [Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/) | Andrej Karpathy |

### Community
| Blog | Focus |
|------|-------|
| [Towards Data Science](https://towardsdatascience.com/) | Data Science & ML |
| [The Morning Paper](https://blog.acolyer.org/) | Paper summaries |
| [Distill](https://distill.pub/) | Clear ML research |
| [Synced Review](https://syncedreview.com/) | AI Industry news |

### Key Articles
| Article | Link |
|---------|------|
| RL Doesn't Work Yet | [Alex Irpan](https://www.alexirpan.com/2018/02/14/rl-hard.html) |
| Math behind TRPO & PPO | [Jonathan Hui](https://medium.com/@jonathan_hui/rl-the-math-behind-trpo-ppo-d12f6c745f33) |
| Intro to A2C | [Hackernoon](https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752) |
| Beyond DQN/A3C Survey | [Towards Data Science](https://towardsdatascience.com/advanced-reinforcement-learning-6d769f529eb3) |

---

## Videos & Lectures

### Full Courses
| Course | Instructor | Link |
|--------|------------|------|
| RL Course (10 videos) | David Silver | [YouTube Playlist](https://www.youtube.com/playlist?list=PL7-jPKtc4r78-wCZcQn5IqyuWhBZ8fOxT) |
| CS294-112 Deep RL (24 lectures) | Sergey Levine | [YouTube Playlist](https://www.youtube.com/playlist?list=PLkFD6_40KJIznC9CDbVTjAF2oyt8_VAe3) |
| Deep RL Bootcamp 2017 | Pieter Abbeel | [Bootcamp Site](https://sites.google.com/view/deep-rl-bootcamp/lectures) |

### Key Lectures
| Lecture | Link |
|---------|------|
| Deep RL Bootcamp: Nuts and Bolts | [YouTube](https://www.youtube.com/watch?v=8EcdaCk9KaQ) |
| Deep Learning: The Good, Bad, Ugly | [YouTube Playlist](https://www.youtube.com/playlist?list=PLVjVSqmQgPG-yy8vUHQXnQ7-qhsbKjd9s) |

---

## Cloud Platforms

| Platform | Description | Link |
|----------|-------------|------|
| Google Colab | Free GPU notebooks | [colab.research.google.com](https://colab.research.google.com) |
| FloydHub | ML platform | [floydhub.com](https://www.floydhub.com/) |
| Vast.ai | Low cost GPU instances | [vast.ai](https://www.vast.ai) |
| Google AutoML | AutoML | [cloud.google.com/automl](https://cloud.google.com/automl/) |
| VectorDash | GPU cloud | [vectordash.com](https://vectordash.com/) |

---

## Competitions

| Competition | Link |
|-------------|------|
| Pommerman | [pommerman.com](https://www.pommerman.com/) |
| Halite | [halite.io](https://halite.io/) |
| OpenAI Retro Contest | [contest.openai.com](https://contest.openai.com/2018-1/) |
| NeurIPS Competition Track | [neurips.cc](https://nips.cc/Conferences/2018/CompetitionTrack) |

---

## Cheatsheets & Glossaries

### Cheatsheets
| Resource | Link |
|----------|------|
| AI/ML/DL Cheat Sheets | [Becoming Human](https://becominghuman.ai/cheat-sheets-for-ai-neural-networks-machine-learning-deep-learning-big-data-678c51b4b463) |
| C++ Python Cheatsheet | [Udacity PDF](https://d17h27t6h515a5.cloudfront.net/topher/2018/January/5a4d862b_c-python-cheatsheet/c-python-cheatsheet.pdf) |

### Glossaries
| Resource | Link |
|----------|------|
| NVIDIA Deep Learning Glossary | [PDF](https://www.nvidia.com/content/g/pdfs/nvidia-deeplearning-glossary-llkcmb.pdf) |
| Google ML Glossary | [developers.google.com](https://developers.google.com/machine-learning/glossary/) |

---

## Articles

### Critical Perspectives
| Article | Author | Link |
|---------|--------|------|
| RL Doesn't Work Yet | Alex Irpan | [Blog](https://www.alexirpan.com/2018/02/14/rl-hard.html) |
| Why RL is Flawed | The Gradient | [Article](https://thegradient.pub/why-rl-is-flawed/) |
| How to Fix RL | The Gradient | [Article](https://thegradient.pub/how-to-fix-rl/) |
| Lessons Learned Reproducing Deep RL | Matthew Rahtz | [Blog](http://amid.fish/reproducing-deep-rl) |

### Evolution Strategies
| Article | Link |
|---------|------|
| ES as Scalable Alternative to RL | [The Morning Paper](https://blog.acolyer.org/) |
| ES Outperforms DL at Video Games | [MIT Tech Review](https://www.technologyreview.com/s/611568/evolutionary-algorithm-outperforms-deep-learning-machines-at-video-games/) |

### Practical Guides
| Article | Link |
|---------|------|
| Math for Deep Learning | [Blog](http://leiluoray.com/2018/08/29/Deep-Learning-Math/) |
| Which GPU for Deep Learning | [Tim Dettmers](http://timdettmers.com/2018/08/21/which-gpu-for-deep-learning/) |
| How to Read Scientific Papers | [HuffPost](https://www.huffingtonpost.com/jennifer-raff/how-to-read-and-understand-a-scientific-paper_b_5501628.html) |

---

## Hardware

| Resource | Link |
|----------|------|
| Which GPU(s) for Deep Learning | [Tim Dettmers](http://timdettmers.com/2018/08/21/which-gpu-for-deep-learning/) |

---

## Quick Reference Links

| Short URL | Description |
|-----------|-------------|
| [bit.ly/drlndlinks](http://bit.ly/drlndlinks) | Student-curated resources |
| [bit.ly/gdrl_u](http://bit.ly/gdrl_u) | Grokking DRL book discount |

---

## Contributing

This is a community-maintained resource. Feel free to add new links and resources!

---

*Last updated: January 2025*
