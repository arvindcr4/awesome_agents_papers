# RL Agent Universe: Research Papers by Direction

> A curated collection of seminal papers organized by the 8 major research directions in Reinforcement Learning.
>
> See [rl-research-directions.md](rl-research-directions.md) for practical code examples and conceptual guides.

---

## Research Directions Map

| # | Direction | Analogy | Focus |
|---|-----------|---------|-------|
| 1 | [Applied RL](#1-applied-rl-the-world-builder) | The World Builder | Custom environments, reward design |
| 2 | [Algorithm Research](#2-algorithm-research-the-architect) | The Architect | Neural architectures, learning improvements |
| 3 | [Offline RL](#3-offline-rl-the-historian) | The Historian | Learning from static datasets |
| 4 | [Multi-Agent RL](#4-multi-agent-rl-the-coordinator) | The Coordinator | Cooperation, competition, emergence |
| 5 | [RLHF & Alignment](#5-rlhf--alignment-the-aligner) | The Aligner | Human preferences, AI safety |
| 6 | [Generalization & ProcGen](#6-generalization--procgen-the-dungeon-master) | The Dungeon Master | Zero-shot transfer, robustness |
| 7 | [Agentic RL & Tool Use](#7-agentic-rl--tool-use-the-operator) | The Operator | Web agents, LLM tools |
| 8 | [Systems & Scale](#8-systems--scale-the-mechanic) | The Mechanic | Distributed training, infrastructure |

---

## 1. Applied RL (The World Builder)

> **Focus:** Creating environments that accurately simulate real-world problems.

### Simulation Platforms & Environments

| Paper | Authors | Year | Contribution |
|-------|---------|------|--------------|
| **OpenAI Gym** | Brockman et al. | 2016 | Standard API for RL environments |
| **MuJoCo: Physics for Model-Based Control** | Todorov et al. | 2012 | Fast, accurate physics simulation |
| **DeepMind Control Suite** | Tassa et al. | 2018 | Standardized continuous control benchmarks |
| **Isaac Gym: High Performance GPU Simulation** | Makoviychuk et al. | 2021 | Massively parallel robot simulation |
| **Habitat: Embodied AI Platform** | Savva et al. | 2019 | Photorealistic indoor navigation |
| **AI2-THOR** | Kolve et al. | 2017 | Interactive household environments |

### Domain-Specific Applications

| Paper | Authors | Year | Domain |
|-------|---------|------|--------|
| **Deep RL for HVAC Control** | Wei et al. | 2017 | Building energy (20-40% savings) |
| **Data Center Cooling with RL** | DeepMind | 2016 | Infrastructure (40% energy reduction) |
| **Gas Turbine Auto-Tuner** | Siemens | 2017 | Industrial control |
| **RL for Inventory Management** | Various | 2018+ | Supply chain optimization |

### Key Concepts
- **State/Observation Design** - Defining what the agent perceives
- **Reward Engineering** - Balancing sparse vs dense rewards
- **Domain Knowledge Encoding** - Physics constraints, business rules

---

## 2. Algorithm Research (The Architect)

> **Focus:** Building better neural network architectures and learning algorithms.

### Foundational Algorithms

| Paper | Authors | Year | Contribution |
|-------|---------|------|--------------|
| **DQN: Human-Level Control** | Mnih et al. | 2015 | Deep Q-learning + experience replay |
| **A3C: Asynchronous Methods** | Mnih et al. | 2016 | Parallel actor-critic training |
| **TRPO: Trust Region Policy Optimization** | Schulman et al. | 2015 | Stable policy updates |
| **PPO: Proximal Policy Optimization** | Schulman et al. | 2017 | Simple, effective policy gradient |
| **SAC: Soft Actor-Critic** | Haarnoja et al. | 2018 | Maximum entropy framework |
| **TD3: Twin Delayed DDPG** | Fujimoto et al. | 2018 | Addressing overestimation |

### Value-Based Improvements

| Paper | Authors | Year | Contribution |
|-------|---------|------|--------------|
| **Double DQN** | van Hasselt et al. | 2016 | Reduced overestimation bias |
| **Dueling DQN** | Wang et al. | 2016 | Separate value/advantage streams |
| **Prioritized Experience Replay** | Schaul et al. | 2015 | Important transition sampling |
| **Rainbow** | Hessel et al. | 2018 | Combined DQN improvements |
| **C51: Distributional RL** | Bellemare et al. | 2017 | Learning return distributions |

### Model-Based RL

| Paper | Authors | Year | Contribution |
|-------|---------|------|--------------|
| **World Models** | Ha & Schmidhuber | 2018 | Learning in latent imagination |
| **Dreamer** | Hafner et al. | 2019 | World model + actor-critic |
| **DreamerV2** | Hafner et al. | 2020 | Discrete latents, improved performance |
| **DreamerV3** | Hafner et al. | 2023 | General algorithm across domains |
| **MuZero** | Schrittwieser et al. | 2020 | Learned model without environment rules |

### Key Concepts
- **Feature Extraction** - CNNs, Transformers, Graph Networks
- **Memory Mechanisms** - LSTMs, Transformers, External Memory
- **Exploration Strategies** - Intrinsic motivation, curiosity

---

## 3. Offline RL (The Historian)

> **Focus:** Learning optimal policies from static datasets without environment interaction.

### Core Algorithms

| Paper | Authors | Year | Contribution |
|-------|---------|------|--------------|
| **BCQ: Batch-Constrained Q-Learning** | Fujimoto et al. | 2019 | First principled offline RL |
| **CQL: Conservative Q-Learning** | Kumar et al. | 2020 | Pessimistic value estimation |
| **IQL: Implicit Q-Learning** | Kostrikov et al. | 2021 | Avoids OOD actions entirely |
| **Decision Transformer** | Chen et al. | 2021 | RL as sequence modeling |
| **Trajectory Transformer** | Janner et al. | 2021 | Planning via beam search |

### Imitation Learning

| Paper | Authors | Year | Contribution |
|-------|---------|------|--------------|
| **DAGGER** | Ross et al. | 2011 | Dataset aggregation |
| **GAIL: Generative Adversarial IL** | Ho & Ermon | 2016 | Adversarial imitation |
| **BCO: Behavioral Cloning from Observation** | Torabi et al. | 2018 | Learning from state-only demos |

### Key Concepts
- **Distribution Shift** - Training data vs deployment mismatch
- **Conservative Estimation** - Avoiding overconfident Q-values
- **Behavior Regularization** - Staying close to data distribution

---

## 4. Multi-Agent RL (The Coordinator)

> **Focus:** Multiple agents learning to cooperate or compete.

### Foundational Methods

| Paper | Authors | Year | Contribution |
|-------|---------|------|--------------|
| **MADDPG: Multi-Agent DDPG** | Lowe et al. | 2017 | Centralized training, decentralized execution |
| **QMIX** | Rashid et al. | 2018 | Value decomposition for cooperation |
| **COMA: Counterfactual Multi-Agent** | Foerster et al. | 2018 | Credit assignment via counterfactuals |
| **MAPPO** | Yu et al. | 2021 | PPO for multi-agent settings |

### Emergent Behavior

| Paper | Authors | Year | Contribution |
|-------|---------|------|--------------|
| **Emergent Tool Use (Hide & Seek)** | Baker et al. | 2019 | Complex behaviors from self-play |
| **Emergent Communication** | Lazaridou et al. | 2016 | Learned language protocols |
| **Neural MMO** | Suarez et al. | 2019 | Massively multi-agent environments |

### Environments

| Environment | Paper | Year | Focus |
|-------------|-------|------|-------|
| **PettingZoo** | Terry et al. | 2021 | Multi-agent Gym API |
| **SMAC (StarCraft)** | Samvelyan et al. | 2019 | Cooperative micromanagement |
| **Hanabi** | Bard et al. | 2020 | Theory of mind, coordination |
| **Melting Pot** | Leibo et al. | 2021 | Social dilemmas at scale |

### Key Concepts
- **CTDE** - Centralized Training, Decentralized Execution
- **Credit Assignment** - Attributing team reward to individuals
- **Non-Stationarity** - Other agents change the environment

---

## 5. RLHF & Alignment (The Aligner)

> **Focus:** Learning from human preferences and ensuring AI safety.

### Core Methods

| Paper | Authors | Year | Contribution |
|-------|---------|------|--------------|
| **RLHF: InstructGPT** | Ouyang et al. | 2022 | PPO from human preferences |
| **Constitutional AI** | Bai et al. | 2022 | RL from AI feedback (RLAIF) |
| **DPO: Direct Preference Optimization** | Rafailov et al. | 2023 | Simplified alternative to RLHF |
| **RLAIF** | Lee et al. | 2023 | AI-generated preference labels |

### Reward Modeling

| Paper | Authors | Year | Contribution |
|-------|---------|------|--------------|
| **Deep RL from Human Preferences** | Christiano et al. | 2017 | Learning rewards from comparisons |
| **WebGPT** | Nakano et al. | 2022 | Web browsing with human feedback |
| **Reward Hacking** | Skalse et al. | 2022 | Failures of learned rewards |

### Safety & Robustness

| Paper | Authors | Year | Contribution |
|-------|---------|------|--------------|
| **Safe RL Survey** | Garcia & Fernandez | 2015 | Comprehensive safety methods |
| **Constrained Policy Optimization** | Achiam et al. | 2017 | RL with hard constraints |
| **Shielded RL** | Alshiekh et al. | 2018 | Formal safety logic |

### Key Concepts
- **Reward Modeling** - Learning preferences from comparisons
- **KL Divergence Constraints** - Don't drift from base model
- **Red Teaming** - Adversarial evaluation

---

## 6. Generalization & ProcGen (The Dungeon Master)

> **Focus:** Training agents that generalize to unseen environments.

### Procedural Generation Benchmarks

| Paper | Authors | Year | Contribution |
|-------|---------|------|--------------|
| **Procgen Benchmark** | Cobbe et al. | 2020 | 16 procedurally-generated games |
| **CoinRun: Quantifying Generalization** | Cobbe et al. | 2019 | Showed need for training diversity |
| **Obstacle Tower** | Juliani et al. | 2019 | 3D procedural generalization test |
| **XLand** | DeepMind | 2021 | Universe of procedural 3D games |

### Domain Randomization

| Paper | Authors | Year | Contribution |
|-------|---------|------|--------------|
| **Domain Randomization for Sim2Real** | Tobin et al. | 2017 | Visual randomization for transfer |
| **CAD2RL** | Sadeghi & Levine | 2017 | Drone flight from synthetic data |
| **Solving Rubik's Cube (ADR)** | OpenAI | 2019 | Automatic domain randomization |

### Curriculum Learning & UED

| Paper | Authors | Year | Contribution |
|-------|---------|------|--------------|
| **ALP-GMM** | Portelas et al. | 2019 | Learning progress-based generation |
| **POET** | Wang et al. | 2019 | Co-evolution of envs and agents |
| **PAIRED** | Dennis et al. | 2020 | Adversarial environment design |
| **PLR: Prioritized Level Replay** | Jiang et al. | 2021 | Replay buffer for levels |
| **ACCEL** | Xu et al. | 2022 | Level editing for complexity |

### Robust RL Environments

| Environment | Paper | Year | Focus |
|-------------|-------|------|-------|
| **MiniGrid** | Chevalier-Boisvert et al. | 2018 | Partial observability, memory |
| **BabyAI** | Chevalier-Boisvert et al. | 2019 | Language-conditioned navigation |
| **NetHack Learning Environment** | Kuttler et al. | 2020 | Procedural roguelike |
| **Crafter** | Hafner et al. | 2021 | Minecraft-like survival |

### Key Concepts
- **Zero-Shot Transfer** - Solving unseen instances
- **Overfitting to Levels** - Memorization vs generalization
- **Automatic Curriculum** - Environment difficulty progression

---

## 7. Agentic RL & Tool Use (The Operator)

> **Focus:** Agents that control software, use tools, and follow instructions.

### Web-Based Agents

| Paper | Authors | Year | Contribution |
|-------|---------|------|--------------|
| **World of Bits (MiniWoB)** | Shi et al. | 2017 | 100 synthetic web tasks |
| **MiniWoB++ with WGE** | Liu et al. | 2018 | Workflow-guided exploration |
| **DOM-Q-NET** | Jia et al. | 2019 | Graph neural networks for DOM |
| **WebArena** | Zhou et al. | 2024 | Realistic functional websites |
| **WorkArena** | Drouin et al. | 2024 | Enterprise software tasks |

### LLM-Based Agents

| Paper | Authors | Year | Contribution |
|-------|---------|------|--------------|
| **CC-Net** | Humphreys et al. | 2022 | Human-level MiniWoB++ via BC+RL |
| **WebGUM** | Furuta et al. | 2024 | 3B multimodal web agent |
| **Synapse** | Zheng et al. | 2023 | 99.2% via trajectory prompts |
| **RCI: Recursive Criticism** | Kim et al. | 2023 | Self-correcting LLM prompts |

### Tool Use & Reasoning

| Paper | Authors | Year | Contribution |
|-------|---------|------|--------------|
| **Toolformer** | Schick et al. | 2023 | Self-taught API calling |
| **ReAct** | Yao et al. | 2022 | Reasoning + Acting interleaved |
| **Chain-of-Thought** | Wei et al. | 2022 | Step-by-step reasoning |
| **Tree of Thoughts** | Yao et al. | 2023 | Search over reasoning paths |

### Key Concepts
- **Grounding** - Language to actions mapping
- **Hierarchical Actions** - High-level plans, low-level execution
- **Multimodal Observations** - Screenshots, DOM, text

---

## 8. Systems & Scale (The Mechanic)

> **Focus:** Infrastructure for training RL at massive scale.

### Distributed RL Frameworks

| Paper | Authors | Year | Contribution |
|-------|---------|------|--------------|
| **Ray** | Moritz et al. | 2018 | Distributed computing for AI |
| **RLlib** | Liang et al. | 2018 | Scalable, composable RL |
| **IMPALA** | Espeholt et al. | 2018 | Scalable actor-learner architecture |
| **Ape-X** | Horgan et al. | 2018 | Distributed prioritized replay |
| **R2D2** | Kapturowski et al. | 2018 | Recurrent experience replay |

### Modular Libraries

| Library | Paper/Team | Year | Contribution |
|---------|------------|------|--------------|
| **TorchRL** | Bou et al. | 2023 | Modular PyTorch RL with TensorDict |
| **Stable Baselines3** | Raffin et al. | 2021 | Clean PyTorch implementations |
| **CleanRL** | Huang et al. | 2022 | Single-file clarity |
| **TF-Agents** | Guadarrama et al. | 2018 | Production-ready TensorFlow RL |
| **RLax** | DeepMind | 2020 | Functional JAX primitives |
| **Brax** | Freeman et al. | 2021 | Differentiable physics in JAX |

### Cloud Platforms

| Platform | Organization | Year | Contribution |
|----------|--------------|------|--------------|
| **AWS SageMaker RL** | Amazon | 2018 | Managed RL with RLlib/Coach |
| **Azure Bonsai** | Microsoft | 2020 | Machine teaching + Inkling DSL |
| **Horizon** | Meta | 2018 | Production RL with offline eval |

### Systems Papers

| Paper | Authors | Year | Contribution |
|-------|---------|------|--------------|
| **Engineering RL Platforms** | Kanso et al. | 2022 | Bonsai architecture |
| **Sample Factory** | Petrenko et al. | 2020 | High-throughput on single machine |
| **EnvPool** | Weng et al. | 2022 | Vectorized C++ environments |

### Key Concepts
- **Actor-Learner Separation** - Parallel data collection
- **Vectorized Environments** - Batch simulation
- **TensorDict** - Efficient heterogeneous data handling

---

## Cross-Cutting: Autonomous Driving

> **Combines:** Applied RL, Offline RL, Sim-to-Real, Safety

### CARLA Ecosystem

| Paper | Authors | Year | Contribution |
|-------|---------|------|--------------|
| **CARLA Simulator** | Dosovitskiy et al. | 2017 | High-fidelity driving simulation |
| **CIRL** | Liang et al. | 2018 | First RL to beat BC in CARLA |
| **Learning by Cheating** | Chen et al. | 2019 | Privileged expert + IL student |
| **Roach** | Zhang et al. | 2021 | RL coach for IL training |
| **CaRL** | Kuhn et al. | 2023 | State-of-art RL planner |
| **Think2Drive** | Li et al. | 2024 | World model (DreamerV3) for driving |

### Benchmarks

| Benchmark | Year | Focus |
|-----------|------|-------|
| **CoRL 2017** | 2017 | Basic navigation tasks |
| **NoCrash** | 2019 | Dense traffic, collision-free |
| **CARLA Leaderboard** | 2020+ | Long routes, comprehensive scoring |

---

## Cross-Cutting: Robotics

> **Combines:** Applied RL, Generalization (Sim2Real), Systems

### Sim-to-Real Transfer

| Paper | Authors | Year | Contribution |
|-------|---------|------|--------------|
| **Domain Randomization** | Tobin et al. | 2017 | Visual randomization |
| **Learning Dexterous Manipulation** | OpenAI | 2018 | Shadow hand cube |
| **Solving Rubik's Cube** | OpenAI | 2019 | ADR for complex manipulation |
| **Learning to Walk in Minutes** | Rudin et al. | 2022 | Isaac Gym massive parallelism |

### Simulation Platforms

| Platform | Organization | Focus |
|----------|--------------|-------|
| **Isaac Gym** | NVIDIA | GPU-accelerated robot sim |
| **RoboSuite** | Stanford | Manipulation benchmarks |
| **Meta-World** | UC Berkeley | Multi-task manipulation |
| **dm_control** | DeepMind | Continuous control suite |

---

## Cross-Cutting: 3D Game Environments

> **Combines:** Generalization, Multi-Agent, Applied RL

### Unity ML-Agents

| Paper | Authors | Year | Contribution |
|-------|---------|------|--------------|
| **Unity: A General Platform** | Juliani et al. | 2018 | 3D game engine for RL |
| **Obstacle Tower** | Juliani et al. | 2019 | Procedural 3D generalization |
| **Animal-AI Testbed** | Crosby et al. | 2020 | Cognition-inspired tasks |

### Other 3D Environments

| Environment | Paper | Year | Focus |
|-------------|-------|------|-------|
| **DeepMind Lab** | Beattie et al. | 2016 | First-person navigation |
| **VizDoom** | Kempka et al. | 2016 | Doom-based visual RL |
| **Malmo (Minecraft)** | Johnson et al. | 2016 | Open-world exploration |

---

## Quick Reference: Algorithm Selection

| Research Direction | Recommended Algorithms |
|-------------------|----------------------|
| **Applied RL** | PPO, SAC (continuous), DQN (discrete) |
| **Algorithm Research** | Custom architectures on PPO/SAC base |
| **Offline RL** | CQL, IQL, Decision Transformer |
| **Multi-Agent RL** | MAPPO, QMIX |
| **RLHF** | PPO + reward model, DPO |
| **Generalization** | PPO + PLR/ACCEL curriculum |
| **Agentic RL** | BC + RL fine-tuning, LLM agents |
| **Systems RL** | RLlib, TorchRL, IMPALA patterns |

---

## Resources

### Official Repositories

| Library | Repository |
|---------|------------|
| Stable Baselines3 | [github.com/DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3) |
| CleanRL | [github.com/vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl) |
| RLlib (Ray) | [github.com/ray-project/ray](https://github.com/ray-project/ray) |
| TorchRL | [github.com/pytorch/rl](https://github.com/pytorch/rl) |
| d3rlpy | [github.com/takuseno/d3rlpy](https://github.com/takuseno/d3rlpy) |
| PettingZoo | [github.com/Farama-Foundation/PettingZoo](https://github.com/Farama-Foundation/PettingZoo) |
| Gymnasium | [github.com/Farama-Foundation/Gymnasium](https://github.com/Farama-Foundation/Gymnasium) |
| Procgen | [github.com/openai/procgen](https://github.com/openai/procgen) |
| CARLA | [github.com/carla-simulator/carla](https://github.com/carla-simulator/carla) |
| trl (Hugging Face) | [github.com/huggingface/trl](https://github.com/huggingface/trl) |

### Learning Resources

- **Spinning Up in Deep RL** - OpenAI's educational resource
- **Hugging Face Deep RL Course** - Interactive tutorials
- **CS285: Deep RL** - UC Berkeley (Sergey Levine)
- **David Silver's RL Course** - DeepMind/UCL

### Key Surveys

| Survey | Authors | Year | Focus |
|--------|---------|------|-------|
| **Deep RL Survey** | Li | 2017 | Comprehensive overview |
| **Deep RL That Matters** | Henderson et al. | 2018 | Reproducibility crisis |
| **Offline RL Tutorial** | Levine et al. | 2020 | Batch RL methods |
| **Multi-Agent RL Survey** | Zhang et al. | 2021 | MARL methods |
| **Safe RL Survey** | Garcia & Fernandez | 2015 | Safety in RL |

---
