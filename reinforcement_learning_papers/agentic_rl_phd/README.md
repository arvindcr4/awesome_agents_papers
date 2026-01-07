# Agentic RL PhD: The Roadmap Codebase

This repository contains the scaffolding and reference implementations for the "PhD-level" curriculum in Reinforcement Learning and Autonomous Agents.

## Structure

### 1. Algorithms (The Math)
*   **`algorithms/ppo.py`**: Implementation of **Proximal Policy Optimization (PPO)**.
    *   *Paper:* Schulman et al., 2017
    *   *Status:* Implemented (Core Actor-Critic logic).
*   **`algorithms/dpo.py`**: Implementation of **Direct Preference Optimization (DPO)**.
    *   *Paper:* Rafailov et al., 2023
    *   *Status:* Implemented (Loss function).
*   **[TODO] `algorithms/cql.py`**: Conservative Q-Learning (Offline RL).
    *   *Paper:* Kumar et al., 2020
*   **[TODO] `algorithms/sac.py`**: Soft Actor-Critic.
    *   *Paper:* Haarnoja et al., 2018

### 2. Agents (The Reasoning)
*   **`agents/reflexion.py`**: Implementation of **Reflexion** (Verbal RL).
    *   *Paper:* Shinn et al., 2023
    *   *Status:* Implemented (Verbal feedback loop).
*   **[TODO] `agents/react.py`**: ReAct (Reasoning + Acting).
    *   *Paper:* Yao et al., 2022
*   **[TODO] `agents/voyager.py`**: Voyager (Coding Agent).
    *   *Paper:* Wang et al., 2023
*   **[TODO] `agents/tree_of_thoughts.py`**: ToT (Search over thoughts).
    *   *Paper:* Yao et al., 2023

### 3. World Models (The Simulation)
*   **`world_models/simple_dreamer.py`**: Basic World Model architecture.
    *   *Papers:* Ha & Schmidhuber (2018), Hafner et al. (DreamerV3)
    *   *Status:* Implemented (Encoder/RNN/Controller scaffold).
*   **[TODO] `world_models/muzero.py`**: Planning with learned models.
    *   *Paper:* Schrittwieser et al., 2019

### 4. Cutting Edge (2025 Focus)
*   **[TODO] `experiments/rlvr_reasoning.py`**: Replication of "Does RL Incentivize Reasoning?"
    *   *Goal:* Test if PPO improves pass@k or just pass@1.
*   **[TODO] `experiments/1000_layer_rl.py`**: Scaling ResNets for RL.

## How to use this for your PhD

1.  **Read** the papers in the root directory.
2.  **Run** the implementations in this folder to understand the basics.
3.  **Fill in the [TODO]s**. Implementing MuZero from scratch is a rite of passage.
4.  **Find the Gap**: Try combining `agents/reflexion.py` with `world_models/simple_dreamer.py`. Can an agent "reflect" inside its own "dream"?
