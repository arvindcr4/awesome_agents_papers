# Deep Research Results: Reinforcement Learning

This document contains the extracted research results from your ChatGPT history regarding Reinforcement Learning topics.

## 1. Advanced RL Literature Review

### 1) Multi-agent RL dynamics (non-stationarity, credit assignment, equilibrium issues)
- **MADDPG** – *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments* (Lowe et al., 2017).
  - **Impact:** Popularized *centralized training / decentralized execution (CTDE)* with centralized critics to handle non-stationarity.
- **MAPPO/IPPO** – *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games* (Yu et al., 2021).
  - **Impact:** Reset community expectations: strong, simple on-policy baselines in cooperative MARL across standard testbeds.
- **Survey:** *A survey on multi-agent reinforcement learning and its applications* (Ning et al., 2024).

### 2) Cooperative multi-agent techniques
- **QMIX** – *Monotonic Value Function Factorisation for Deep MARL* (Rashid et al., 2018).
  - **Impact:** Made value factorization practical at scale, becoming a standard baseline.
- **MAPPO** – (Yu et al., 2021).

### 3) Competitive / mixed cooperative–competitive MARL
- **MADDPG** (Lowe et al., 2017).
- **MARL dynamics survey** (Ning et al., 2024).

### 4) Emergent communication
- **RIAL/DIAL** – *Learning to Communicate with Deep MARL* (Foerster et al., 2016).
  - **Impact:** Early flagship for learning communication; DIAL showed backprop through message channels.

### 5) Curriculum learning in RL
- **ALP-GMM** – Portelas et al., 2019/2020.
  - **Impact:** Formalized *automatic curriculum learning*.
- **Survey:** *Automatic Curriculum Learning for Deep RL: A Short Survey* (Portelas et al., 2020).

### 6) Continual / lifelong RL
- **Survey:** *A Survey of Continual Reinforcement Learning* (2025).

### 7) Transfer learning & meta-RL
- **MAML** – *Model-Agnostic Meta-Learning for Fast Adaptation* (Finn et al., 2017).
  - **Impact:** Standard gradient-based meta-learning baseline.

### 8) RLHF (Reinforcement Learning from Human Feedback)
- **Deep RL from Human Preferences** (Christiano et al., 2017).
  - **Impact:** Modern template: collect pairwise preferences → train reward model → optimize policy.
- **InstructGPT** (Ouyang et al., 2022).
  - **Impact:** Made RLHF mainstream for LLM alignment.
- **Book:** *Reinforcement Learning from Human Feedback* (Lambert, 2024).

### 9) RLAIF (RL from AI feedback)
- **Constitutional AI** – *Harmlessness from AI Feedback* (Bai et al., 2022).
  - **Impact:** Demonstrated scalable “AI-assisted supervision”.

### 10) Safe exploration & safe RL
- **Survey:** García & Fernández (2015).
- **Feasible Policy Iteration** (Yang et al., 2023).

### 11) Multi-objective RL (MORL)
- **Survey:** Roijers et al., *A survey of multi-objective sequential decision-making*.

### 12) Hierarchical RL
- **Feudal RL:** Dayan & Hinton (1992).
- **FeUdal Networks:** Vezhnevets et al. (2017).

### 13) World models & model-based RL
- **World Models** – Ha & Schmidhuber (2018).
- **DreamerV3** – Hafner et al. (2023).

### 14) Causal inference in RL
- **Survey:** *Causal Reinforcement Learning: A Survey* (Deng et al., 2023).

### 15) Neuro-symbolic approaches
- **Survey:** *Neurosymbolic Reinforcement Learning and Planning: A Survey* (Acharya et al., 2023).

### 16) Explainable RL (XRL)
- **Survey:** *Explainable Reinforcement Learning: A Survey* (Milani et al., 2024).

### 17) Transformer-based agents
- **Decision Transformer** – Chen et al. (2021).
  - **Impact:** Reframed RL as conditional sequence modeling.

### 18) Unsupervised skill discovery
- **DADS** – *Dynamics-Aware Unsupervised Discovery of Skills* (Sharma et al., 2019).

### 19) Generalization & robustness
- **Procgen Benchmark** (Cobbe et al., 2019).
- **Zero-shot generalization survey** – Kirk et al. (2023).

### 20) Open-ended learning
- **POET** – *Paired Open-Ended Trailblazer* (Wang et al., 2019).

### 21) Human-in-the-loop RL
- **Survey:** *Human-in-the-Loop Reinforcement Learning: A Survey* (Retzlaff et al., 2024).

---

## 2. Seminal Papers on Policy Iteration

1. **Bellman, R. E. (1957). *Dynamic Programming*.**
   - Introduced dynamic programming and the Bellman equation.
2. **Howard, R. A. (1960). *Dynamic Programming and Markov Processes*.**
   - Introduced the policy iteration algorithm explicitly.
3. **Puterman, M. L. & Shin, M. C. (1978). “Modified Policy Iteration Algorithms...”**
   - Introduced Modified Policy Iteration (MPI), interpolating between value and policy iteration.
4. **Barto, A. G., et al. (1983). “Neuronlike Adaptive Elements...”**
   - First actor-critic architecture (approximate policy iteration).
5. **Sutton, R. S., et al. (2000). “Policy Gradient Methods...”**
   - Theoretical framework for policy gradient methods.
6. **Kakade, S. & Langford, J. (2002). “Approximately Optimal Approximate RL.”**
   - Introduced Conservative Policy Iteration (CPI).
7. **Lagoudakis, M. G. & Parr, R. (2003). “Least-Squares Policy Iteration.”**
   - Introduced LSPI, off-policy approximate policy iteration using LSTD.
8. **Schulman, J., et al. (2015). “Trust Region Policy Optimization” (TRPO).**
   - Guaranteed monotonic improvement for deep RL policies.
9. **Lillicrap, T. P., et al. (2016). “Continuous Control with Deep RL” (DDPG).**
   - Adapted policy iteration to continuous action spaces with deep nets.
10. **Schulman, J., et al. (2017). “Proximal Policy Optimization Algorithms” (PPO).**
    - Simpler, robust successor to TRPO.
11. **Silver, D., et al. (2017). “Mastering the Game of Go without Human Knowledge” (AlphaGo Zero).**
    - Large-scale policy iteration via self-play and MCTS.

---

## 3. Seminal Papers on MDPs and Sequential Decision Problems

### Foundations (1950s-1960s)
1. **Bellman (1957) – The Principle of Optimality and Dynamic Programming**
   - Formally introduced the Markovian decision process and Bellman equation.
2. **Howard (1960) – Policy Iteration Algorithm for MDPs**
   - Practical algorithm for computing optimal policies.
3. **Manne (1960) – Linking MDPs to Linear Programming**
   - Showed MDPs can be solved via Linear Programming.
4. **Blackwell (1965) – Discounted Dynamic Programming**
   - Rigorous theoretical foundation, contraction mappings, Blackwell optimality.

### AI & Partial Observability (1970s-1990s)
5. **Smallwood & Sondik (1973) – Planning with Partial Observability (POMDPs)**
   - Proved value function is piecewise linear and convex; first practical algorithm.
6. **Sutton (1988) – Temporal-Difference Learning**
   - Introduced TD learning, learning value functions from experience.
7. **Watkins & Dayan (1992) – Q-Learning**
   - Model-free RL learning optimal policies without a model.
8. **Kaelbling, Littman & Cassandra (1998) – Unified Algorithms for POMDPs**
   - Standard reference on POMDPs in AI.
9. **Sutton, Precup & Singh (1999) – Hierarchical RL with Options**
   - Framework for temporal abstraction (macro-actions).

### Modern Breakthroughs (2010s)
10. **Mnih et al. (2015) – Deep Q-Network (DQN)**
    - Human-level control from pixels using deep RL.
11. **Silver et al. (2016) – AlphaGo**
    - Mastering Go with Deep RL and MCTS.

---

## 4. Seminal Papers on Bellman Equations and Value Recursion

### Early Foundations
- **Bellman (1952)**: *On the Theory of Dynamic Programming*
- **Shapley (1953)**: *Stochastic Games* (Extension to multi-agent)
- **Howard (1960)**: *Dynamic Programming and Markov Processes*
- **Blackwell (1965)**: *Discounted Dynamic Programming*
- **Bellman (1958)**: *On a Routing Problem* (Bellman-Ford algorithm)

### RL & AI Advances
- **Sutton (1988)**: *Learning to Predict by the Method of Temporal Differences*
- **Watkins & Dayan (1992)**: *Q-Learning*
- **Mnih et al. (2015)**: *Human-Level Control Through Deep RL*

### Economics & Finance
- **Merton (1973)**: *An Intertemporal Capital Asset Pricing Model* (Continuous-time Bellman/HJB)
- **Stokey, Lucas & Prescott (1989)**: *Recursive Methods in Economic Dynamics*
- **Dixit & Pindyck (1994)**: *Investment Under Uncertainty* (Real options)

---

## 5. RL Topics: Foundations, Architectures, and Algorithms (Detailed Review)

### RL Foundations
- **MDPs**: Bellman (1957), Howard (1960).
- **Bandits**: Robbins (1952), Gittins (1979), Lai & Robbins (1985).
- **POMDPs**: Aström (1965), Smallwood & Sondik (1973), Littman et al. (1995).
- **Exploration**: Sutton & Barto (1998).
- **Credit Assignment**: Minsky (1961), Sutton (1984 - TD Learning).
- **Model-Based vs Free**: Sutton (1990 - Dyna), Kearns & Singh (2002).

### Agent Architectures
- **Value-Based**: TD-Gammon (1992), DQN (2015).
- **Policy-Based**: REINFORCE (1992).
- **Actor-Critic**: Barto et al. (1983), A3C (2016).
- **Hierarchical**: Options (1999), Option-Critic (2017).
- **Memory**: LSTM in RL (Bakker 2002), DRQN (2015).
- **Model-Based Planning**: AlphaGo (2016), MuZero (2019).
- **Multi-Agent**: Minimax-Q (1994), OpenAI Five (2019).

### Core Algorithms
- **DP**: Value/Policy Iteration.
- **TD**: TD(0), TD(lambda), Q-Learning, SARSA.
- **Deep RL**: DQN, Double DQN, Dueling DQN, Rainbow.
- **Policy Gradient**: REINFORCE, TRPO, PPO, DDPG, TD3, SAC.
- **Distributional RL**: C51, QR-DQN, IQN.
- **MCTS**: UCT (2006), AlphaGo.
- **Evolutionary**: NEAT (2002), OpenAI ES (2017).

### Practical Projects
- **TD-Gammon**: Backgammon mastery.
- **Atari DQN**: General game playing.
- **AlphaGo / AlphaZero**: Board game mastery.
- **AlphaStar**: StarCraft II.
- **OpenAI Five**: Dota 2.
- **Data Center Cooling**: Industrial application.
- **Robotics**: Dactyl (Rubik's cube), Boston Dynamics.
