# Awesome Agents Papers Collection

A comprehensive collection of papers and presentation slides on LLM agents, reasoning, and AI systems.

> Sources:
> - [arvindcr4/awesome-agents](https://github.com/arvindcr4/awesome-agents)
> - [redhat-et/agentic-reasoning-reinforcement-fine-tuning](https://github.com/redhat-et/agentic-reasoning-reinforcement-fine-tuning)

## Quick Stats
- **Papers:** 88 PDFs (organized in 12 folders)
- **Slides:** 92 presentation decks (~504 MB)
- **Topics:** 15 categories
- **Resources:** See [DEEP_RL_RESOURCES.md](DEEP_RL_RESOURCES.md) for comprehensive RL learning materials

## Folder Structure

```
papers/
├── agent-frameworks/    # 10 papers - ReAct, AutoGen, DSPy, etc.
├── benchmarks/          #  6 papers - SWE-bench, WorkArena, evals
├── computer-use/        #  5 papers - OSWorld, DigiRL, SWE-agent
├── memory-rag/          #  3 papers - HippoRAG, retrieval systems
├── multi-agent/         #  2 papers - AgentNet, MasRouter
├── planning/            #  5 papers - Tree search, optimization
├── reasoning/           #  9 papers - Chain-of-thought, reasoning
├── rl-finetuning/       # 16 papers - DeepSeek R1, GRPO, DPO
├── robotics/            #  6 papers - Eureka, Voyager, GR00T
├── security/            # 10 papers - Prompt injection, red-teaming
├── theorem-proving/     #  9 papers - LeanDojo, AlphaGeometry
└── web-agents/          #  7 papers - WebArena, Mind2Web

slides/                  # 92 presentation decks (504 MB)
```

---

## Table of Contents
- [Inference-Time Techniques](#inference-time-techniques)
- [Post-Training & Alignment](#post-training--alignment)
- [Memory & Planning](#memory--planning)
- [Agent Frameworks](#agent-frameworks)
- [Code Generation & Software Agents](#code-generation--software-agents)
- [Web & Multimodal Agents](#web--multimodal-agents)
- [Enterprise & Workflow Agents](#enterprise--workflow-agents)
- [Mathematics & Theorem Proving](#mathematics--theorem-proving)
- [Robotics & Embodied Agents](#robotics--embodied-agents)
- [Scientific Discovery](#scientific-discovery)
- [Safety & Security](#safety--security)
- [Evaluation & Benchmarking](#evaluation--benchmarking)
- [Neural & Symbolic Reasoning](#neural--symbolic-reasoning)
- [Agentic Reasoning & RL Fine-Tuning](#agentic-reasoning--rl-fine-tuning)
- [Agentic Architectures & Coordination](#agentic-architectures--coordination)
- [Deep Reinforcement Learning](#deep-reinforcement-learning)

---

## Inference-Time Techniques

| Paper | Slides | Code |
|-------|--------|------|
| [Large Language Models as Optimizers](papers/planning/Large_Language_Models_as_Optimizers.pdf) | - | - |
| [Large Language Models Cannot Self-Correct Reasoning Yet](papers/reasoning/Large_Language_Models_Cannot_Self-Correct_Reasoning_Yet.pdf) | - | - |
| [Teaching Large Language Models to Self-Debug](papers/agent-frameworks/Teaching_Large_Language_Models_to_Self-Debug.pdf) | - | - |
| [Chain-of-Thought Reasoning Without Prompting](papers/reasoning/Chain-of-Thought_Reasoning_Without_Prompting.pdf) | [CoT Princeton Lecture](slides/CoT_Princeton_Lecture.pdf), [CoT Toronto](slides/CoT_Toronto_Presentation.pdf), [CoT SJTU](slides/CoT_SJTU_Slides.pdf), [CoT Interpretable ML](slides/CoT_Interpretable_ML_Lecture.pdf), [Concise CoT](slides/Concise_CoT_Benefits.pdf) | - |
| [Premise Order Matters in Reasoning with LLMs](papers/reasoning/Premise_Order_Matters_in_Reasoning_with_Large_Language_Models.pdf) | - | - |
| [Chain-of-Thought Empowers Transformers](papers/reasoning/Chain-of-Thought_Empowers_Transformers_to_Solve_Inherently_Serial_Problems.pdf) | [CoT Slides](slides/CoT_SJTU_Slides.pdf) | - |

## Post-Training & Alignment

| Paper | Slides | Code |
|-------|--------|------|
| [Direct Preference Optimization (DPO)](papers/rl-finetuning/Direct_Preference_Optimization.pdf) | [DPO CMU](slides/DPO_CMU_Lecture.pdf), [DPO UT Austin](slides/DPO_UT_Austin.pdf), [DPO Toronto](slides/DPO_Toronto_Presentation.pdf), [DPO Jinen](slides/DPO_Jinen_Slides.pdf) | [GitHub](https://github.com/eric-mitchell/direct-preference-optimization) |
| [Iterative Reasoning Preference Optimization](papers/reasoning/Iterative_Reasoning_Preference_Optimization.pdf) | - | - |
| [Chain-of-Verification Reduces Hallucination](papers/reasoning/Chain-of-Verification_Reduces_Hallucination.pdf) | - | - |
| [Unpacking DPO and PPO](papers/rl-finetuning/Unpacking_DPO_and_PPO.pdf) | [DPO Slides](slides/DPO_CMU_Lecture.pdf) | - |
| **RLHF Background** | [RLHF UT Austin](slides/RLHF_UT_Austin_Slides.pdf) | - |

## Memory & Planning

| Paper | Slides | Code |
|-------|--------|------|
| [Grokked Transformers are Implicit Reasoners](papers/reasoning/Grokked_Transformers_are_Implicit_Reasoners.pdf) | - | - |
| [HippoRAG: Neurobiologically Inspired Long-Term Memory](papers/memory-rag/HippoRAG_Neurobiologically_Inspired_Long-Term_Memory.pdf) | [HippoRAG NeurIPS](slides/HippoRAG_NeurIPS_Slides.pdf) | [GitHub](https://github.com/OSU-NLP-Group/HippoRAG) |
| [Is Your LLM Secretly a World Model of the Internet](papers/memory-rag/Is_Your_LLM_Secretly_a_World_Model_of_the_Internet.pdf) | - | - |
| [Tree Search for Language Model Agents](papers/planning/Tree_Search_for_Language_Model_Agents.pdf) | - | - |

## Agent Frameworks

| Paper | Slides | Code |
|-------|--------|------|
| [ReAct: Synergizing Reasoning and Acting](papers/agent-frameworks/ReAct_Synergizing_Reasoning_and_Acting.pdf) | [ReAct UVA Lecture](slides/ReAct_UVA_Lecture.pdf) | [GitHub](https://github.com/ysymyth/ReAct) |
| [AutoGen: Multi-Agent Conversation](papers/agent-frameworks/AutoGen_Multi-Agent_Conversation.pdf) | - | [GitHub](https://github.com/microsoft/autogen) |
| [StateFlow: Enhancing LLM Task-Solving](papers/agent-frameworks/StateFlow_Enhancing_LLM_Task-Solving.pdf) | - | [GitHub](https://github.com/yiranwu0/StateFlow) |
| [DSPy: Compiling Declarative Language Model](papers/agent-frameworks/DSPy_Compiling_Declarative_Language_Model.pdf) | - | [GitHub](https://github.com/stanfordnlp/dspy) |
| **LLM Agents Tutorials** | [EMNLP 2024 Tutorial](slides/EMNLP2024_Language_Agents_Tutorial.pdf), [WWW 2024 Tutorial](slides/WWW2024_LLM_Agents_Tutorial.pdf), [Berkeley Training Agents](slides/Berkeley_LLM_Training_Agents.pdf) | - |

## Code Generation & Software Agents

| Paper | Slides | Code |
|-------|--------|------|
| [SWE-agent: Agent-Computer Interfaces](papers/computer-use/SWE-agent_Agent-Computer_Interfaces.pdf) | [Software Agents (Neubig)](slides/Software_Agents_Neubig.pdf) | [GitHub](https://github.com/SWE-agent/SWE-agent) |
| [OpenHands: AI Software Developers](papers/agent-frameworks/OpenHands_AI_Software_Developers.pdf) | [Software Agents (Neubig)](slides/Software_Agents_Neubig.pdf) | [GitHub](https://github.com/OpenHands/OpenHands) |
| [Interactive Tools Assist LM Agents Security Vulnerabilities](papers/security/Interactive_Tools_Assist_LM_Agents_Security_Vulnerabilities.pdf) | [Code Agents & Vulnerability Detection](slides/Code_Agents_Vulnerability_Detection_Berkeley.pdf) | - |
| [Big Sleep: LLM Vulnerabilities Real-World](papers/security/Big_Sleep_LLM_Vulnerabilities_Real-World.pdf) | [Code Agents & Vulnerability Detection](slides/Code_Agents_Vulnerability_Detection_Berkeley.pdf) | - |
| [SWE-bench Verified](papers/benchmarks/SWE-bench_Verified.pdf) | - | [GitHub](https://github.com/SWE-bench/SWE-bench) |

## Web & Multimodal Agents

| Paper | Slides | Code |
|-------|--------|------|
| [WebShop: Scalable Real-World Web Interaction](papers/web-agents/WebShop_Scalable_Real-World_Web_Interaction.pdf) | [Multimodal Agents Berkeley](slides/Multimodal_Agents_Berkeley.pdf) | [GitHub](https://github.com/princeton-nlp/WebShop) |
| [Mind2Web: Generalist Agent for the Web](papers/web-agents/Mind2Web_Generalist_Agent_for_the_Web.pdf) | [Multimodal Agents Berkeley](slides/Multimodal_Agents_Berkeley.pdf) | [GitHub](https://github.com/OSU-NLP-Group/Mind2Web) |
| [WebArena: Realistic Web Environment](papers/web-agents/WebArena_Realistic_Web_Environment.pdf) | [Multimodal Agents Berkeley](slides/Multimodal_Agents_Berkeley.pdf), [Web Agent Evaluation](slides/Web_Agent_Evaluation_Refinement.pdf) | [GitHub](https://github.com/web-arena-x/webarena) |
| [VisualWebArena](papers/web-agents/VisualWebArena.pdf) | [Multimodal Agents Berkeley](slides/Multimodal_Agents_Berkeley.pdf) | [GitHub](https://github.com/web-arena-x/visualwebarena) |
| [AGUVIS: Unified Pure Vision Agents GUI](papers/web-agents/AGUVIS_Unified_Pure_Vision_Agents_GUI.pdf) | - | [GitHub](https://github.com/xlang-ai/aguvis) |
| [BrowseComp: Web Browsing Benchmark](papers/web-agents/BrowseComp_Web_Browsing_Benchmark.pdf) | - | - |

## Enterprise & Workflow Agents

| Paper | Slides | Code |
|-------|--------|------|
| [WorkArena: Common Knowledge Work Tasks](papers/benchmarks/WorkArena_Common_Knowledge_Work_Tasks.pdf) | - | [GitHub](https://github.com/ServiceNow/WorkArena) |
| [WorkArena++: Compositional Planning](papers/benchmarks/WorkArena_Compositional_Planning.pdf) | - | [GitHub](https://github.com/ServiceNow/WorkArena) |
| [TapeAgents: Holistic Framework Agent Development](papers/agent-frameworks/TapeAgents_Holistic_Framework_Agent_Development.pdf) | - | [GitHub](https://github.com/ServiceNow/TapeAgents) |

## Mathematics & Theorem Proving

| Paper | Slides | Code |
|-------|--------|------|
| [LeanDojo: Theorem Proving Retrieval-Augmented](papers/theorem-proving/LeanDojo_Theorem_Proving_Retrieval-Augmented.pdf) | [LeanDojo AITP](slides/LeanDojo_AITP_Slides.pdf), [LeanDojo NeurIPS](slides/LeanDojo_NeurIPS_Slides.pdf), [Theorem Proving ML](slides/Theorem_Proving_ML_Slides.pdf) | [GitHub](https://github.com/lean-dojo/LeanDojo) |
| [Autoformalization with Large Language Models](papers/theorem-proving/Autoformalization_with_Large_Language_Models.pdf) | - | - |
| [Autoformalizing Euclidean Geometry](papers/theorem-proving/Autoformalizing_Euclidean_Geometry.pdf) | - | - |
| [Draft, Sketch and Prove: Formal Theorem Provers](papers/theorem-proving/Draft_Sketch_and_Prove_Formal_Theorem_Provers.pdf) | [Theorem Proving ML](slides/Theorem_Proving_ML_Slides.pdf) | - |
| [miniCTX: Neural Theorem Proving Long-Contexts](papers/theorem-proving/miniCTX_Neural_Theorem_Proving_Long-Contexts.pdf) | - | - |
| [Lean-STaR: Interleave Thinking and Proving](papers/theorem-proving/Lean-STaR_Interleave_Thinking_and_Proving.pdf) | - | - |
| [ImProver: Agent-Based Automated Proof Optimization](papers/theorem-proving/ImProver_Agent-Based_Automated_Proof_Optimization.pdf) | - | - |
| [In-Context Learning Agent Formal Theorem-Proving](papers/theorem-proving/In-Context_Learning_Agent_Formal_Theorem-Proving.pdf) | - | - |
| [Symbolic Regression: Learned Concept Library](papers/planning/Symbolic_Regression_Learned_Concept_Library.pdf) | - | - |
| [AlphaGeometry: Solving Olympiad Geometry](papers/theorem-proving/AlphaGeometry_Solving_Olympiad_Geometry.pdf) | - | [GitHub](https://github.com/google-deepmind/alphageometry) |

## Robotics & Embodied Agents

| Paper | Slides | Code |
|-------|--------|------|
| [Voyager: Open-Ended Embodied Agent](papers/robotics/Voyager_Open-Ended_Embodied_Agent.pdf) | [Voyager UT Austin](slides/Voyager_UT_Austin_Presentation.pdf) | [GitHub](https://github.com/MineDojo/Voyager) |
| [Eureka: Human-Level Reward Design](papers/robotics/Eureka_Human-Level_Reward_Design.pdf) | [Eureka Paper/Slides](slides/Eureka_Reward_Design_Paper.pdf) | [GitHub](https://github.com/eureka-research/Eureka) |
| [DrEureka: Language Model Guided Sim-To-Real](papers/robotics/DrEureka_Language_Model_Guided_Sim-To-Real.pdf) | - | [GitHub](https://github.com/eureka-research/DrEureka) |
| [Gran Turismo: Deep Reinforcement Learning](papers/robotics/Gran_Turismo_Deep_Reinforcement_Learning.pdf) | - | - |
| [GR00T N1: Foundation Model Humanoid](papers/robotics/GR00T_N1_Foundation_Model_Humanoid.pdf) | - | [GitHub](https://github.com/NVIDIA/Isaac-GR00T) |
| [SLAC: Simulation-Pretrained Latent Action](papers/robotics/SLAC_Simulation-Pretrained_Latent_Action.pdf) | - | - |

## Scientific Discovery

| Paper | Slides | Code |
|-------|--------|------|
| [Paper2Agent: Research Papers as AI Agents](papers/agent-frameworks/Paper2Agent_Research_Papers_as_AI_Agents.pdf) | - | - |
| [OpenScholar: Synthesizing Scientific Literature](papers/memory-rag/OpenScholar_Synthesizing_Scientific_Literature.pdf) | - | [GitHub](https://github.com/AkariAsai/OpenScholar) |

## Safety & Security

| Paper | Slides | Code |
|-------|--------|------|
| [DataSentinel: Game-Theoretic Detection Prompt Injection](papers/security/DataSentinel_Game-Theoretic_Detection_Prompt_Injection.pdf) | [Prompt Injection Duke](slides/Prompt_Injection_Duke_Slides.pdf) | - |
| [AgentPoison: Red-teaming LLM Agents](papers/security/AgentPoison_Red-teaming_LLM_Agents.pdf) | [Prompt Injection Duke](slides/Prompt_Injection_Duke_Slides.pdf) | [GitHub](https://github.com/AI-secure/AgentPoison) |
| [Progent: Programmable Privilege Control](papers/security/Progent_Programmable_Privilege_Control.pdf) | - | - |
| [DecodingTrust: Trustworthiness GPT Models](papers/security/DecodingTrust_Trustworthiness_GPT_Models.pdf) | - | [GitHub](https://github.com/AI-secure/DecodingTrust) |
| [Representation Engineering: AI Transparency](papers/security/Representation_Engineering_AI_Transparency.pdf) | - | [GitHub](https://github.com/andyzoujm/representation-engineering) |
| [Extracting Training Data from LLMs](papers/security/Extracting_Training_Data_from_LLMs.pdf) | - | - |
| [The Secret Sharer: Unintended Memorization](papers/security/The_Secret_Sharer_Unintended_Memorization.pdf) | - | - |
| [Privtrans: Privilege Separation](papers/security/Privtrans_Privilege_Separation.pdf) | - | - |

## Evaluation & Benchmarking

| Paper | Slides | Code |
|-------|--------|------|
| [Survey: Evaluation LLM-based Agents](papers/benchmarks/Survey_Evaluation_LLM-based_Agents.pdf) | [AgentBench Multi-Turn NeurIPS](slides/AgentBench_Multi_Turn_NeurIPS.pdf) | - |
| [Adding Error Bars to Evals](papers/benchmarks/Adding_Error_Bars_to_Evals.pdf) | - | - |
| [Tau2-Bench: Conversational Agents Dual-Control](papers/benchmarks/Tau2-Bench_Conversational_Agents_Dual-Control.pdf) | - | - |
| **Data Science Agents** | [Data Science Agents Benchmark](slides/Data_Science_Agents_Benchmark.pdf) | - |

## Neural & Symbolic Reasoning

| Paper | Slides | Code |
|-------|--------|------|
| [Beyond A-Star: Better Planning Transformers](papers/reasoning/Beyond_A-Star_Better_Planning_Transformers.pdf) | - | - |
| [Dualformer: Controllable Fast and Slow Thinking](papers/reasoning/Dualformer_Controllable_Fast_and_Slow_Thinking.pdf) | - | - |
| [Composing Global Optimizers: Algebraic Objects](papers/planning/Composing_Global_Optimizers_Algebraic_Objects.pdf) | - | - |
| [SurCo: Learning Linear Surrogates](papers/planning/SurCo_Learning_Linear_Surrogates.pdf) | - | - |

## Agentic Reasoning & RL Fine-Tuning

> Source: [redhat-et/agentic-reasoning-reinforcement-fine-tuning](https://github.com/redhat-et/agentic-reasoning-reinforcement-fine-tuning)

### DeepSeek R1 & Reasoning Models

| Paper | Slides | Code |
|-------|--------|------|
| [DeepSeek-R1: Reasoning via RL](papers/rl-finetuning/DeepSeek-R1_Reasoning_via_RL.pdf) | [DeepSeek R1 Intro](slides/DeepSeek_R1_Introduction.pdf), [DeepSeek R1 Toronto](slides/DeepSeek_R1_Toronto.pdf), [DeepSeek R1 CMU](slides/DeepSeek_R1_CMU_Reasoning.pdf), [DeepSeek R1 Seoul](slides/DeepSeek_R1_Seoul_National.pdf) | [GitHub](https://github.com/deepseek-ai/DeepSeek-R1) |
| [DeepSeek R1: Implications for AI](papers/rl-finetuning/DeepSeek_R1_Implications_for_AI.pdf) | [DeepSeek R1 Intro](slides/DeepSeek_R1_Introduction.pdf) | - |
| [DeepSeek R1: Are Reasoning Models Faithful?](papers/rl-finetuning/DeepSeek_R1_Reasoning_Models_Faithful.pdf) | - | - |
| [OpenAI O1 Replication Journey](papers/rl-finetuning/OpenAI_O1_Replication_Journey.pdf) | - | - |
| [Qwen QwQ Reasoning Model](papers/rl-finetuning/Qwen_QwQ_Reasoning_Model.pdf) | - | - |
| [Sky-T1: Training Small Reasoning LLMs](papers/rl-finetuning/Sky-T1_Training_Small_Reasoning_LLMs.pdf) | - | - |
| [s1: Simple Test-Time Scaling](papers/rl-finetuning/s1_Simple_Test-Time_Scaling.pdf) | - | - |

### GRPO & RL Fine-Tuning

| Paper | Slides | Code |
|-------|--------|------|
| [DeepSeekMath: GRPO Algorithm](papers/rl-finetuning/DeepSeekMath_GRPO.pdf) | [Stanford RL for Reasoning](slides/Stanford_RL_for_LLM_Reasoning.pdf) | [GitHub](https://github.com/deepseek-ai/DeepSeek-Math) |
| [Guided GRPO: Adaptive Guidance](papers/rl-finetuning/Guided_GRPO_Adaptive_Guidance.pdf) | [PTA-GRPO Planning](slides/PTA_GRPO_Planning_Reasoning.pdf) | - |
| [R-Search: Multi-Step Reasoning](papers/rl-finetuning/R-Search_Multi-Step_Reasoning.pdf) | [Stanford RL for Reasoning](slides/Stanford_RL_for_LLM_Reasoning.pdf) | - |
| [RL Fine-tuning: Instruction Following](papers/rl-finetuning/RL_Fine-tuning_Instruction_Following.pdf) | - | - |
| [RFT Powers Multimodal Reasoning](papers/rl-finetuning/RFT_Powers_Multimodal_Reasoning.pdf) | - | - |
| [STILL-2: Distilling Reasoning](papers/rl-finetuning/STILL-2_Distilling_Reasoning.pdf) | - | - |

### Agentic RL

| Paper | Slides | Code |
|-------|--------|------|
| [WebAgent-R1: Multi-Turn RL for Web Agents](papers/web-agents/WebAgent-R1_Multi-Turn_RL.pdf) | - | - |
| [ARTIST: Agentic Reasoning & Tool Integration](papers/agent-frameworks/ARTIST_Agentic_Reasoning_Tool_Integration.pdf) | [ARTIST Microsoft](slides/ARTIST_Agentic_Reasoning_Microsoft.pdf) | - |

## Agentic Architectures & Coordination

> Papers on multi-agent systems, decentralized coordination, and agentic frameworks

### Decentralized Multi-Agent Systems

| Paper | Slides | Code |
|-------|--------|------|
| [AgentNet: Decentralized Multi-Agent Coordination](papers/multi-agent/AgentNet_Decentralized_Multi-Agent.pdf) | - | - |
| [MasRouter: Multi-Agent Routing](papers/multi-agent/MasRouter_Multi-Agent_Routing.pdf) | [MasRouter ACL 2025](slides/MasRouter_ACL_2025.pdf) | - |
| **Multi-Agent RL Overview** | [Edinburgh MARL Intro](slides/Edinburgh_Multi_Agent_RL_Intro.pdf) | - |

### Device & Computer Control

| Paper | Slides | Code |
|-------|--------|------|
| [DigiRL: Device Control Agents](papers/computer-use/DigiRL_Device_Control_Agents.pdf) | [DigiRL NeurIPS 2024](slides/DigiRL_NeurIPS_2024.pdf) | [GitHub](https://github.com/DigiRL-agent/digirl) |
| [OSWorld: Multimodal Agents Benchmark](papers/computer-use/OSWorld_Multimodal_Agents_Benchmark.pdf) | - | [GitHub](https://github.com/xlang-ai/OSWorld) |
| [OS-Harm: Computer Use Safety](papers/computer-use/OS-Harm_Computer_Use_Safety.pdf) | [OS-Harm Benchmark](slides/OS_Harm_Benchmark.pdf) | - |

### Agent Fine-Tuning & Tool Use

| Paper | Slides | Code |
|-------|--------|------|
| [FireAct: Language Agent Fine-tuning](papers/agent-frameworks/FireAct_Language_Agent_Fine-tuning.pdf) | [LLM Agents Tool Learning](slides/LLM_Agents_Tool_Learning_Tutorial.pdf) | [GitHub](https://github.com/anchen1011/FireAct) |
| [DeepSeek Janus Pro: Multimodal](papers/rl-finetuning/DeepSeek_Janus_Pro_Multimodal.pdf) | - | - |
| [PTA-GRPO: High-Level Planning](slides/PTA_GRPO_High_Level_Planning.pdf) | [PTA-GRPO Planning](slides/PTA_GRPO_High_Level_Planning.pdf) | - |
| **Stanford RL for Agents** | [Stanford RL Agents 2025](slides/Stanford_RL_for_Agents_2025.pdf) | - |
| **CMU LM Agents** | [CMU Language Models as Agents](slides/CMU_Language_Models_as_Agents.pdf) | - |
| **Mannheim Tool Use** | [Mannheim LLM Agents Tool Use](slides/Mannheim_LLM_Agents_Tool_Use.pdf) | - |

### Enterprise & Industry Guides

| Resource | Description | Code |
|----------|-------------|------|
| [Intel AI Agents Architecture](slides/Intel_AI_Agents_Architecture.pdf) | AI agents resource guide | - |
| [Cisco Agentic Frameworks](slides/Cisco_Agentic_Frameworks_Overview.pdf) | Overview of agentic frameworks | - |

---

## Deep Reinforcement Learning

> **[See Full Deep RL Resources Guide](DEEP_RL_RESOURCES.md)** - Comprehensive collection with 100+ resources and 92 slides

### Value-Based Methods (DQN Family)

| Paper | arXiv | Slides | Code |
|-------|-------|--------|------|
| Playing Atari with Deep RL (DQN) | [1312.5602](https://arxiv.org/abs/1312.5602) | [CMU](slides/DQN_CMU_Deep_Q_Learning.pdf), [CVUT](slides/DQN_CVUT_Q_Learning.pdf), [NTHU](slides/DQN_NTHU_Deep_RL.pdf), [Waterloo](slides/DQN_Waterloo_CS885.pdf) | [OpenAI Baselines](https://github.com/openai/baselines) |
| Deep RL with Double Q-learning | [1509.06461](https://arxiv.org/abs/1509.06461) | [CMU DQN](slides/DQN_CMU_Deep_Q_Learning.pdf) | [OpenAI Baselines](https://github.com/openai/baselines) |
| Dueling Network Architectures | [1511.06581](https://arxiv.org/abs/1511.06581) | [Buffalo](slides/Dueling_DQN_PER_Buffalo.pdf) | [OpenAI Baselines](https://github.com/openai/baselines) |
| Prioritized Experience Replay | [1511.05952](https://arxiv.org/abs/1511.05952) | [Buffalo](slides/Dueling_DQN_PER_Buffalo.pdf), [Julien Vitay](slides/PER_Julien_Vitay.pdf), [ICML 2020](slides/Experience_Replay_ICML2020.pdf) | [OpenAI Baselines](https://github.com/openai/baselines) |
| Rainbow: Combining Improvements | [1710.02298](https://arxiv.org/abs/1710.02298) | [Prague](slides/Rainbow_Prague_NPFL122.pdf), [Berkeley](slides/Rainbow_Berkeley_Off_Policy.pdf), [Wisconsin](slides/Rainbow_Wisconsin_CS760.pdf) | [Dopamine](https://github.com/google/dopamine) |

### Policy Gradient Methods

| Paper | arXiv | Slides | Code |
|-------|-------|--------|------|
| Policy Gradient Methods | - | [Toronto](slides/Policy_Gradient_Toronto.pdf), [Berkeley CS285](slides/Policy_Gradient_Berkeley_CS285.pdf), [REINFORCE Stanford](slides/REINFORCE_Stanford_CS229.pdf) | [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) |
| Proximal Policy Optimization (PPO) | [1707.06347](https://arxiv.org/abs/1707.06347) | [Waterloo](slides/PPO_Waterloo_CS885.pdf), [NTU Taiwan](slides/PPO_NTU_Taiwan.pdf) | [OpenAI Baselines](https://github.com/openai/baselines) |
| Trust Region Policy Optimization (TRPO) | [1502.05477](https://arxiv.org/abs/1502.05477) | [FAU](slides/TRPO_FAU_Mutschler.pdf), [UT Austin](slides/TRPO_UT_Austin.pdf), [CMU Natural PG](slides/TRPO_CMU_Natural_PG.pdf), [Toronto PAIR](slides/TRPO_Toronto_PAIR.pdf) | [OpenAI Baselines](https://github.com/openai/baselines) |
| High-Dimensional Continuous Control (GAE) | [1506.02438](https://arxiv.org/abs/1506.02438) | [Berkeley CS285](slides/GAE_Berkeley_CS285.pdf) | [OpenAI Baselines](https://github.com/openai/baselines) |

### Actor-Critic Methods

| Paper | arXiv | Slides | Code |
|-------|-------|--------|------|
| Asynchronous Methods (A3C) | [1602.01783](https://arxiv.org/abs/1602.01783) | [WPI](slides/A3C_WPI_DS595.pdf), [Buffalo](slides/A3C_Buffalo_Actor_Critic.pdf), [NTU](slides/A3C_NTU_Taiwan.pdf), [UIUC](slides/A3C_UIUC_ECE448.pdf), [Julien Vitay](slides/A3C_Julien_Vitay.pdf) | [OpenAI Baselines](https://github.com/openai/baselines) |
| Continuous Control (DDPG) | [1509.02971](https://arxiv.org/abs/1509.02971) | [Paderborn](slides/DDPG_Paderborn_DPG.pdf), [FAU](slides/DDPG_FAU_Mutschler.pdf), [Julien Vitay](slides/DDPG_Julien_Vitay.pdf), [Buffalo](slides/DDPG_Buffalo_DPG.pdf) | [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) |
| Addressing Function Approximation (TD3) | [1802.09477](https://arxiv.org/abs/1802.09477) | [Prague](slides/TD3_SAC_Prague_NPFL139.pdf) | [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) |
| Soft Actor-Critic (SAC) | [1801.01290](https://arxiv.org/abs/1801.01290) | [Toronto PAIR](slides/SAC_Toronto_PAIR.pdf), [Purdue](slides/SAC_Purdue_RL_Inference.pdf), [Stanford CS231n](slides/SAC_Stanford_CS231n.pdf), [Prague](slides/TD3_SAC_Prague_NPFL139.pdf) | [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) |

### Temporal Difference & Q-Learning

| Paper | arXiv | Slides | Code |
|-------|-------|--------|------|
| TD Learning Fundamentals | - | [CMU](slides/TD_Learning_CMU.pdf), [Michigan](slides/TD_Methods_Michigan.pdf), [Sutton & Barto](slides/TD_Sutton_Barto.pdf) | - |
| Q-Learning | - | [Northeastern](slides/Q_Learning_Northeastern.pdf), [CMU TD](slides/TD_Learning_CMU.pdf) | - |

### Model-Based RL

| Paper | arXiv | Slides | Code |
|-------|-------|--------|------|
| Model-Based RL | - | [FAU](slides/Model_Based_RL_FAU.pdf), [Toronto](slides/Model_Based_RL_Toronto.pdf), [Berkeley](slides/Model_Based_RL_Berkeley.pdf), [CMU](slides/Model_Based_RL_CMU.pdf) | [MBRL-Lib](https://github.com/facebookresearch/mbrl-lib) |

### Imitation & Inverse RL

| Paper | arXiv | Slides | Code |
|-------|-------|--------|------|
| Imitation Learning | - | [WPI](slides/Imitation_Learning_WPI.pdf), [EPFL](slides/Imitation_Learning_EPFL.pdf) | [imitation](https://github.com/HumanCompatibleAI/imitation) |
| Inverse Reinforcement Learning | - | [TU Darmstadt](slides/Inverse_RL_TU_Darmstadt.pdf), [Berkeley CS285](slides/Inverse_RL_Berkeley_CS285.pdf) | [imitation](https://github.com/HumanCompatibleAI/imitation) |

### Introductory Lectures

| Topic | Slides |
|-------|--------|
| Deep RL Introduction | [Berkeley CS294](slides/Berkeley_CS294_Intro.pdf), [Berkeley 2017](slides/Berkeley_CS294_Intro_2017.pdf) |

### Frameworks & Tools

| Tool | Link | Description |
|------|------|-------------|
| OpenAI Gym | [GitHub](https://github.com/openai/gym) | RL environments |
| Gymnasium | [GitHub](https://github.com/Farama-Foundation/Gymnasium) | Maintained fork of Gym |
| Stable-Baselines3 | [GitHub](https://github.com/DLR-RM/stable-baselines3) | RL algorithms in PyTorch |
| Unity ML-Agents | [GitHub](https://github.com/Unity-Technologies/ml-agents) | 3D environments |
| PyTorch | [pytorch.org](https://pytorch.org) | Deep learning framework |
| Google Dopamine | [GitHub](https://github.com/google/dopamine) | RL research framework |
| CleanRL | [GitHub](https://github.com/vwxyzjn/cleanrl) | Single-file RL implementations |
| RLlib | [GitHub](https://github.com/ray-project/ray/tree/master/rllib) | Scalable RL library |

**[View all 100+ resources in DEEP_RL_RESOURCES.md](DEEP_RL_RESOURCES.md)**

---

## Recommended Study Path

### Beginner
1. Start with [WWW 2024 LLM Agents Tutorial](slides/WWW2024_LLM_Agents_Tutorial.pdf) - comprehensive overview
2. Read [ReAct paper](papers/agent-frameworks/ReAct_Synergizing_Reasoning_and_Acting.pdf) + [slides](slides/ReAct_UVA_Lecture.pdf) + [code](https://github.com/ysymyth/ReAct)
3. Study Chain-of-Thought with [CoT Princeton Lecture](slides/CoT_Princeton_Lecture.pdf)

### Intermediate
1. [Software Agents (Neubig)](slides/Software_Agents_Neubig.pdf) for code agents + [SWE-agent code](https://github.com/SWE-agent/SWE-agent)
2. [DPO CMU Lecture](slides/DPO_CMU_Lecture.pdf) for alignment + [DPO code](https://github.com/eric-mitchell/direct-preference-optimization)
3. [Multimodal Agents Berkeley](slides/Multimodal_Agents_Berkeley.pdf) for web agents + [WebArena code](https://github.com/web-arena-x/webarena)

### Advanced
1. [LeanDojo slides](slides/LeanDojo_NeurIPS_Slides.pdf) for theorem proving + [code](https://github.com/lean-dojo/LeanDojo)
2. [HippoRAG NeurIPS](slides/HippoRAG_NeurIPS_Slides.pdf) for memory systems + [code](https://github.com/OSU-NLP-Group/HippoRAG)
3. [Prompt Injection Duke](slides/Prompt_Injection_Duke_Slides.pdf) for security

### Reasoning & RL Fine-Tuning Path
1. [DeepSeek-R1 paper](papers/rl-finetuning/DeepSeek-R1_Reasoning_via_RL.pdf) + [DeepSeek R1 CMU slides](slides/DeepSeek_R1_CMU_Reasoning.pdf) + [code](https://github.com/deepseek-ai/DeepSeek-R1)
2. [DeepSeekMath GRPO](papers/rl-finetuning/DeepSeekMath_GRPO.pdf) + [Stanford RL for Reasoning](slides/Stanford_RL_for_LLM_Reasoning.pdf) + [code](https://github.com/deepseek-ai/DeepSeek-Math)
3. [ARTIST paper](papers/agent-frameworks/ARTIST_Agentic_Reasoning_Tool_Integration.pdf) for agentic reasoning with tools

---

## License

Papers are property of their respective authors. This collection is for educational purposes.
