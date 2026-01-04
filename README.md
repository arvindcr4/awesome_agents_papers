# Awesome Agents Papers Collection

A comprehensive collection of papers and presentation slides on LLM agents, reasoning, and AI systems.

> Sources:
> - [arvindcr4/awesome-agents](https://github.com/arvindcr4/awesome-agents)
> - [redhat-et/agentic-reasoning-reinforcement-fine-tuning](https://github.com/redhat-et/agentic-reasoning-reinforcement-fine-tuning)

## Quick Stats
- **Papers:** 88 PDFs
- **Slides:** 45 presentation decks
- **Topics:** 15 categories

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
- [Agentic Architectures & Coordination](#agentic-architectures--coordination) *(NEW)*

---

## Inference-Time Techniques

| Paper | Slides |
|-------|--------|
| [Large Language Models as Optimizers](Large_Language_Models_as_Optimizers.pdf) | - |
| [Large Language Models Cannot Self-Correct Reasoning Yet](Large_Language_Models_Cannot_Self-Correct_Reasoning_Yet.pdf) | - |
| [Teaching Large Language Models to Self-Debug](Teaching_Large_Language_Models_to_Self-Debug.pdf) | - |
| [Chain-of-Thought Reasoning Without Prompting](Chain-of-Thought_Reasoning_Without_Prompting.pdf) | [CoT Princeton Lecture](slides/CoT_Princeton_Lecture.pdf), [CoT Toronto](slides/CoT_Toronto_Presentation.pdf), [CoT SJTU](slides/CoT_SJTU_Slides.pdf), [CoT Interpretable ML](slides/CoT_Interpretable_ML_Lecture.pdf), [Concise CoT](slides/Concise_CoT_Benefits.pdf) |
| [Premise Order Matters in Reasoning with LLMs](Premise_Order_Matters_in_Reasoning_with_Large_Language_Models.pdf) | - |
| [Chain-of-Thought Empowers Transformers](Chain-of-Thought_Empowers_Transformers_to_Solve_Inherently_Serial_Problems.pdf) | [CoT Slides](slides/CoT_SJTU_Slides.pdf) |

## Post-Training & Alignment

| Paper | Slides |
|-------|--------|
| [Direct Preference Optimization (DPO)](Direct_Preference_Optimization.pdf) | [DPO CMU](slides/DPO_CMU_Lecture.pdf), [DPO UT Austin](slides/DPO_UT_Austin.pdf), [DPO Toronto](slides/DPO_Toronto_Presentation.pdf), [DPO Jinen](slides/DPO_Jinen_Slides.pdf) |
| [Iterative Reasoning Preference Optimization](Iterative_Reasoning_Preference_Optimization.pdf) | - |
| [Chain-of-Verification Reduces Hallucination](Chain-of-Verification_Reduces_Hallucination.pdf) | - |
| [Unpacking DPO and PPO](Unpacking_DPO_and_PPO.pdf) | [DPO Slides](slides/DPO_CMU_Lecture.pdf) |
| **RLHF Background** | [RLHF UT Austin](slides/RLHF_UT_Austin_Slides.pdf) |

## Memory & Planning

| Paper | Slides |
|-------|--------|
| [Grokked Transformers are Implicit Reasoners](Grokked_Transformers_are_Implicit_Reasoners.pdf) | - |
| [HippoRAG: Neurobiologically Inspired Long-Term Memory](HippoRAG_Neurobiologically_Inspired_Long-Term_Memory.pdf) | [HippoRAG NeurIPS](slides/HippoRAG_NeurIPS_Slides.pdf) |
| [Is Your LLM Secretly a World Model of the Internet](Is_Your_LLM_Secretly_a_World_Model_of_the_Internet.pdf) | - |
| [Tree Search for Language Model Agents](Tree_Search_for_Language_Model_Agents.pdf) | - |

## Agent Frameworks

| Paper | Slides |
|-------|--------|
| [ReAct: Synergizing Reasoning and Acting](ReAct_Synergizing_Reasoning_and_Acting.pdf) | [ReAct UVA Lecture](slides/ReAct_UVA_Lecture.pdf) |
| [AutoGen: Multi-Agent Conversation](AutoGen_Multi-Agent_Conversation.pdf) | - |
| [StateFlow: Enhancing LLM Task-Solving](StateFlow_Enhancing_LLM_Task-Solving.pdf) | - |
| [DSPy: Compiling Declarative Language Model](DSPy_Compiling_Declarative_Language_Model.pdf) | - |
| **LLM Agents Tutorials** | [EMNLP 2024 Tutorial](slides/EMNLP2024_Language_Agents_Tutorial.pdf), [WWW 2024 Tutorial](slides/WWW2024_LLM_Agents_Tutorial.pdf), [Berkeley Training Agents](slides/Berkeley_LLM_Training_Agents.pdf) |

## Code Generation & Software Agents

| Paper | Slides |
|-------|--------|
| [SWE-agent: Agent-Computer Interfaces](SWE-agent_Agent-Computer_Interfaces.pdf) | [Software Agents (Neubig)](slides/Software_Agents_Neubig.pdf) |
| [OpenHands: AI Software Developers](OpenHands_AI_Software_Developers.pdf) | [Software Agents (Neubig)](slides/Software_Agents_Neubig.pdf) |
| [Interactive Tools Assist LM Agents Security Vulnerabilities](Interactive_Tools_Assist_LM_Agents_Security_Vulnerabilities.pdf) | [Code Agents & Vulnerability Detection](slides/Code_Agents_Vulnerability_Detection_Berkeley.pdf) |
| [Big Sleep: LLM Vulnerabilities Real-World](Big_Sleep_LLM_Vulnerabilities_Real-World.pdf) | [Code Agents & Vulnerability Detection](slides/Code_Agents_Vulnerability_Detection_Berkeley.pdf) |
| [SWE-bench Verified](SWE-bench_Verified.pdf) | - |

## Web & Multimodal Agents

| Paper | Slides |
|-------|--------|
| [WebShop: Scalable Real-World Web Interaction](WebShop_Scalable_Real-World_Web_Interaction.pdf) | [Multimodal Agents Berkeley](slides/Multimodal_Agents_Berkeley.pdf) |
| [Mind2Web: Generalist Agent for the Web](Mind2Web_Generalist_Agent_for_the_Web.pdf) | [Multimodal Agents Berkeley](slides/Multimodal_Agents_Berkeley.pdf) |
| [WebArena: Realistic Web Environment](WebArena_Realistic_Web_Environment.pdf) | [Multimodal Agents Berkeley](slides/Multimodal_Agents_Berkeley.pdf), [Web Agent Evaluation](slides/Web_Agent_Evaluation_Refinement.pdf) |
| [VisualWebArena](VisualWebArena.pdf) | [Multimodal Agents Berkeley](slides/Multimodal_Agents_Berkeley.pdf) |
| [OSWORLD: Benchmarking Multimodal Agents](OSWORLD_Benchmarking_Multimodal_Agents.pdf) | [Multimodal Agents Berkeley](slides/Multimodal_Agents_Berkeley.pdf) |
| [AGUVIS: Unified Pure Vision Agents GUI](AGUVIS_Unified_Pure_Vision_Agents_GUI.pdf) | - |
| [BrowseComp: Web Browsing Benchmark](BrowseComp_Web_Browsing_Benchmark.pdf) | - |

## Enterprise & Workflow Agents

| Paper | Slides |
|-------|--------|
| [WorkArena: Common Knowledge Work Tasks](WorkArena_Common_Knowledge_Work_Tasks.pdf) | - |
| [WorkArena++: Compositional Planning](WorkArena_Compositional_Planning.pdf) | - |
| [TapeAgents: Holistic Framework Agent Development](TapeAgents_Holistic_Framework_Agent_Development.pdf) | - |

## Mathematics & Theorem Proving

| Paper | Slides |
|-------|--------|
| [LeanDojo: Theorem Proving Retrieval-Augmented](LeanDojo_Theorem_Proving_Retrieval-Augmented.pdf) | [LeanDojo AITP](slides/LeanDojo_AITP_Slides.pdf), [LeanDojo NeurIPS](slides/LeanDojo_NeurIPS_Slides.pdf), [Theorem Proving ML](slides/Theorem_Proving_ML_Slides.pdf) |
| [Autoformalization with Large Language Models](Autoformalization_with_Large_Language_Models.pdf) | - |
| [Autoformalizing Euclidean Geometry](Autoformalizing_Euclidean_Geometry.pdf) | - |
| [Draft, Sketch and Prove: Formal Theorem Provers](Draft_Sketch_and_Prove_Formal_Theorem_Provers.pdf) | [Theorem Proving ML](slides/Theorem_Proving_ML_Slides.pdf) |
| [miniCTX: Neural Theorem Proving Long-Contexts](miniCTX_Neural_Theorem_Proving_Long-Contexts.pdf) | - |
| [Lean-STaR: Interleave Thinking and Proving](Lean-STaR_Interleave_Thinking_and_Proving.pdf) | - |
| [ImProver: Agent-Based Automated Proof Optimization](ImProver_Agent-Based_Automated_Proof_Optimization.pdf) | - |
| [In-Context Learning Agent Formal Theorem-Proving](In-Context_Learning_Agent_Formal_Theorem-Proving.pdf) | - |
| [Symbolic Regression: Learned Concept Library](Symbolic_Regression_Learned_Concept_Library.pdf) | - |
| [AlphaGeometry: Solving Olympiad Geometry](AlphaGeometry_Solving_Olympiad_Geometry.pdf) | - |

## Robotics & Embodied Agents

| Paper | Slides |
|-------|--------|
| [Voyager: Open-Ended Embodied Agent](Voyager_Open-Ended_Embodied_Agent.pdf) | [Voyager UT Austin](slides/Voyager_UT_Austin_Presentation.pdf) |
| [Eureka: Human-Level Reward Design](Eureka_Human-Level_Reward_Design.pdf) | [Eureka Paper/Slides](slides/Eureka_Reward_Design_Paper.pdf) |
| [DrEureka: Language Model Guided Sim-To-Real](DrEureka_Language_Model_Guided_Sim-To-Real.pdf) | - |
| [Gran Turismo: Deep Reinforcement Learning](Gran_Turismo_Deep_Reinforcement_Learning.pdf) | - |
| [GR00T N1: Foundation Model Humanoid](GR00T_N1_Foundation_Model_Humanoid.pdf) | - |
| [SLAC: Simulation-Pretrained Latent Action](SLAC_Simulation-Pretrained_Latent_Action.pdf) | - |

## Scientific Discovery

| Paper | Slides |
|-------|--------|
| [Paper2Agent: Research Papers as AI Agents](Paper2Agent_Research_Papers_as_AI_Agents.pdf) | - |
| [OpenScholar: Synthesizing Scientific Literature](OpenScholar_Synthesizing_Scientific_Literature.pdf) | - |

## Safety & Security

| Paper | Slides |
|-------|--------|
| [DataSentinel: Game-Theoretic Detection Prompt Injection](DataSentinel_Game-Theoretic_Detection_Prompt_Injection.pdf) | [Prompt Injection Duke](slides/Prompt_Injection_Duke_Slides.pdf) |
| [AgentPoison: Red-teaming LLM Agents](AgentPoison_Red-teaming_LLM_Agents.pdf) | [Prompt Injection Duke](slides/Prompt_Injection_Duke_Slides.pdf) |
| [Progent: Programmable Privilege Control](Progent_Programmable_Privilege_Control.pdf) | - |
| [DecodingTrust: Trustworthiness GPT Models](DecodingTrust_Trustworthiness_GPT_Models.pdf) | - |
| [Representation Engineering: AI Transparency](Representation_Engineering_AI_Transparency.pdf) | - |
| [Extracting Training Data from LLMs](Extracting_Training_Data_from_LLMs.pdf) | - |
| [The Secret Sharer: Unintended Memorization](The_Secret_Sharer_Unintended_Memorization.pdf) | - |
| [Privtrans: Privilege Separation](Privtrans_Privilege_Separation.pdf) | - |

## Evaluation & Benchmarking

| Paper | Slides |
|-------|--------|
| [Survey: Evaluation LLM-based Agents](Survey_Evaluation_LLM-based_Agents.pdf) | [AgentBench Multi-Turn NeurIPS](slides/AgentBench_Multi_Turn_NeurIPS.pdf) |
| [Adding Error Bars to Evals](Adding_Error_Bars_to_Evals.pdf) | - |
| [Tau2-Bench: Conversational Agents Dual-Control](Tau2-Bench_Conversational_Agents_Dual-Control.pdf) | - |
| **Data Science Agents** | [Data Science Agents Benchmark](slides/Data_Science_Agents_Benchmark.pdf) |

## Neural & Symbolic Reasoning

| Paper | Slides |
|-------|--------|
| [Beyond A-Star: Better Planning Transformers](Beyond_A-Star_Better_Planning_Transformers.pdf) | - |
| [Dualformer: Controllable Fast and Slow Thinking](Dualformer_Controllable_Fast_and_Slow_Thinking.pdf) | - |
| [Composing Global Optimizers: Algebraic Objects](Composing_Global_Optimizers_Algebraic_Objects.pdf) | - |
| [SurCo: Learning Linear Surrogates](SurCo_Learning_Linear_Surrogates.pdf) | - |

## Agentic Reasoning & RL Fine-Tuning

> Source: [redhat-et/agentic-reasoning-reinforcement-fine-tuning](https://github.com/redhat-et/agentic-reasoning-reinforcement-fine-tuning)

### DeepSeek R1 & Reasoning Models

| Paper | Slides |
|-------|--------|
| [DeepSeek-R1: Reasoning via RL](DeepSeek-R1_Reasoning_via_RL.pdf) | [DeepSeek R1 Intro](slides/DeepSeek_R1_Introduction.pdf), [DeepSeek R1 Toronto](slides/DeepSeek_R1_Toronto.pdf), [DeepSeek R1 CMU](slides/DeepSeek_R1_CMU_Reasoning.pdf), [DeepSeek R1 Seoul](slides/DeepSeek_R1_Seoul_National.pdf) |
| [DeepSeek R1: Implications for AI](DeepSeek_R1_Implications_for_AI.pdf) | [DeepSeek R1 Intro](slides/DeepSeek_R1_Introduction.pdf) |
| [DeepSeek R1: Are Reasoning Models Faithful?](DeepSeek_R1_Reasoning_Models_Faithful.pdf) | - |
| [OpenAI O1 Replication Journey](OpenAI_O1_Replication_Journey.pdf) | - |
| [Qwen QwQ Reasoning Model](Qwen_QwQ_Reasoning_Model.pdf) | - |
| [Sky-T1: Training Small Reasoning LLMs](Sky-T1_Training_Small_Reasoning_LLMs.pdf) | - |
| [s1: Simple Test-Time Scaling](s1_Simple_Test-Time_Scaling.pdf) | - |

### GRPO & RL Fine-Tuning

| Paper | Slides |
|-------|--------|
| [DeepSeekMath: GRPO Algorithm](DeepSeekMath_GRPO.pdf) | [Stanford RL for Reasoning](slides/Stanford_RL_for_LLM_Reasoning.pdf) |
| [Guided GRPO: Adaptive Guidance](Guided_GRPO_Adaptive_Guidance.pdf) | [PTA-GRPO Planning](slides/PTA_GRPO_Planning_Reasoning.pdf) |
| [R-Search: Multi-Step Reasoning](R-Search_Multi-Step_Reasoning.pdf) | [Stanford RL for Reasoning](slides/Stanford_RL_for_LLM_Reasoning.pdf) |
| [RL Fine-tuning: Instruction Following](RL_Fine-tuning_Instruction_Following.pdf) | - |
| [RFT Powers Multimodal Reasoning](RFT_Powers_Multimodal_Reasoning.pdf) | - |
| [STILL-2: Distilling Reasoning](STILL-2_Distilling_Reasoning.pdf) | - |

### Agentic RL

| Paper | Slides |
|-------|--------|
| [WebAgent-R1: Multi-Turn RL for Web Agents](WebAgent-R1_Multi-Turn_RL.pdf) | - |
| [ARTIST: Agentic Reasoning & Tool Integration](ARTIST_Agentic_Reasoning_Tool_Integration.pdf) | [ARTIST Microsoft](slides/ARTIST_Agentic_Reasoning_Microsoft.pdf) |

## Agentic Architectures & Coordination

> Papers on multi-agent systems, decentralized coordination, and agentic frameworks

### Decentralized Multi-Agent Systems

| Paper | Slides |
|-------|--------|
| [AgentNet: Decentralized Multi-Agent Coordination](AgentNet_Decentralized_Multi-Agent.pdf) | - |
| [MasRouter: Multi-Agent Routing](MasRouter_Multi-Agent_Routing.pdf) | [MasRouter ACL 2025](slides/MasRouter_ACL_2025.pdf) |
| **Multi-Agent RL Overview** | [Edinburgh MARL Intro](slides/Edinburgh_Multi_Agent_RL_Intro.pdf) |

### Device & Computer Control

| Paper | Slides |
|-------|--------|
| [DigiRL: Device Control Agents](DigiRL_Device_Control_Agents.pdf) | [DigiRL NeurIPS 2024](slides/DigiRL_NeurIPS_2024.pdf) |
| [OSWorld: Multimodal Agents Benchmark](OSWorld_Multimodal_Agents_Benchmark.pdf) | - |
| [OS-Harm: Computer Use Safety](OS-Harm_Computer_Use_Safety.pdf) | [OS-Harm Benchmark](slides/OS_Harm_Benchmark.pdf) |

### Agent Fine-Tuning & Tool Use

| Paper | Slides |
|-------|--------|
| [FireAct: Language Agent Fine-tuning](FireAct_Language_Agent_Fine-tuning.pdf) | [LLM Agents Tool Learning](slides/LLM_Agents_Tool_Learning_Tutorial.pdf) |
| [DeepSeek Janus Pro: Multimodal](DeepSeek_Janus_Pro_Multimodal.pdf) | - |
| [PTA-GRPO: High-Level Planning](slides/PTA_GRPO_High_Level_Planning.pdf) | [PTA-GRPO Planning](slides/PTA_GRPO_High_Level_Planning.pdf) |
| **Stanford RL for Agents** | [Stanford RL Agents 2025](slides/Stanford_RL_for_Agents_2025.pdf) |
| **CMU LM Agents** | [CMU Language Models as Agents](slides/CMU_Language_Models_as_Agents.pdf) |
| **Mannheim Tool Use** | [Mannheim LLM Agents Tool Use](slides/Mannheim_LLM_Agents_Tool_Use.pdf) |

### Enterprise & Industry Guides

| Resource | Description |
|----------|-------------|
| [Intel AI Agents Architecture](slides/Intel_AI_Agents_Architecture.pdf) | AI agents resource guide |
| [Cisco Agentic Frameworks](slides/Cisco_Agentic_Frameworks_Overview.pdf) | Overview of agentic frameworks |

---

## Recommended Study Path

### Beginner
1. Start with [WWW 2024 LLM Agents Tutorial](slides/WWW2024_LLM_Agents_Tutorial.pdf) - comprehensive overview
2. Read [ReAct paper](ReAct_Synergizing_Reasoning_and_Acting.pdf) + [slides](slides/ReAct_UVA_Lecture.pdf)
3. Study Chain-of-Thought with [CoT Princeton Lecture](slides/CoT_Princeton_Lecture.pdf)

### Intermediate
1. [Software Agents (Neubig)](slides/Software_Agents_Neubig.pdf) for code agents
2. [DPO CMU Lecture](slides/DPO_CMU_Lecture.pdf) for alignment
3. [Multimodal Agents Berkeley](slides/Multimodal_Agents_Berkeley.pdf) for web agents

### Advanced
1. [LeanDojo slides](slides/LeanDojo_NeurIPS_Slides.pdf) for theorem proving
2. [HippoRAG NeurIPS](slides/HippoRAG_NeurIPS_Slides.pdf) for memory systems
3. [Prompt Injection Duke](slides/Prompt_Injection_Duke_Slides.pdf) for security

### Reasoning & RL Fine-Tuning Path
1. [DeepSeek-R1 paper](DeepSeek-R1_Reasoning_via_RL.pdf) + [DeepSeek R1 CMU slides](slides/DeepSeek_R1_CMU_Reasoning.pdf)
2. [DeepSeekMath GRPO](DeepSeekMath_GRPO.pdf) + [Stanford RL for Reasoning](slides/Stanford_RL_for_LLM_Reasoning.pdf)
3. [ARTIST paper](ARTIST_Agentic_Reasoning_Tool_Integration.pdf) for agentic reasoning with tools

---

## License

Papers are property of their respective authors. This collection is for educational purposes.
