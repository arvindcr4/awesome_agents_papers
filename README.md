# Awesome Agents Papers Collection

A comprehensive collection of papers and presentation slides on LLM agents, reasoning, and AI systems.

> Sources:
> - [arvindcr4/awesome-agents](https://github.com/arvindcr4/awesome-agents)
> - [redhat-et/agentic-reasoning-reinforcement-fine-tuning](https://github.com/redhat-et/agentic-reasoning-reinforcement-fine-tuning)

## Quick Stats
- **Papers:** 88 PDFs (organized in 12 folders)
- **Slides:** 93 presentation decks (~504 MB)
- **Topics:** 15 categories
- **Audio Overviews:** See [NOTEBOOKLM_LINKS.md](NOTEBOOKLM_LINKS.md) for AI-generated podcast summaries
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

| Paper | Slides | Code | Media |
|-------|--------|------|-------|
| [Large Language Models as Optimizers](reinforcement_learning_papers/17_reasoning_and_search/Large_Language_Models_as_Optimizers.pdf) | [CS839 Prompting II](slides/CS839_Prompting_II_OPRO.pdf) | [GitHub](https://github.com/google-deepmind/opro) | [🖼️](media/planning/Large_Language_Models_as_Optimizers/README.md) |
| [Large Language Models Cannot Self-Correct Reasoning Yet](reinforcement_learning_papers/17_reasoning_and_search/Large_Language_Models_Cannot_Self-Correct_Reasoning_Yet.pdf) | - | - | [🎨](diagrams/downloaded_images/reasoning_03_LLMs_Cannot_Self_Correct.png) [🖼️](media/reasoning/Large_Language_Models_Cannot_Self-Correct_Reasoning_Yet/README.md) |
| [Teaching Large Language Models to Self-Debug](reinforcement_learning_papers/09_agentic_rl/Teaching_Large_Language_Models_to_Self-Debug.pdf) | - | - | [🎨](diagrams/downloaded_images/agent-frameworks_DSPy-_Compiling_Declarative_Language_Model.png) [🖼️](media/agent-frameworks/Teaching_Large_Language_Models_to_Self-Debug/README.md) [🎧](https://notebooklm.cloud.google.com/global/notebook/9368678b-0ef3-4ef6-a52b-250073dfd750?project=358208640342) |
| [Chain-of-Thought Reasoning Without Prompting](reinforcement_learning_papers/17_reasoning_and_search/Chain-of-Thought_Reasoning_Without_Prompting.pdf) | [CoT Princeton Lecture](slides/CoT_Princeton_Lecture.pdf), [CoT Toronto](slides/CoT_Toronto_Presentation.pdf), [CoT SJTU](slides/CoT_SJTU_Slides.pdf), [CoT Interpretable ML](slides/CoT_Interpretable_ML_Lecture.pdf), [Concise CoT](slides/Concise_CoT_Benefits.pdf) | [GitHub (unofficial)](https://github.com/fangyuan-ksgk/CoT-Reasoning-without-Prompting) | [🎨](diagrams/downloaded_images/reasoning_01_Chain_of_Thought_Reasoning_Without_Prompting.png) [🖼️](media/reasoning/Chain-of-Thought_Reasoning_Without_Prompting/README.md) |
| [Premise Order Matters in Reasoning with LLMs](reinforcement_learning_papers/17_reasoning_and_search/Premise_Order_Matters_in_Reasoning_with_Large_Language_Models.pdf) | - | - | [🎨](diagrams/downloaded_images/reasoning_04_Premise_Order_Matters.png) [🖼️](media/reasoning/Premise_Order_Matters_in_Reasoning_with_Large_Language_Models/README.md) |
| [Chain-of-Thought Empowers Transformers](reinforcement_learning_papers/17_reasoning_and_search/Chain-of-Thought_Empowers_Transformers_to_Solve_Inherently_Serial_Problems.pdf) | [CoT Slides](slides/CoT_SJTU_Slides.pdf) | - | [🎨](diagrams/downloaded_images/reasoning_02_Chain_of_Thought_Empowers_Transformers.png) [🖼️](media/reasoning/Chain-of-Thought_Empowers_Transformers_to_Solve_Inherently_Serial_Problems/README.md) |

## Post-Training & Alignment

| Paper | Slides | Code | Media |
|-------|--------|------|-------|
| [Direct Preference Optimization (DPO)](reinforcement_learning_papers/02_rlhf_alignment/Direct_Preference_Optimization.pdf) | [DPO CMU](slides/DPO_CMU_Lecture.pdf), [DPO UT Austin](slides/DPO_UT_Austin.pdf), [DPO Toronto](slides/DPO_Toronto_Presentation.pdf), [DPO Jinen](slides/DPO_Jinen_Slides.pdf) | [GitHub](https://github.com/eric-mitchell/direct-preference-optimization) | [🎨](diagrams/downloaded_images/reasoning_07_Iterative_Reasoning_Preference_Optimization.png) [🖼️](media/rl-finetuning/Direct_Preference_Optimization/README.md) |
| [Iterative Reasoning Preference Optimization](reinforcement_learning_papers/09_agentic_rl/Iterative_Reasoning_Preference_Optimization.pdf) | - | - | [🎨](diagrams/downloaded_images/reasoning_07_Iterative_Reasoning_Preference_Optimization.png) [🖼️](media/reasoning/Iterative_Reasoning_Preference_Optimization/README.md) |
| [Chain-of-Verification Reduces Hallucination](reinforcement_learning_papers/17_reasoning_and_search/Chain-of-Verification_Reduces_Hallucination.pdf) | - | [GitHub (unofficial)](https://github.com/hwchase17/chain-of-verification) | [🎨](diagrams/downloaded_images/reasoning_05_Chain_of_Verification.png) [🖼️](media/reasoning/Chain-of-Verification_Reduces_Hallucination/README.md) |
| [Unpacking DPO and PPO](reinforcement_learning_papers/02_rlhf_alignment/Unpacking_DPO_and_PPO.pdf) | [DPO Slides](slides/DPO_CMU_Lecture.pdf) | [GitHub](https://github.com/allenai/open-instruct) | [🖼️](media/rl-finetuning/Unpacking_DPO_and_PPO/README.md) |
| **RLHF Background** | [RLHF UT Austin](slides/RLHF_UT_Austin_Slides.pdf) | - | - |

## Memory & Planning

| Paper | Slides | Code | Media |
|-------|--------|------|-------|
| [Grokked Transformers are Implicit Reasoners](reinforcement_learning_papers/17_reasoning_and_search/Grokked_Transformers_are_Implicit_Reasoners.pdf) | - | [GitHub](https://github.com/OSU-NLP-Group/GrokkedTransformer) | [🎨](diagrams/downloaded_images/reasoning_06_Grokked_Transformers.png) [🖼️](media/reasoning/Grokked_Transformers_are_Implicit_Reasoners/README.md) |
| [HippoRAG: Neurobiologically Inspired Long-Term Memory](reinforcement_learning_papers/09_agentic_rl/HippoRAG_Neurobiologically_Inspired_Long-Term_Memory.pdf) | [HippoRAG NeurIPS](slides/HippoRAG_NeurIPS_Slides.pdf) | [GitHub](https://github.com/OSU-NLP-Group/HippoRAG) | [🎨](diagrams/downloaded_images/memory-rag_HippoRAG.png) [🖼️](media/memory-rag/HippoRAG_Neurobiologically_Inspired_Long-Term_Memory/README.md) |
| [Is Your LLM Secretly a World Model of the Internet](reinforcement_learning_papers/09_agentic_rl/Is_Your_LLM_Secretly_a_World_Model_of_the_Internet.pdf) | - | [GitHub](https://github.com/OSU-NLP-Group/WebDreamer) | [🖼️](media/memory-rag/Is_Your_LLM_Secretly_a_World_Model_of_the_Internet/README.md) |
| [Tree Search for Language Model Agents](reinforcement_learning_papers/09_agentic_rl/Tree_Search_for_Language_Model_Agents.pdf) | - | [GitHub](https://github.com/kohjingyu/search-agents) | [🖼️](media/planning/Tree_Search_for_Language_Model_Agents/README.md) |

## Agent Frameworks

| Paper | Slides | Code | Media |
|-------|--------|------|-------|
| [ReAct: Synergizing Reasoning and Acting](reinforcement_learning_papers/09_agentic_rl/ReAct_Synergizing_Reasoning_and_Acting.pdf) | [ReAct UVA Lecture](slides/ReAct_UVA_Lecture.pdf) | [GitHub](https://github.com/ysymyth/ReAct) | [🎨](diagrams/downloaded_images/agent-frameworks_ReAct-_Synergizing_Reasoning_and_Acting.png) [🖼️](media/agent-frameworks/ReAct_Synergizing_Reasoning_and_Acting/README.md) [🎧](https://notebooklm.cloud.google.com/global/notebook/fe6430c3-2fee-4244-ace3-1db109cc2c0c?project=358208640342) |
| [AutoGen: Multi-Agent Conversation](reinforcement_learning_papers/09_agentic_rl/AutoGen_Multi-Agent_Conversation.pdf) | - | [GitHub](https://github.com/microsoft/autogen) | [🎨](diagrams/downloaded_images/agent-frameworks_AutoGen-_Multi-Agent_Conversation.png) [🖼️](media/agent-frameworks/AutoGen_Multi-Agent_Conversation/README.md) [🎧](https://notebooklm.cloud.google.com/global/notebook/bd26a628-e81f-4dc0-9371-70d9b54dcdca?project=358208640342) |
| [StateFlow: Enhancing LLM Task-Solving](reinforcement_learning_papers/09_agentic_rl/StateFlow_Enhancing_LLM_Task-Solving.pdf) | - | [GitHub](https://github.com/yiranwu0/StateFlow) | [🖼️](media/agent-frameworks/StateFlow_Enhancing_LLM_Task-Solving/README.md) [🎧](https://notebooklm.cloud.google.com/global/notebook/918935e4-1b33-4a47-be5f-ac7c115cd09f?project=358208640342) |
| [DSPy: Compiling Declarative Language Model](reinforcement_learning_papers/19_benchmarks_and_evaluations/DSPy_Compiling_Declarative_Language_Model.pdf) | - | [GitHub](https://github.com/stanfordnlp/dspy) | [🎨](diagrams/downloaded_images/agent-frameworks_DSPy-_Compiling_Declarative_Language_Model.png) [🖼️](media/agent-frameworks/DSPy_Compiling_Declarative_Language_Model/README.md) [🎧](https://notebooklm.cloud.google.com/global/notebook/7c68d545-0d36-402e-97f5-44e730389da2?project=358208640342) |
| **LLM Agents Tutorials** | [EMNLP 2024 Tutorial](slides/EMNLP2024_Language_Agents_Tutorial.pdf), [WWW 2024 Tutorial](slides/WWW2024_LLM_Agents_Tutorial.pdf), [Berkeley Training Agents](slides/Berkeley_LLM_Training_Agents.pdf) | - | - |

## Code Generation & Software Agents

| Paper | Slides | Code | Media |
|-------|--------|------|-------|
| [SWE-agent: Agent-Computer Interfaces](reinforcement_learning_papers/19_benchmarks_and_evaluations/SWE-agent_Agent-Computer_Interfaces.pdf) | [Software Agents (Neubig)](slides/Software_Agents_Neubig.pdf) | [GitHub](https://github.com/SWE-agent/SWE-agent) | [🎨](diagrams/downloaded_images/agent-frameworks_AutoGen-_Multi-Agent_Conversation.png) [🖼️](media/computer-use/SWE-agent_Agent-Computer_Interfaces/README.md) |
| [OpenHands: AI Software Developers](reinforcement_learning_papers/09_agentic_rl/OpenHands_AI_Software_Developers.pdf) | [Software Agents (Neubig)](slides/Software_Agents_Neubig.pdf) | [GitHub](https://github.com/OpenHands/OpenHands) | [🖼️](media/agent-frameworks/OpenHands_AI_Software_Developers/README.md) [🎧](https://notebooklm.cloud.google.com/global/notebook/ed315459-c9ed-478d-975c-d17dd7f357ac?project=358208640342) |
| [Interactive Tools Assist LM Agents Security Vulnerabilities](reinforcement_learning_papers/18_llm_security_and_redteaming/Interactive_Tools_Assist_LM_Agents_Security_Vulnerabilities.pdf) | [Code Agents & Vulnerability Detection](slides/Code_Agents_Vulnerability_Detection_Berkeley.pdf) | [GitHub](https://github.com/NYU-LLM-CTF/NYU_CTF_) | - |
| [Big Sleep: LLM Vulnerabilities Real-World](reinforcement_learning_papers/18_llm_security_and_redteaming/Big_Sleep_LLM_Vulnerabilities_Real-World.pdf) | [Code Agents & Vulnerability Detection](slides/Code_Agents_Vulnerability_Detection_Berkeley.pdf) | - | - |
| [SWE-bench Verified](reinforcement_learning_papers/19_benchmarks_and_evaluations/SWE-bench_Verified.pdf) | - | [GitHub](https://github.com/SWE-bench/SWE-bench) | [🖼️](media/benchmarks/SWE-bench_Verified/README.md) [🎧](https://notebooklm.cloud.google.com/global/notebook/a23fbadb-1949-4a19-b9b1-77d65d8478d4?project=358208640342) |

## Web & Multimodal Agents

| Paper | Slides | Code | Media |
|-------|--------|------|-------|
| [WebShop: Scalable Real-World Web Interaction](reinforcement_learning_papers/09_agentic_rl/WebShop_Scalable_Real-World_Web_Interaction.pdf) | [Multimodal Agents Berkeley](slides/Multimodal_Agents_Berkeley.pdf) | [GitHub](https://github.com/princeton-nlp/WebShop) | - |
| [Mind2Web: Generalist Agent for the Web](reinforcement_learning_papers/09_agentic_rl/Mind2Web_Generalist_Agent_for_the_Web.pdf) | [Multimodal Agents Berkeley](slides/Multimodal_Agents_Berkeley.pdf) | [GitHub](https://github.com/OSU-NLP-Group/Mind2Web) | - |
| [WebArena: Realistic Web Environment](reinforcement_learning_papers/19_benchmarks_and_evaluations/WebArena_Realistic_Web_Environment.pdf) | [Multimodal Agents Berkeley](slides/Multimodal_Agents_Berkeley.pdf), [Web Agent Evaluation](slides/Web_Agent_Evaluation_Refinement.pdf) | [GitHub](https://github.com/web-arena-x/webarena) | - |
| [VisualWebArena](reinforcement_learning_papers/19_benchmarks_and_evaluations/VisualWebArena.pdf) | [Multimodal Agents Berkeley](slides/Multimodal_Agents_Berkeley.pdf) | [GitHub](https://github.com/web-arena-x/visualwebarena) | - |
| [AGUVIS: Unified Pure Vision Agents GUI](reinforcement_learning_papers/09_agentic_rl/AGUVIS_Unified_Pure_Vision_Agents_GUI.pdf) | - | [GitHub](https://github.com/xlang-ai/aguvis) | - |
| [BrowseComp: Web Browsing Benchmark](reinforcement_learning_papers/19_benchmarks_and_evaluations/BrowseComp_Web_Browsing_Benchmark.pdf) | - | [GitHub](https://github.com/openai/simple-evals) | - |

## Enterprise & Workflow Agents

| Paper | Slides | Code | Media |
|-------|--------|------|-------|
| [WorkArena: Common Knowledge Work Tasks](reinforcement_learning_papers/19_benchmarks_and_evaluations/WorkArena_Common_Knowledge_Work_Tasks.pdf) | - | [GitHub](https://github.com/ServiceNow/WorkArena) | [🖼️](media/benchmarks/WorkArena_Common_Knowledge_Work_Tasks/README.md) [🎧](https://notebooklm.cloud.google.com/global/notebook/e719ad68-4c9a-4587-b792-c35c10e8b405?project=358208640342) |
| [WorkArena++: Compositional Planning](reinforcement_learning_papers/19_benchmarks_and_evaluations/WorkArena_Compositional_Planning.pdf) | - | [GitHub](https://github.com/ServiceNow/WorkArena) | [🖼️](media/benchmarks/WorkArena_Compositional_Planning/README.md) [🎧](https://notebooklm.cloud.google.com/global/notebook/53f4fdb1-7eb9-4ecf-a0be-11bfbdcd4e29?project=358208640342) |
| [TapeAgents: Holistic Framework Agent Development](reinforcement_learning_papers/09_agentic_rl/TapeAgents_Holistic_Framework_Agent_Development.pdf) | [TapeAgents Slides](slides/TapeAgents_ServiceNow.pdf) | [GitHub](https://github.com/ServiceNow/TapeAgents) | [🖼️](media/agent-frameworks/TapeAgents_Holistic_Framework_Agent_Development/README.md) [🎧](https://notebooklm.cloud.google.com/global/notebook/8cd2f3fb-cb23-44a8-863d-3242c823aa04?project=358208640342) |

## Mathematics & Theorem Proving

| Paper | Slides | Code | Media |
|-------|--------|------|-------|
| [LeanDojo: Theorem Proving Retrieval-Augmented](reinforcement_learning_papers/15_embodied_and_robotics_rl/LeanDojo_Theorem_Proving_Retrieval-Augmented.pdf) | [LeanDojo AITP](slides/LeanDojo_AITP_Slides.pdf), [LeanDojo NeurIPS](slides/LeanDojo_NeurIPS_Slides.pdf), [Theorem Proving ML](slides/Theorem_Proving_ML_Slides.pdf) | [GitHub](https://github.com/lean-dojo/LeanDojo) | [🎨](diagrams/downloaded_images/theorem-proving_leandojo-_retrieval-augmented.png) |
| [Autoformalization with Large Language Models](reinforcement_learning_papers/15_embodied_and_robotics_rl/Autoformalization_with_Large_Language_Models.pdf) | - | - | [🎨](diagrams/downloaded_images/agent-frameworks_DSPy-_Compiling_Declarative_Language_Model.png) |
| [Autoformalizing Euclidean Geometry](reinforcement_learning_papers/15_embodied_and_robotics_rl/Autoformalizing_Euclidean_Geometry.pdf) | - | [GitHub](https://github.com/loganrjmurphy/LeanEuclid) | [🎨](diagrams/downloaded_images/theorem-proving_autoformalizing_euclidean_geometry.png) |
| [Draft, Sketch and Prove: Formal Theorem Provers](reinforcement_learning_papers/15_embodied_and_robotics_rl/Draft_Sketch_and_Prove_Formal_Theorem_Provers.pdf) | [Theorem Proving ML](slides/Theorem_Proving_ML_Slides.pdf) | [GitHub](https://github.com/albertqjiang/draft_sketch_prove) | [🎨](diagrams/downloaded_images/theorem-proving_draft_sketch_prove.png) |
| [miniCTX: Neural Theorem Proving Long-Contexts](reinforcement_learning_papers/15_embodied_and_robotics_rl/miniCTX_Neural_Theorem_Proving_Long-Contexts.pdf) | - | [GitHub](https://github.com/cmu-l3/minictx-eval) | [🎨](diagrams/downloaded_images/theorem-proving_minictx-_long-context.png) |
| [Lean-STaR: Interleave Thinking and Proving](reinforcement_learning_papers/15_embodied_and_robotics_rl/Lean-STaR_Interleave_Thinking_and_Proving.pdf) | [Berkeley Slides](slides/Welleck_Berkeley_Bridging_Informal_Formal_Math.pdf) | [GitHub](https://github.com/Lagooon/LeanSTaR) [Website](https://leanstar.github.io/) | [🎨](diagrams/downloaded_images/agent-frameworks_ReAct-_Synergizing_Reasoning_and_Acting.png) |
| [ImProver: Agent-Based Automated Proof Optimization](reinforcement_learning_papers/15_embodied_and_robotics_rl/ImProver_Agent-Based_Automated_Proof_Optimization.pdf) | - | [GitHub](https://github.com/riyazahuja/ImProver) | [🎨](diagrams/downloaded_images/reasoning_07_Iterative_Reasoning_Preference_Optimization.png) |
| [In-Context Learning Agent Formal Theorem-Proving](reinforcement_learning_papers/15_embodied_and_robotics_rl/In-Context_Learning_Agent_Formal_Theorem-Proving.pdf) | - | [GitHub](https://github.com/trishullab/copra) | - |
| [Symbolic Regression: Learned Concept Library](reinforcement_learning_papers/17_reasoning_and_search/Symbolic_Regression_Learned_Concept_Library.pdf) | - | [GitHub](https://github.com/trishullab/LibraryAugmentedSymbolicRegression.jl) | [🖼️](media/planning/Symbolic_Regression_Learned_Concept_Library/README.md) |
| [AlphaGeometry: Solving Olympiad Geometry](reinforcement_learning_papers/15_embodied_and_robotics_rl/AlphaGeometry_Solving_Olympiad_Geometry.pdf) | - | [GitHub](https://github.com/google-deepmind/alphageometry) | - |

## Robotics & Embodied Agents

| Paper | Slides | Code | Media |
|-------|--------|------|-------|
| [Voyager: Open-Ended Embodied Agent](reinforcement_learning_papers/15_embodied_and_robotics_rl/Voyager_Open-Ended_Embodied_Agent.pdf) | [Voyager UT Austin](slides/Voyager_UT_Austin_Presentation.pdf) | [GitHub](https://github.com/MineDojo/Voyager) | [🎨](diagrams/downloaded_images/robotics_Voyager-_Open-Ended_Embodied_Agent.png) |
| [Eureka: Human-Level Reward Design](reinforcement_learning_papers/15_embodied_and_robotics_rl/Eureka_Human-Level_Reward_Design.pdf) | [Eureka Paper/Slides](slides/Eureka_Reward_Design_Paper.pdf) | [GitHub](https://github.com/eureka-research/Eureka) | [🎨](diagrams/downloaded_images/robotics_Eureka-_Human-Level_Reward_Design.png) [🖼️](media/robotics/Eureka_Human-Level_Reward_Design/README.md) |
| [DrEureka: Language Model Guided Sim-To-Real](reinforcement_learning_papers/15_embodied_and_robotics_rl/DrEureka_Language_Model_Guided_Sim-To-Real.pdf) | - | [GitHub](https://github.com/eureka-research/DrEureka) | [🎨](diagrams/downloaded_images/robotics_DrEureka-_Sim-to-Real.png) [🖼️](media/robotics/DrEureka_Language_Model_Guided_Sim-To-Real/README.md) |
| [Gran Turismo: Deep Reinforcement Learning](reinforcement_learning_papers/15_embodied_and_robotics_rl/Gran_Turismo_Deep_Reinforcement_Learning.pdf) | - | - | [🖼️](media/robotics/Gran_Turismo_Deep_Reinforcement_Learning/README.md) |
| [GR00T N1: Foundation Model Humanoid](reinforcement_learning_papers/15_embodied_and_robotics_rl/GR00T_N1_Foundation_Model_Humanoid.pdf) | - | [GitHub](https://github.com/NVIDIA/Isaac-GR00T) | [🎨](diagrams/downloaded_images/robotics_GR00T_N1-_Humanoid_Foundation_Model.png) [🖼️](media/robotics/GR00T_N1_Foundation_Model_Humanoid/README.md) |
| [SLAC: Simulation-Pretrained Latent Action](reinforcement_learning_papers/15_embodied_and_robotics_rl/SLAC_Simulation-Pretrained_Latent_Action.pdf) | - | - | - |

## Scientific Discovery

| Paper | Slides | Code | Media |
|-------|--------|------|-------|
| [Paper2Agent: Research Papers as AI Agents](reinforcement_learning_papers/09_agentic_rl/Paper2Agent_Research_Papers_as_AI_Agents.pdf) | - | [GitHub](https://github.com/jmiao24/Paper2Agent) | [🖼️](media/agent-frameworks/Paper2Agent_Research_Papers_as_AI_Agents/README.md) [🎧](https://notebooklm.cloud.google.com/global/notebook/9db0bff2-5fa0-4eeb-87d4-477d420ef5c9?project=358208640342) |
| [OpenScholar: Synthesizing Scientific Literature](reinforcement_learning_papers/09_agentic_rl/OpenScholar_Synthesizing_Scientific_Literature.pdf) | - | [GitHub](https://github.com/AkariAsai/OpenScholar) | [🖼️](media/memory-rag/OpenScholar_Synthesizing_Scientific_Literature/README.md) |

## Safety & Security

| Paper | Slides | Code | Media |
|-------|--------|------|-------|
| [DataSentinel: Game-Theoretic Detection Prompt Injection](reinforcement_learning_papers/18_llm_security_and_redteaming/DataSentinel_Game-Theoretic_Detection_Prompt_Injection.pdf) | [Prompt Injection Duke](slides/Prompt_Injection_Duke_Slides.pdf) | [GitHub](https://github.com/liu00222/Open-Prompt-Injection) | - |
| [AgentPoison: Red-teaming LLM Agents](reinforcement_learning_papers/18_llm_security_and_redteaming/AgentPoison_Red-teaming_LLM_Agents.pdf) | [Prompt Injection Duke](slides/Prompt_Injection_Duke_Slides.pdf) | [GitHub](https://github.com/AI-secure/AgentPoison) | [🎨](diagrams/downloaded_images/robotics_Voyager-_Open-Ended_Embodied_Agent.png) |
| [Progent: Programmable Privilege Control](reinforcement_learning_papers/18_llm_security_and_redteaming/Progent_Programmable_Privilege_Control.pdf) | - | - | - |
| [DecodingTrust: Trustworthiness GPT Models](reinforcement_learning_papers/19_benchmarks_and_evaluations/DecodingTrust_Trustworthiness_GPT_Models.pdf) | - | [GitHub](https://github.com/AI-secure/DecodingTrust) | - |
| [Representation Engineering: AI Transparency](reinforcement_learning_papers/18_llm_security_and_redteaming/Representation_Engineering_AI_Transparency.pdf) | - | [GitHub](https://github.com/andyzoujm/representation-engineering) | - |
| [Extracting Training Data from LLMs](reinforcement_learning_papers/18_llm_security_and_redteaming/Extracting_Training_Data_from_LLMs.pdf) | - | - | - |
| [The Secret Sharer: Unintended Memorization](reinforcement_learning_papers/18_llm_security_and_redteaming/The_Secret_Sharer_Unintended_Memorization.pdf) | - | - | - |
| [Privtrans: Privilege Separation](reinforcement_learning_papers/18_llm_security_and_redteaming/Privtrans_Privilege_Separation.pdf) | - | - | - |

## Evaluation & Benchmarking

| Paper | Slides | Code | Media |
|-------|--------|------|-------|
| [Survey: Evaluation LLM-based Agents](reinforcement_learning_papers/19_benchmarks_and_evaluations/Survey_Evaluation_LLM-based_Agents.pdf) | [AgentBench Multi-Turn NeurIPS](slides/AgentBench_Multi_Turn_NeurIPS.pdf) | - | [🖼️](media/benchmarks/Survey_Evaluation_LLM-based_Agents/README.md) [🎧](https://notebooklm.cloud.google.com/global/notebook/728fb1d8-6903-4ac3-bd97-eec90cc67182?project=358208640342) |
| [Adding Error Bars to Evals](reinforcement_learning_papers/19_benchmarks_and_evaluations/Adding_Error_Bars_to_Evals.pdf) | - | [GitHub](https://github.com/openai/evals) | [🖼️](media/benchmarks/Adding_Error_Bars_to_Evals/README.md) [🎧](https://notebooklm.cloud.google.com/global/notebook/2df09327-6444-4c47-995c-4d5389b15fe9?project=358208640342) |
| [Tau2-Bench: Conversational Agents Dual-Control](reinforcement_learning_papers/19_benchmarks_and_evaluations/Tau2-Bench_Conversational_Agents_Dual-Control.pdf) | - | [GitHub](https://github.com/sierra-research/tau2-bench) | [🖼️](media/benchmarks/Tau2-Bench_Conversational_Agents_Dual-Control/README.md) [🎧](https://notebooklm.cloud.google.com/global/notebook/f5a83e17-9a24-4ad6-a32f-e8921ee4c73d?project=358208640342) |
| **Data Science Agents** | [Data Science Agents Benchmark](slides/Data_Science_Agents_Benchmark.pdf) | - | - |

## Neural & Symbolic Reasoning

| Paper | Slides | Code | Media |
|-------|--------|------|-------|
| [Beyond A-Star: Better Planning Transformers](reinforcement_learning_papers/17_reasoning_and_search/Beyond_A-Star_Better_Planning_Transformers.pdf) | - | [GitHub](https://github.com/facebookresearch/searchformer) | [🖼️](media/reasoning/Beyond_A-Star_Better_Planning_Transformers/README.md) |
| [Dualformer: Controllable Fast and Slow Thinking](reinforcement_learning_papers/09_agentic_rl/Dualformer_Controllable_Fast_and_Slow_Thinking.pdf) | - | [GitHub](https://github.com/facebookresearch/dualformer) | [🖼️](media/reasoning/Dualformer_Controllable_Fast_and_Slow_Thinking/README.md) |
| [Composing Global Optimizers: Algebraic Objects](reinforcement_learning_papers/17_reasoning_and_search/Composing_Global_Optimizers_Algebraic_Objects.pdf) | - | - | [🖼️](media/planning/Composing_Global_Optimizers_Algebraic_Objects/README.md) |
| [SurCo: Learning Linear Surrogates](reinforcement_learning_papers/09_agentic_rl/SurCo_Learning_Linear_Surrogates.pdf) | - | - | [🖼️](media/planning/SurCo_Learning_Linear_Surrogates/README.md) |

## Agentic Reasoning & RL Fine-Tuning

> Source: [redhat-et/agentic-reasoning-reinforcement-fine-tuning](https://github.com/redhat-et/agentic-reasoning-reinforcement-fine-tuning)

### DeepSeek R1 & Reasoning Models

| Paper | Slides | Code | Media |
|-------|--------|------|-------|
| [DeepSeek-R1: Reasoning via RL](reinforcement_learning_papers/02_rlhf_alignment/DeepSeek-R1_Reasoning_via_RL.pdf) | [DeepSeek R1 Intro](slides/DeepSeek_R1_Introduction.pdf), [DeepSeek R1 Toronto](slides/DeepSeek_R1_Toronto.pdf), [DeepSeek R1 CMU](slides/DeepSeek_R1_CMU_Reasoning.pdf), [DeepSeek R1 Seoul](slides/DeepSeek_R1_Seoul_National.pdf) | [GitHub](https://github.com/deepseek-ai/DeepSeek-R1) | [🎨](diagrams/downloaded_images/rl-finetuning_DeepSeek-R1-_Reasoning_via_RL.png) [🖼️](media/rl-finetuning/DeepSeek-R1_Reasoning_via_RL/README.md) |
| [DeepSeek R1: Implications for AI](reinforcement_learning_papers/02_rlhf_alignment/DeepSeek_R1_Implications_for_AI.pdf) | [DeepSeek R1 Intro](slides/DeepSeek_R1_Introduction.pdf) | - | [🎨](diagrams/downloaded_images/rl-finetuning_DeepSeek_R1-_Implications_for_AI.png) [🖼️](media/rl-finetuning/DeepSeek_R1_Implications_for_AI/README.md) |
| [DeepSeek R1: Are Reasoning Models Faithful?](reinforcement_learning_papers/02_rlhf_alignment/DeepSeek_R1_Reasoning_Models_Faithful.pdf) | - | - | [🎨](diagrams/downloaded_images/rl-finetuning_DeepSeek-R1-_Reasoning_via_RL.png) [🖼️](media/rl-finetuning/DeepSeek_R1_Reasoning_Models_Faithful/README.md) |
| [OpenAI O1 Replication Journey](reinforcement_learning_papers/02_rlhf_alignment/OpenAI_O1_Replication_Journey.pdf) | - | [GitHub](https://github.com/GAIR-NLP/O1-Journey) | [🎨](diagrams/downloaded_images/rl-finetuning_DeepSeek_R1-_Implications_for_AI.png) [🖼️](media/rl-finetuning/OpenAI_O1_Replication_Journey/README.md) |
| [Qwen QwQ Reasoning Model](reinforcement_learning_papers/02_rlhf_alignment/Qwen_QwQ_Reasoning_Model.pdf) | - | [HuggingFace](https://huggingface.co/Qwen/QwQ-32B-Preview) | [🎨](diagrams/downloaded_images/rl-finetuning_DeepSeek-R1-_Reasoning_via_RL.png) [🖼️](media/rl-finetuning/Qwen_QwQ_Reasoning_Model/README.md) |
| [Sky-T1: Training Small Reasoning LLMs](reinforcement_learning_papers/02_rlhf_alignment/Sky-T1_Training_Small_Reasoning_LLMs.pdf) | - | [GitHub](https://github.com/NovaSky-AI/SkyThought) | [🖼️](media/rl-finetuning/Sky-T1_Training_Small_Reasoning_LLMs/README.md) |
| [s1: Simple Test-Time Scaling](reinforcement_learning_papers/02_rlhf_alignment/s1_Simple_Test-Time_Scaling.pdf) | - | [GitHub](https://github.com/simplescaling/s1) | [🖼️](media/rl-finetuning/s1_Simple_Test-Time_Scaling/README.md) |

### GRPO & RL Fine-Tuning

| Paper | Slides | Code | Media |
|-------|--------|------|-------|
| [DeepSeekMath: GRPO Algorithm](reinforcement_learning_papers/02_rlhf_alignment/DeepSeekMath_GRPO.pdf) | [Stanford RL for Reasoning](slides/Stanford_RL_for_LLM_Reasoning.pdf) | [GitHub](https://github.com/deepseek-ai/DeepSeek-Math) | [🎨](diagrams/downloaded_images/rl-finetuning_DeepSeek_R1-_Implications_for_AI.png) [🖼️](media/rl-finetuning/DeepSeekMath_GRPO/README.md) |
| [Guided GRPO: Adaptive Guidance](reinforcement_learning_papers/02_rlhf_alignment/Guided_GRPO_Adaptive_Guidance.pdf) | [PTA-GRPO Planning](slides/PTA_GRPO_Planning_Reasoning.pdf) | [GitHub](https://github.com/T-Lab-CUHKSZ/G2RPO-A) | [🖼️](media/rl-finetuning/Guided_GRPO_Adaptive_Guidance/README.md) |
| [R-Search: Multi-Step Reasoning](reinforcement_learning_papers/02_rlhf_alignment/R-Search_Multi-Step_Reasoning.pdf) | [Stanford RL for Reasoning](slides/Stanford_RL_for_LLM_Reasoning.pdf) | [GitHub](https://github.com/wentao0429/Reasoning-search) | [🖼️](media/rl-finetuning/R-Search_Multi-Step_Reasoning/README.md) |
| [RL Fine-tuning: Instruction Following](reinforcement_learning_papers/02_rlhf_alignment/RL_Fine-tuning_Instruction_Following.pdf) | - | - | [🖼️](media/rl-finetuning/RL_Fine-tuning_Instruction_Following/README.md) |
| [RFT Powers Multimodal Reasoning](reinforcement_learning_papers/02_rlhf_alignment/RFT_Powers_Multimodal_Reasoning.pdf) | - | - | [🖼️](media/rl-finetuning/RFT_Powers_Multimodal_Reasoning/README.md) |
| [STILL-2: Distilling Reasoning](reinforcement_learning_papers/02_rlhf_alignment/STILL-2_Distilling_Reasoning.pdf) | - | - | [🖼️](media/rl-finetuning/STILL-2_Distilling_Reasoning/README.md) |

### Agentic RL

| Paper | Slides | Code | Media |
|-------|--------|------|-------|
| [WebAgent-R1: Multi-Turn RL for Web Agents](reinforcement_learning_papers/09_agentic_rl/WebAgent-R1_Multi-Turn_RL.pdf) | - | [GitHub](https://github.com/weizhepei/WebAgent-R1) | - |
| [ARTIST: Agentic Reasoning & Tool Integration](reinforcement_learning_papers/09_agentic_rl/ARTIST_Agentic_Reasoning_Tool_Integration.pdf) | [ARTIST Microsoft](slides/ARTIST_Agentic_Reasoning_Microsoft.pdf) | [GitHub](https://github.com/microsoft/ARTIST) | [🎨](diagrams/downloaded_images/agent-frameworks_ReAct-_Synergizing_Reasoning_and_Acting.png) [🖼️](media/agent-frameworks/ARTIST_Agentic_Reasoning_Tool_Integration/README.md) [🎧](https://notebooklm.cloud.google.com/global/notebook/dd5aab2c-a230-4161-b521-81c63cbf2902?project=358208640342) |

## Agentic Architectures & Coordination

> Papers on multi-agent systems, decentralized coordination, and agentic frameworks

### Decentralized Multi-Agent Systems

| Paper | Slides | Code | Media |
|-------|--------|------|-------|
| [AgentNet: Decentralized Multi-Agent Coordination](reinforcement_learning_papers/03_multi_agent_rl/AgentNet_Decentralized_Multi-Agent.pdf) | - | [GitHub](https://github.com/zoe-yyx/AgentNet) | [🎨](diagrams/downloaded_images/multi-agent_AgentNet.png) [🖼️](media/multi-agent/AgentNet_Decentralized_Multi-Agent/README.md) |
| [MasRouter: Multi-Agent Routing](reinforcement_learning_papers/03_multi_agent_rl/MasRouter_Multi-Agent_Routing.pdf) | [MasRouter ACL 2025](slides/MasRouter_ACL_2025.pdf) | [GitHub](https://github.com/THU-KEG/MasRouter) | [🎨](diagrams/downloaded_images/agent-frameworks_AutoGen-_Multi-Agent_Conversation.png) [🖼️](media/multi-agent/MasRouter_Multi-Agent_Routing/README.md) |
| **Multi-Agent RL Overview** | [Edinburgh MARL Intro](slides/Edinburgh_Multi_Agent_RL_Intro.pdf) | - | - |

### Device & Computer Control

| Paper | Slides | Code | Media |
|-------|--------|------|-------|
| [DigiRL: Device Control Agents](reinforcement_learning_papers/09_agentic_rl/DigiRL_Device_Control_Agents.pdf) | [DigiRL NeurIPS 2024](slides/DigiRL_NeurIPS_2024.pdf) | [GitHub](https://github.com/DigiRL-agent/digirl) | [🖼️](media/computer-use/DigiRL_Device_Control_Agents/README.md) [🎧](https://notebooklm.cloud.google.com/global/notebook/931f05c0-39d7-48d2-a51a-8d273bd3d6af?project=358208640342) |
| [OSWorld: Multimodal Agents Benchmark](reinforcement_learning_papers/19_benchmarks_and_evaluations/OSWorld_Multimodal_Agents_Benchmark.pdf) | - | [GitHub](https://github.com/xlang-ai/OSWorld) | [🖼️](media/computer-use/OSWorld_Multimodal_Agents_Benchmark/README.md) |
| [OS-Harm: Computer Use Safety](reinforcement_learning_papers/19_benchmarks_and_evaluations/OS-Harm_Computer_Use_Safety.pdf) | [OS-Harm Benchmark](slides/OS_Harm_Benchmark.pdf) | [GitHub](https://github.com/tml-epfl/os-harm) | [🖼️](media/computer-use/OS-Harm_Computer_Use_Safety/README.md) [🎧](https://notebooklm.cloud.google.com/global/notebook/09f307c6-9716-45a3-9c83-be2e99b1f8e6?project=358208640342) |

### Agent Fine-Tuning & Tool Use

| Paper | Slides | Code | Media |
|-------|--------|------|-------|
| [FireAct: Language Agent Fine-tuning](reinforcement_learning_papers/09_agentic_rl/FireAct_Language_Agent_Fine-tuning.pdf) | [LLM Agents Tool Learning](slides/LLM_Agents_Tool_Learning_Tutorial.pdf) | [GitHub](https://github.com/anchen1011/FireAct) | [🖼️](media/agent-frameworks/FireAct_Language_Agent_Fine-tuning/README.md) [🎧](https://notebooklm.cloud.google.com/global/notebook/79089fc8-bf2d-406d-985b-e06fd724c9fc?project=358208640342) |
| [DeepSeek Janus Pro: Multimodal](reinforcement_learning_papers/02_rlhf_alignment/DeepSeek_Janus_Pro_Multimodal.pdf) | - | [GitHub](https://github.com/deepseek-ai/Janus) | [🎨](diagrams/downloaded_images/rl-finetuning_DeepSeek_R1-_Implications_for_AI.png) [🖼️](media/rl-finetuning/DeepSeek_Janus_Pro_Multimodal/README.md) |
| [PTA-GRPO: High-Level Planning](slides/PTA_GRPO_High_Level_Planning.pdf) | [PTA-GRPO Planning](slides/PTA_GRPO_High_Level_Planning.pdf) | - | - |
| **Stanford RL for Agents** | [Stanford RL Agents 2025](slides/Stanford_RL_for_Agents_2025.pdf) | - | - |
| **CMU LM Agents** | [CMU Language Models as Agents](slides/CMU_Language_Models_as_Agents.pdf) | - | - |
| **Mannheim Tool Use** | [Mannheim LLM Agents Tool Use](slides/Mannheim_LLM_Agents_Tool_Use.pdf) | - | - |

### Enterprise & Industry Guides

| Resource | Description | Code |
|----------|-------------|------|
| [Intel AI Agents Architecture](slides/Intel_AI_Agents_Architecture.pdf) | AI agents resource guide | - |
| [Cisco Agentic Frameworks](slides/Cisco_Agentic_Frameworks_Overview.pdf) | Overview of agentic frameworks | - |

---

## Deep Reinforcement Learning

> **[See Full Deep RL Resources Guide](DEEP_RL_RESOURCES.md)** - Comprehensive collection with 100+ resources and 92 slides

### Value-Based Methods (DQN Family)

| Paper | arXiv | Slides | Code | Media |
|-------|-------|--------|------|-------|
| Playing Atari with Deep RL (DQN) | [1312.5602](https://arxiv.org/abs/1312.5602) | [CMU](slides/DQN_CMU_Deep_Q_Learning.pdf), [CVUT](slides/DQN_CVUT_Q_Learning.pdf), [NTHU](slides/DQN_NTHU_Deep_RL.pdf), [Waterloo](slides/DQN_Waterloo_CS885.pdf) | [OpenAI Baselines](https://github.com/openai/baselines) | - |
| Deep RL with Double Q-learning | [1509.06461](https://arxiv.org/abs/1509.06461) | [CMU DQN](slides/DQN_CMU_Deep_Q_Learning.pdf) | [OpenAI Baselines](https://github.com/openai/baselines) | - |
| Dueling Network Architectures | [1511.06581](https://arxiv.org/abs/1511.06581) | [Buffalo](slides/Dueling_DQN_PER_Buffalo.pdf) | [OpenAI Baselines](https://github.com/openai/baselines) | - |
| Prioritized Experience Replay | [1511.05952](https://arxiv.org/abs/1511.05952) | [Buffalo](slides/Dueling_DQN_PER_Buffalo.pdf), [Julien Vitay](slides/PER_Julien_Vitay.pdf), [ICML 2020](slides/Experience_Replay_ICML2020.pdf) | [OpenAI Baselines](https://github.com/openai/baselines) | - |
| Rainbow: Combining Improvements | [1710.02298](https://arxiv.org/abs/1710.02298) | [Prague](slides/Rainbow_Prague_NPFL122.pdf), [Berkeley](slides/Rainbow_Berkeley_Off_Policy.pdf), [Wisconsin](slides/Rainbow_Wisconsin_CS760.pdf) | [Dopamine](https://github.com/google/dopamine) | - |

### Policy Gradient Methods

| Paper | arXiv | Slides | Code | Media |
|-------|-------|--------|------|-------|
| Policy Gradient Methods | - | [Toronto](slides/Policy_Gradient_Toronto.pdf), [Berkeley CS285](slides/Policy_Gradient_Berkeley_CS285.pdf), [REINFORCE Stanford](slides/REINFORCE_Stanford_CS229.pdf) | [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) | - |
| Proximal Policy Optimization (PPO) | [1707.06347](https://arxiv.org/abs/1707.06347) | [Waterloo](slides/PPO_Waterloo_CS885.pdf), [NTU Taiwan](slides/PPO_NTU_Taiwan.pdf) | [OpenAI Baselines](https://github.com/openai/baselines) | - |
| Trust Region Policy Optimization (TRPO) | [1502.05477](https://arxiv.org/abs/1502.05477) | [FAU](slides/TRPO_FAU_Mutschler.pdf), [UT Austin](slides/TRPO_UT_Austin.pdf), [CMU Natural PG](slides/TRPO_CMU_Natural_PG.pdf), [Toronto PAIR](slides/TRPO_Toronto_PAIR.pdf) | [OpenAI Baselines](https://github.com/openai/baselines) | - |
| High-Dimensional Continuous Control (GAE) | [1506.02438](https://arxiv.org/abs/1506.02438) | [Berkeley CS285](slides/GAE_Berkeley_CS285.pdf) | [OpenAI Baselines](https://github.com/openai/baselines) | - |

### Actor-Critic Methods

| Paper | arXiv | Slides | Code | Media |
|-------|-------|--------|------|-------|
| Asynchronous Methods (A3C) | [1602.01783](https://arxiv.org/abs/1602.01783) | [WPI](slides/A3C_WPI_DS595.pdf), [Buffalo](slides/A3C_Buffalo_Actor_Critic.pdf), [NTU](slides/A3C_NTU_Taiwan.pdf), [UIUC](slides/A3C_UIUC_ECE448.pdf), [Julien Vitay](slides/A3C_Julien_Vitay.pdf) | [OpenAI Baselines](https://github.com/openai/baselines) | - |
| Continuous Control (DDPG) | [1509.02971](https://arxiv.org/abs/1509.02971) | [Paderborn](slides/DDPG_Paderborn_DPG.pdf), [FAU](slides/DDPG_FAU_Mutschler.pdf), [Julien Vitay](slides/DDPG_Julien_Vitay.pdf), [Buffalo](slides/DDPG_Buffalo_DPG.pdf) | [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) | - |
| Addressing Function Approximation (TD3) | [1802.09477](https://arxiv.org/abs/1802.09477) | [Prague](slides/TD3_SAC_Prague_NPFL139.pdf) | [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) | - |
| Soft Actor-Critic (SAC) | [1801.01290](https://arxiv.org/abs/1801.01290) | [Toronto PAIR](slides/SAC_Toronto_PAIR.pdf), [Purdue](slides/SAC_Purdue_RL_Inference.pdf), [Stanford CS231n](slides/SAC_Stanford_CS231n.pdf), [Prague](slides/TD3_SAC_Prague_NPFL139.pdf) | [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) | - |

### Temporal Difference & Q-Learning

| Paper | arXiv | Slides | Code | Media |
|-------|-------|--------|------|-------|
| TD Learning Fundamentals | - | [CMU](slides/TD_Learning_CMU.pdf), [Michigan](slides/TD_Methods_Michigan.pdf), [Sutton & Barto](slides/TD_Sutton_Barto.pdf) | - | - |
| Q-Learning | - | [Northeastern](slides/Q_Learning_Northeastern.pdf), [CMU TD](slides/TD_Learning_CMU.pdf) | - | - |

### Model-Based RL

| Paper | arXiv | Slides | Code | Media |
|-------|-------|--------|------|-------|
| Model-Based RL | - | [FAU](slides/Model_Based_RL_FAU.pdf), [Toronto](slides/Model_Based_RL_Toronto.pdf), [Berkeley](slides/Model_Based_RL_Berkeley.pdf), [CMU](slides/Model_Based_RL_CMU.pdf) | [MBRL-Lib](https://github.com/facebookresearch/mbrl-lib) | - |

### Imitation & Inverse RL

| Paper | arXiv | Slides | Code | Media |
|-------|-------|--------|------|-------|
| Imitation Learning | - | [WPI](slides/Imitation_Learning_WPI.pdf), [EPFL](slides/Imitation_Learning_EPFL.pdf) | [imitation](https://github.com/HumanCompatibleAI/imitation) | - |
| Inverse Reinforcement Learning | - | [TU Darmstadt](slides/Inverse_RL_TU_Darmstadt.pdf), [Berkeley CS285](slides/Inverse_RL_Berkeley_CS285.pdf) | [imitation](https://github.com/HumanCompatibleAI/imitation) | - |

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
2. Read [ReAct paper](reinforcement_learning_papers/09_agentic_rl/ReAct_Synergizing_Reasoning_and_Acting.pdf) + [slides](slides/ReAct_UVA_Lecture.pdf) + [code](https://github.com/ysymyth/ReAct)
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
1. [DeepSeek-R1 paper](reinforcement_learning_papers/02_rlhf_alignment/DeepSeek-R1_Reasoning_via_RL.pdf) + [DeepSeek R1 CMU slides](slides/DeepSeek_R1_CMU_Reasoning.pdf) + [code](https://github.com/deepseek-ai/DeepSeek-R1)
2. [DeepSeekMath GRPO](reinforcement_learning_papers/02_rlhf_alignment/DeepSeekMath_GRPO.pdf) + [Stanford RL for Reasoning](slides/Stanford_RL_for_LLM_Reasoning.pdf) + [code](https://github.com/deepseek-ai/DeepSeek-Math)
3. [ARTIST paper](reinforcement_learning_papers/09_agentic_rl/ARTIST_Agentic_Reasoning_Tool_Integration.pdf) for agentic reasoning with tools

---

## License

Papers are property of their respective authors. This collection is for educational purposes.
