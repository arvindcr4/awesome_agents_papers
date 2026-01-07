# RL Papers Classification Audit

**Date:** 2025-01-07
**Scope:** targeted audit of merged directories (`09_agentic_rl`, `05_safe_constrained_rl`) against the 16 Research Directions definitions.

## Summary
The majority of papers are correctly classified. However, several clusters of papers from the source `papers/` directory do not strictly fit the *Reinforcement Learning* definitions of their assigned categories.

## 1. Category 09: The Operator (Agentic RL)
**Definition:** "Controls software... bridges RL with HCI and LLMs... actions are clicking buttons, typing text..."

### Flagged Papers (Likely Pure Prompting/Reasoning)
The following papers appear to focus on **Chain-of-Thought (CoT)** prompting or **Transformer Reasoning** capabilities *without* an explicit Reinforcement Learning loop or Agentic Environment interaction. They are "Pre-RL" or "Prompt Engineering" research.

*   `Chain-of-Thought_Reasoning_Without_Prompting.pdf` (Prompting)
*   `Chain-of-Thought_Empowers_Transformers...` (Analysis)
*   `Grokked_Transformers_are_Implicit_Reasoners.pdf` (Interpretability)
*   `Large_Language_Models_Cannot_Self-Correct_Reasoning_Yet.pdf` (Evaluation)
*   `Premise_Order_Matters_in_Reasoning...` (Analysis)
*   `Beyond_A-Star_Better_Planning_Transformers.pdf` (Planning, might be Search not RL)
*   `Composing_Global_Optimizers_Algebraic_Objects.pdf` (Optimization)

**Recommendation:** Consider moving these to a new `llm_reasoning_and_prompting` category or explicitly acknowledging them as "Foundational Reasoning" for agents.

## 2. Category 05: The Safety Engineer (Safe/Constrained RL)
**Definition:** "Optimize reward while obeying strict safety constraints... Constrained Policy Optimization (CPO)..." (Control Theory focus).

### Flagged Papers (LLM Security & Privacy)
The following papers focus on **Adversarial Attacks, Red Teaming, and Data Privacy** for LLMs. While "Safety" related, they do not fall under the standard "Safe RL" (Constrained MDP) research direction.

*   `Extracting_Training_Data_from_LLMs.pdf` (Privacy/Extraction)
*   `The_Secret_Sharer_Unintended_Memorization.pdf` (Privacy)
*   `AgentPoison_Red-teaming_LLM_Agents.pdf` (Adversarial/Red Teaming)
*   `DataSentinel_Game-Theoretic_Detection_Prompt_Injection.pdf` (Prompt Injection)
*   `DecodingTrust_Trustworthiness_GPT_Models.pdf` (Trust Benchmark)
*   `Big_Sleep_LLM_Vulnerabilities_Real-World.pdf` (Vulnerabilities)
*   `Representation_Engineering_AI_Transparency.pdf` (Interpretability/Transparency)

**Recommendation:** These are distinct from "Constrained RL". Move to a `llm_security_and_redteaming` category.

## 3. Category 02: The Aligner (RLHF)
**Definition:** "AI Safety, Alignment, learning from subjective feedback... DPO...".

### Observation (Reasoning Models)
Papers like `DeepSeek-R1`, `OpenAI_O1_Replication`, `Qwen_QwQ` use RL (GRPO/PPO) for *Reasoning*, not just Alignment/Safety. They are correctly placed here as they use RL-Finetuning, but they represent a shift from "Alignment" to "Reasoning Scaling".

*   `DeepSeek-R1_Reasoning_via_RL.pdf`
*   `R-Search_Multi-Step_Reasoning.pdf`
*   `Sky-T1_Training_Small_Reasoning_LLMs.pdf`

**Status:** Acceptable fit, as they use RL methods (PPO/GRPO), but could also fit `09_agentic_rl` or `01_core_methods`. Keeping in `02` or `09` is a choice of "Method" vs "Capability".

## 4. Other Notes
*   **HippoRAG** (`09_agentic_rl`): RAG is typically a memory module. Fits broadly under Agentic Tool Use.
*   **MasRouter** (`03_multi_agent_rl`): Fits correctly (Multi-Agent Routing).
*   **DigiRL** (`09_agentic_rl`): Fits perfectly (Device Control).

## Conclusion
~15-20 papers are "misfits" in strict RL categories. They represent the adjacent fields of **LLM Reasoning** and **LLM Security** which are often bundled with Agentic research but are distinct from RL algorithms.
