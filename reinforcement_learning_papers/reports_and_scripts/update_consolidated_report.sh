#!/bin/bash
# Update the consolidated report with new papers

REPORT="/Users/arvind/reinforcement_learning_papers/consolidated_rl_papers_report.md"
CATEGORIZED_DIR="/Users/arvind/reinforcement_learning_papers/categorized"

# Create new report with updated stats
cat > "$REPORT" << 'HEADER'
# Reinforcement Learning Research Papers Analysis Report

**Generated:** 2026-01-06 (Updated)
**Total Papers Analyzed:** 235 out of 277 (85% success rate)
**Source:** ArXiv papers from /Users/arvind/reinforcement_learning_papers

---

## Summary Statistics

| Category | Count | Percentage |
|----------|-------|------------|
| 03_multi_agent_rl | 128 | 54% |
| 02_rlhf_alignment | 62 | 26% |
| 07_model_based_rl | 32 | 14% |
| 04_hierarchical_rl | 6 | 3% |
| 01_core_methods | 4 | 2% |
| 05_safe_constrained_rl | 3 | 1% |
| 06_curiosity_exploration | 0 | 0% |
| 08_imitation_learning | 0 | 0% |

**Processing phases:**
- Phase 1 (Initial): 201 papers (73%)
- Phase 2 (Chunking): +34 papers
- **Total: 235 papers (85%)**

**Note:** 42 papers failed due to:
- Context window overflow (>55K tokens)
- Poor PDF text extraction
- Unparseable output format

---

## Research Method Categories

### 01_core_methods
Q-learning, policy gradients, actor-critic, DQN, basic RL algorithms

### 02_rlhf_alignment
RLHF, human feedback, preference learning, alignment

### 03_multi_agent_rl
MARL, cooperative/competitive agents, multi-agent systems

### 04_hierarchical_rl
Options, skills, feudal networks, hierarchical approaches

### 05_safe_constrained_rl
Safe RL, constrained MDPs, risk-sensitive RL

### 06_curiosity_exploration
Intrinsic motivation, exploration bonuses, curiosity-driven

### 07_model_based_rl
World models, planning, dynamics models, model-based approaches

### 08_imitation_learning
Inverse RL, behavior cloning, GAIL, imitation learning

---

HEADER

# Add all papers by category
for category in 01_core_methods 02_rlhf_alignment 03_multi_agent_rl 04_hierarchical_rl 05_safe_constrained_rl 07_model_based_rl; do
  dir="$CATEGORIZED_DIR/$category"
  if [ -d "$dir" ]; then
    echo "## $category" >> "$REPORT"
    echo "" >> "$REPORT"
    
    find "$dir" -name "*_summary.txt" -type f | sort | while read file; do
      echo "---" >> "$REPORT"
      cat "$file" >> "$REPORT"
      echo "" >> "$REPORT"
    done
  fi
done

echo "Report updated!"
wc -l "$REPORT"
