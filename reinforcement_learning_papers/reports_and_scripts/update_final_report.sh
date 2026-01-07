#!/bin/bash
# Update the final consolidated report with all 336 papers

REPORT="/Users/arvind/reinforcement_learning_papers/consolidated_rl_papers_report.md"
CATEGORIZED_DIR="/Users/arvind/reinforcement_learning_papers/categorized"

cat > "$REPORT" << 'HEADER'
# Reinforcement Learning Research Papers Analysis Report

**Generated:** 2026-01-06 (Final)
**Total Papers Analyzed:** 336 out of 386 (87% success rate)
**Source:** /Users/arvind/reinforcement_learning_papers

---

## Summary Statistics

| Category | Count | Percentage |
|----------|-------|------------|
| 03_multi_agent_rl | 162 | 48% |
| 02_rlhf_alignment | 98 | 29% |
| 07_model_based_rl | 47 | 14% |
| 04_hierarchical_rl | 11 | 3% |
| 01_core_methods | 8 | 2% |
| 05_safe_constrained_rl | 6 | 2% |
| 06_curiosity_exploration | 4 | 1% |
| 08_imitation_learning | 0 | 0% |

**Processing success:** 336/386 papers (87%)

**Failed:** 50 papers (13%) - mostly due to API timeouts

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
for category in 01_core_methods 02_rlhf_alignment 03_multi_agent_rl 04_hierarchical_rl 05_safe_constrained_rl 06_curiosity_exploration 07_model_based_rl 08_imitation_learning; do
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
ls -lh "$REPORT"
