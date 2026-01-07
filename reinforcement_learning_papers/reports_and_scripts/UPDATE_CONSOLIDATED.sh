#!/bin/bash
# Update consolidated report with all 275 papers

REPORT="/Users/arvind/reinforcement_learning_papers/consolidated_rl_papers_report.md"
CATEGORIZED_DIR="/Users/arvind/reinforcement_learning_papers/categorized"

cat > "$REPORT" << 'HEADER'
# Reinforcement Learning Research Papers Analysis Report

**Generated:** 2026-01-06 (Final)
**Total Papers Analyzed:** 275 out of 277 (99.3% success rate)
**Source:** ArXiv papers from /Users/arvind/reinforcement_learning_papers

---

## Summary Statistics

| Category | Count | Percentage |
|----------|-------|------------|
| 03_multi_agent_rl | 160 | 58% |
| 02_rlhf_alignment | 63 | 23% |
| 07_model_based_rl | 37 | 13% |
| 04_hierarchical_rl | 8 | 3% |
| 01_core_methods | 4 | 1% |
| 05_safe_constrained_rl | 3 | 1% |
| 06_curiosity_exploration | 0 | 0% |
| 08_imitation_learning | 0 | 0% |

**Processing phases:**
- Phase 1 (Direct): 201 papers (73%)
- Phase 2 (Chunking): +34 papers (12%)
- Phase 3 (First-chunk): +40 papers (14%)
- **Total: 275 papers (99.3%)**

**Failed:** 2 papers (0.7%)

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

echo "Consolidated report updated!"
wc -l "$REPORT"
