#!/usr/bin/env zsh
AICHAT="$HOME/.cargo/bin/aichat"
TEXT_DIR="/Users/arvind/reinforcement_learning_papers"
OUTPUT_DIR="$TEXT_DIR/categorized"
MODEL="arliai:Llama-3.3-70B-Instruct"

PROMPT='Extract from this RL paper:
PAPER: [filename]
TITLE: [title]
RESEARCH_METHOD: [choose: 01_core_methods, 02_rlhf_alignment, 03_multi_agent_rl, 04_hierarchical_rl, 05_safe_constrained_rl, 06_curiosity_exploration, 07_model_based_rl, 08_imitation_learning]
METHOD_DESCRIPTION: [2-3 sentences]
KEY_CONTRIBUTIONS:
- [3-5 bullets]'

echo "=== Processing Remaining New Papers ==="
echo ""

# Skip the 3 problematic papers
remaining=(
"CQL_Conservative_Q_Learning.txt"
"Curiosity_ICM.txt"
"DDPG_Continuous_Control.txt"
"Decision_Transformer.txt"
"Does_RL_Incentivize_Reasoning.txt"
"DPO_Direct_Preference_Optimization.txt"
"DQN_Playing_Atari.txt"
"DreamerV3_Mastering_Diverse_Domains.txt"
"Generative_Agents_Simulacra.txt"
"ML_Agent_Autonomous_Engineering.txt"
"MuZero_Mastering_Atari.txt"
"PPO_Proximal_Policy_Optimization.txt"
"ReAct_Reasoning_Acting.txt"
"Reflexion_Verbal_RL.txt"
"RL_from_Human_Preferences.txt"
"RND_Exploration.txt"
"SAC_Soft_Actor_Critic.txt"
"Search_R1_Reasoning_Search.txt"
"Toolformer.txt"
"Tree_of_Thoughts.txt"
"Voyager_Open_Ended_Agent.txt"
"World_Models.txt"
)

processed=0

for paper in "${remaining[@]}"; do
    base="${paper%.txt}"
    
    if find "$OUTPUT_DIR" -name "${base}_summary.txt" 2>/dev/null | grep -q .; then
        echo "⊘ Skipping: $base (exists)"
        continue
    fi
    
    echo "Processing: $base"
    
    if [ ! -f "$TEXT_DIR/$paper" ]; then
        echo "  ✗ File not found"
        continue
    fi
    
    # Use first chunk only for speed
    chunk_file="/tmp/${base}_chunk.txt"
    head -1000 "$TEXT_DIR/$paper" > "$chunk_file"
    
    if output=$($AICHAT -m "$MODEL" --prompt "$PROMPT" -f "$chunk_file" --no-stream 2>&1); then
        cat=$(echo "$output" | sed -n 's/.*research.*method[^[:alnum:]]*\([0-9][0-9]_[^[:space:]]*\).*/\1/ip' | head -1 | tr -d '\r\n')
        if [[ "$cat" =~ ^0[1-8]_.* ]]; then
            final_output=$(echo "$output" | sed "s/PAPER: .*/PAPER: ${base}.pdf/")
            echo "$final_output" > "$OUTPUT_DIR/${cat}/${base}_summary.txt"
            echo "  ✓ $cat"
            ((processed++))
        else
            echo "  ✗ No category extracted"
        fi
    else
        echo "  ✗ aichat failed"
    fi
    
    rm -f "$chunk_file"
    sleep 1
done

echo ""
echo "=== Complete ==="
echo "Processed: $processed new papers"
echo "Total: $(find "$OUTPUT_DIR" -name "*_summary.txt" -type f | wc -l | tr -d ' ')"
