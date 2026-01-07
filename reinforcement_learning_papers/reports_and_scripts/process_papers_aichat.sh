#!/usr/bin/env zsh
# Script to process RL papers using aichat with retry logic and fallback models

AICHAT="$HOME/.cargo/bin/aichat"
PAPERS_DIR="/Users/arvind/reinforcement_learning_papers"
OUTPUT_DIR="$PAPERS_DIR/categorized"

# Create output directories
for dir in 01_core_methods 02_rlhf_alignment 03_multi_agent_rl 04_hierarchical_rl 05_safe_constrained_rl 06_curiosity_exploration 07_model_based_rl 08_imitation_learning; do
    mkdir -p "$OUTPUT_DIR/$dir"
done

# Analysis prompt - focused on extracting key information
ANALYSIS_PROMPT='Analyze this reinforcement learning research paper and provide:

1. Paper Title (exact title)
2. ArXiv ID (from filename or paper header)
3. Research Method Category - choose ONE:
   - 01_core_methods: Q-learning, policy gradients, actor-critic, DQN, basic RL algorithms
   - 02_rlhf_alignment: RLHF, human feedback, preference learning, alignment
   - 03_multi_agent_rl: MARL, cooperative/competitive agents, multi-agent systems
   - 04_hierarchical_rl: options, skills, feudal networks, hierarchical approaches
   - 05_safe_constrained_rl: safe RL, constrained MDPs, risk-sensitive RL
   - 06_curiosity_exploration: intrinsic motivation, exploration bonuses, curiosity-driven
   - 07_model_based_rl: world models, planning, dynamics models, model-based approaches
   - 08_imitation_learning: inverse RL, behavior cloning, GAIL, imitation learning
4. Method Description (2-3 sentences explaining the core approach)
5. Key Contributions (3-5 bullet points of main contributions)

Required output format:
---
PAPER: [filename.pdf]
TITLE: [paper title]
ARXIV_ID: [arxiv ID like 1812.00922]
RESEARCH_METHOD: [category code like 07_model_based_rl]
METHOD_DESCRIPTION: [2-3 sentence description]
KEY_CONTRIBUTIONS:
- [contribution 1]
- [contribution 2]
- [contribution 3]
---'

# Models to try in order (primary + fallbacks)
MODELS=("arliai:GLM-4.7" "arliai:GLM-4.6-Derestricted-v5" "arliai:Llama-3.3-70B-Instruct")

# Process a single paper with retry and fallback models
process_paper() {
    local pdf_file="$1"
    local pdf_name=$(basename "$pdf_file" .pdf)
    local max_retries=3

    echo "Processing: $pdf_name"

    for model in "${MODELS[@]}"; do
        local retry_count=0

        while [ $retry_count -lt $max_retries ]; do
            # Use aichat to analyze the paper
            if output=$($AICHAT -m "$model" --prompt "$ANALYSIS_PROMPT" -f "$pdf_file" --no-stream 2>&1); then
                # Check if output contains the required markers
                if echo "$output" | grep -q "RESEARCH_METHOD:"; then
                    # Extract category from output
                    category=$(echo "$output" | grep "RESEARCH_METHOD:" | head -1 | sed 's/.*RESEARCH_METHOD: //' | tr -d ' \r\n')

                    # Validate category
                    if [[ "$category" =~ ^0[1-8]_.* ]]; then
                        # Save summary
                        echo "$output" > "$OUTPUT_DIR/${category}/${pdf_name}_summary.txt"
                        echo "  ✓ [$model] Saved to ${category}/"
                        return 0
                    else
                        echo "  ✗ Invalid category: '$category'"
                    fi
                else
                    echo "  ✗ No RESEARCH_METHOD found in output (attempt $((retry_count + 1))/$max_retries)"
                fi
            else
                echo "  ✗ aichat failed (attempt $((retry_count + 1))/$max_retries)"
            fi

            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $max_retries ]; then
                sleep 3
            fi
        done

        echo "  → Trying next model..."
    done

    echo "  ✗ Failed with all models"
    return 1
}

# Main processing - process in small batches
batch_size=3  # Small batch size to avoid limits
start_idx=1
total_papers=$(ls -1 "$PAPERS_DIR"/*.pdf 2>/dev/null | wc -l | tr -d ' ')
end_idx=$total_papers

echo "=== Paper Analysis with aichat ==="
echo "Total papers: $total_papers"
echo "Batch size: $batch_size"
echo "Models: ${MODELS[@]}"
echo ""

# Create log file
LOG_FILE="$PAPERS_DIR/processing.log"
echo "Started at $(date)" > "$LOG_FILE"

processed=0
failed=0

while [ $start_idx -le $end_idx ]; do
    batch_end=$((start_idx + batch_size - 1))
    if [ $batch_end -gt $end_idx ]; then
        batch_end=$end_idx
    fi

    echo ""
    echo "=== Batch $start_idx-$batch_end ($(date +%H:%M:%S)) ==="

    # Get papers for this batch
    papers=($(ls -1 "$PAPERS_DIR"/*.pdf 2>/dev/null | sort | sed -n "${start_idx},${batch_end}p"))

    for paper in "${papers[@]}"; do
        if process_paper "$paper"; then
            ((processed++))
        else
            ((failed++))
            echo "$paper - FAILED" >> "$LOG_FILE"
        fi
    done

    echo "Progress: $processed/$total_papers processed, $failed failed"
    start_idx=$((batch_end + 1))

    # Pause between batches to avoid rate limits
    sleep 5
done

echo ""
echo "=== Processing Complete ==="
echo "Processed: $processed"
echo "Failed: $failed"
echo "Finished at $(date)" >> "$LOG_FILE"
