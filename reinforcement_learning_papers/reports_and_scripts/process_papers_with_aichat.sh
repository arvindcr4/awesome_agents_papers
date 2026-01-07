#!/usr/bin/env zsh
# Script to process RL papers using aichat with retry logic for API limits

AICHAT="$HOME/.cargo/bin/aichat"
PAPERS_DIR="/Users/arvind/reinforcement_learning_papers"
OUTPUT_DIR="$PAPERS_DIR/categorized"

# Create output directories
for dir in 01_core_methods 02_rlhf_alignment 03_multi_agent_rl 04_hierarchical_rl 05_safe_constrained_rl 06_curiosity_exploration 07_model_based_rl 08_imitation_learning; do
    mkdir -p "$OUTPUT_DIR/$dir"
done

# Prompt template for paper analysis
ANALYSIS_PROMPT='Analyze this reinforcement learning paper and extract:
1. Paper Title
2. ArXiv ID (from filename or content)
3. Research Method Category (choose one):
   - 01_core_methods: Q-learning, policy gradients, actor-critic, DQN, basic RL
   - 02_rlhf_alignment: RLHF, human feedback, preference learning
   - 03_multi_agent_rl: MARL, cooperative/competitive agents
   - 04_hierarchical_rl: options, skills, feudal networks
   - 05_safe_constrained_rl: safe RL, constrained MDPs
   - 06_curiosity_exploration: intrinsic motivation, exploration
   - 07_model_based_rl: world models, planning, dynamics
   - 08_imitation_learning: inverse RL, behavior cloning, GAIL
4. Method Description (2-3 sentences)
5. Key Contributions (3-5 bullet points)

Format output as:
PAPER: filename
TITLE: [title]
ARXIV_ID: [id]
RESEARCH_METHOD: [category]
METHOD_DESCRIPTION: [description]
KEY_CONTRIBUTIONS:
- [contribution 1]
- [contribution 2]
- ...'

# Process a single paper with retry
process_paper() {
    local pdf_file="$1"
    local pdf_name=$(basename "$pdf_file" .pdf)
    local max_retries=3
    local retry_count=0

    echo "Processing: $pdf_name"

    while [ $retry_count -lt $max_retries ]; do
        # Use aichat to analyze the paper
        if output=$($AICHAT -m "arliai:GLM-4.7" --prompt "$ANALYSIS_PROMPT" -f "$pdf_file" --no-stream 2>&1); then
            # Extract category from output
            category=$(echo "$output" | grep "RESEARCH_METHOD:" | head -1 | sed 's/RESEARCH_METHOD: //' | tr -d ' ')

            if [ -n "$category" ] && [ -d "$OUTPUT_DIR/$category" ]; then
                # Save summary
                echo "$output" > "$OUTPUT_DIR/${category}/${pdf_name}_summary.txt"
                echo "  ✓ Saved to $category/"
                return 0
            else
                echo "  ✗ Failed to extract category"
            fi
        else
            echo "  ✗ aichat failed (attempt $((retry_count + 1))/$max_retries)"
        fi

        retry_count=$((retry_count + 1))
        if [ $retry_count -lt $max_retries ]; then
            sleep 2  # Wait before retry
        fi
    done

    return 1
}

# Main processing - process in batches
batch_size=5  # Process 5 papers at a time to avoid limits
start_idx=1
end_idx=355

while [ $start_idx -le $end_idx ]; do
    batch_end=$((start_idx + batch_size - 1))
    if [ $batch_end -gt $end_idx ]; then
        batch_end=$end_idx
    fi

    echo ""
    echo "=== Processing batch $start_idx-$batch_end ==="

    # Get papers for this batch
    papers=($(ls -1 "$PAPERS_DIR"/*.pdf | sort | sed -n "${start_idx},${batch_end}p"))

    for paper in "${papers[@]}"; do
        process_paper "$paper"
    done

    start_idx=$((batch_end + 1))
    sleep 3  # Brief pause between batches
done

echo ""
echo "=== Processing complete ==="
