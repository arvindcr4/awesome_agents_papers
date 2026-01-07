#!/usr/bin/env zsh
# Process extracted text files with aichat - much faster than PDFs

AICHAT="$HOME/.cargo/bin/aichat"
PAPERS_DIR="/Users/arvind/reinforcement_learning_papers"
TEXT_DIR="$PAPERS_DIR"
OUTPUT_DIR="$PAPERS_DIR/categorized"

# Create output directories
for dir in 01_core_methods 02_rlhf_alignment 03_multi_agent_rl 04_hierarchical_rl 05_safe_constrained_rl 06_curiosity_exploration 07_model_based_rl 08_imitation_learning; do
    mkdir -p "$OUTPUT_DIR/$dir"
done

# Analysis prompt
ANALYSIS_PROMPT='Analyze this reinforcement learning paper and provide:

1. Paper Title
2. ArXiv ID
3. Research Method Category - choose ONE:
   01_core_methods - Q-learning, policy gradients, actor-critic, DQN
   02_rlhf_alignment - RLHF, human feedback, preference learning
   03_multi_agent_rl - MARL, cooperative/competitive agents
   04_hierarchical_rl - options, skills, feudal networks
   05_safe_constrained_rl - safe RL, constrained MDPs
   06_curiosity_exploration - intrinsic motivation, exploration
   07_model_based_rl - world models, planning, dynamics
   08_imitation_learning - inverse RL, behavior cloning
4. Method Description (2-3 sentences)
5. Key Contributions (3-5 bullets)

Required format:
PAPER: [filename]
TITLE: [title]
ARXIV_ID: [id]
RESEARCH_METHOD: [category]
METHOD_DESCRIPTION: [description]
KEY_CONTRIBUTIONS:
- [contribution]'

# Use faster Llama model
MODEL="arliai:Llama-3.3-70B-Instruct"

# Process text file with timeout
process_text_file() {
    local txt_file="$1"
    local txt_name=$(basename "$txt_file" .txt)
    local pdf_name="${txt_name}.pdf"

    # Skip if summary already exists
    if find "$OUTPUT_DIR" -name "${txt_name}_summary.txt" -type f 2>/dev/null | grep -q .; then
        echo "  ⊘ Skipping (exists): $txt_name"
        return 0
    fi

    echo "Processing: $txt_name"

    # Run aichat with timeout using background process
    output=$($AICHAT -m "$MODEL" --prompt "$ANALYSIS_PROMPT" -f "$txt_file" --no-stream 2>&1)
    result=$?

    if [ $result -eq 0 ] && echo "$output" | grep -q "RESEARCH_METHOD:"; then
        # Extract category
        category=$(echo "$output" | grep "RESEARCH_METHOD:" | head -1 | sed 's/.*RESEARCH_METHOD: //' | tr -d ' \r\n')

        if [[ "$category" =~ ^0[1-8]_.* ]] && [ -d "$OUTPUT_DIR/$category" ]; then
            # Replace filename in output with PDF name
            final_output=$(echo "$output" | sed "s/PAPER: .*/PAPER: $pdf_name/")
            echo "$final_output" > "$OUTPUT_DIR/${category}/${txt_name}_summary.txt"
            echo "  ✓ Saved to ${category}/"
            return 0
        fi
    fi

    echo "  ✗ Failed: $txt_name"
    return 1
}

export AICHAT MODEL ANALYSIS_PROMPT OUTPUT_DIR
export -f process_text_file

# Main processing
batch_size=5
start_idx=1
total_txt=$(ls -1 "$TEXT_DIR"/*.txt 2>/dev/null | grep -E "[0-9]{4}_[0-9]+\.(txt|pdf)" | wc -l | tr -d ' ')
end_idx=$total_txt

echo "=== Processing $total_txt text files with aichat ==="
echo "Model: $MODEL"
echo "Batch size: $batch_size"
echo ""

processed=0
failed=0
LOG_FILE="$PAPERS_DIR/aichat_processing.log"
echo "Started $(date)" > "$LOG_FILE"

while [ $start_idx -le $end_idx ]; do
    batch_end=$((start_idx + batch_size - 1))
    [ $batch_end -gt $end_idx ] && batch_end=$end_idx

    echo "=== Batch $start_idx-$batch_end ==="

    # Get text files for this batch (only numbered arxiv papers)
    files=($(ls -1 "$TEXT_DIR"/*.txt 2>/dev/null | grep -E "[0-9]{4}_[0-9]+\.txt" | sort | sed -n "${start_idx},${batch_end}p"))

    for file in "${files[@]}"; do
        if process_text_file "$file"; then
            ((processed++))
        else
            ((failed++))
            echo "$file - FAILED" >> "$LOG_FILE"
        fi
    done

    echo "Progress: $processed/$total_txt"
    start_idx=$((batch_end + 1))
    sleep 2
done

echo ""
echo "=== Complete ==="
echo "Processed: $processed"
echo "Failed: $failed"
echo "Finished $(date)" >> "$LOG_FILE"
