#!/usr/bin/env zsh
# Parallel processing with aichat - 4 workers

AICHAT="$HOME/.cargo/bin/aichat"
TEXT_DIR="/Users/arvind/reinforcement_learning_papers"
OUTPUT_DIR="$TEXT_DIR/categorized"

MODEL="arliai:Llama-3.3-70B-Instruct"

PROMPT='Extract from this RL paper:
PAPER: [filename]
TITLE: [title]
ARXIV_ID: [arxiv id]
RESEARCH_METHOD: [choose: 01_core_methods, 02_rlhf_alignment, 03_multi_agent_rl, 04_hierarchical_rl, 05_safe_constrained_rl, 06_curiosity_exploration, 07_model_based_rl, 08_imitation_learning]
METHOD_DESCRIPTION: [2-3 sentences]
KEY_CONTRIBUTIONS:
- [3-5 bullets]'

process_one() {
    local txt="$1"
    local base=$(basename "$txt" .txt)

    # Skip if summary exists (using find to avoid glob issues)
    find "$OUTPUT_DIR" -name "${base}_summary.txt" -type f 2>/dev/null | grep -q . && return 0

    # Find matching txt file (might have .txt or .pdf.txt)
    [ ! -f "$txt" ] && txt="${TEXT_DIR}/${base}.pdf.txt"
    [ ! -f "$txt" ] && return 1

    echo "Processing: $base"

    # Run aichat
    if output=$($AICHAT -m "$MODEL" --prompt "$PROMPT" -f "$txt" --no-stream 2>&1); then
        # Extract category (handle bold markdown format)
        cat=$(echo "$output" | sed -n 's/.*research.*method[^[:alnum:]]*\([0-9][0-9]_[^[:space:]]*\).*/\1/ip' | head -1 | tr -d '\r\n')
        [[ "$cat" =~ ^0[1-8]_.* ]] || return 1

        # Fix paper name
        out=$(echo "$output" | sed "s/PAPER: .*/PAPER: ${base}.pdf/")
        echo "$out" > "$OUTPUT_DIR/${cat}/${base}_summary.txt"
        echo "  ✓ $cat"
    fi
}

export AICHAT TEXT_DIR OUTPUT_DIR MODEL PROMPT
export -f process_one

# Get all text files (arxiv numbered ones)
files=($(ls -1 "$TEXT_DIR"/*.txt 2>/dev/null | grep -E '[0-9]{4}_[0-9]+\.txt$' | sort))
total=${#files[@]}

echo "=== Parallel processing $total files ==="
echo "Workers: 4"

processed=0
idx=0
while [ $idx -lt $total ]; do
    # Process next 4 files
    for ((j=0; j<4 && idx<total; j++, idx++)); do
        process_one "${files[$idx]}" &
    done
    wait

    processed=$((idx))
    echo "Progress: $processed/$total"
    sleep 1
done

echo "=== Done ==="
