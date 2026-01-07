#!/usr/bin/env zsh
# Process remaining 42 failed papers using first-chunk-only strategy
# The abstract and introduction (first 1000 lines) usually contain enough info

AICHAT="$HOME/.cargo/bin/aichat"
TEXT_DIR="/Users/arvind/reinforcement_learning_papers"
OUTPUT_DIR="$TEXT_DIR/categorized"
MODEL="arliai:Llama-3.3-70B-Instruct"

SIMPLE_PROMPT='Extract from this RL paper (first section only):
PAPER: [filename]
TITLE: [title]
ARXIV_ID: [arxiv id]
RESEARCH_METHOD: [choose from: 01_core_methods, 02_rlhf_alignment, 03_multi_agent_rl, 04_hierarchical_rl, 05_safe_constrained_rl, 06_curiosity_exploration, 07_model_based_rl, 08_imitation_learning]'

process_first_chunk() {
    local txt="$1"
    local base=$(basename "$txt" .txt)
    
    if find "$OUTPUT_DIR" -name "${base}_summary.txt" -type f 2>/dev/null | grep -q .; then
        return 0
    fi
    
    echo "Processing: $base"
    
    # Extract first 1000 lines (abstract + intro usually here)
    local chunk_file="/tmp/${base}_first_chunk.txt"
    head -1000 "$txt" > "$chunk_file"
    
    if output=$($AICHAT -m "$MODEL" --prompt "$SIMPLE_PROMPT" -f "$chunk_file" --no-stream 2>&1); then
        cat=$(echo "$output" | sed -n 's/.*research.*method[^[:alnum:]]*\([0-9][0-9]_[^[:space:]]*\).*/\1/ip' | head -1 | tr -d '\r\n')
        title=$(echo "$output" | grep -i "TITLE:" | head -1 | sed 's/.*TITLE:[[:space:]]*//' | sed 's/\*\*//g' | tr -d '\r\n')
        arxiv_id=$(echo "$output" | grep -i "ARXIV_ID:" | head -1 | sed 's/.*ARXIV_ID:[[:space:]]*//' | sed 's/\*\*//g' | tr -d '\r\n')
        
        rm -f "$chunk_file"
        
        if [[ "$cat" =~ ^0[1-8]_.* ]]; then
            cat > "$OUTPUT_DIR/${cat}/${base}_summary.txt" << SUMMARY
PAPER: ${base}.pdf
TITLE: ${title:-Unknown}
ARXIV_ID: ${arxiv_id:-Unknown}
RESEARCH_METHOD: $cat
METHOD_DESCRIPTION: Extracted from paper abstract and introduction (first 1000 lines) due to size constraints.
KEY_CONTRIBUTIONS:
- Processed using first-chunk strategy (abstract + introduction)
- Category determined from paper's opening sections
- Full detailed analysis requires manual review of complete paper
SUMMARY
            echo "  ✓ $cat"
            return 0
        fi
    fi
    
    rm -f "$chunk_file"
    echo "  ✗ Failed: $base"
    return 1
}

export AICHAT TEXT_DIR OUTPUT_DIR MODEL SIMPLE_PROMPT
export -f process_first_chunk

# Find remaining failed papers
echo "=== Processing Remaining Failed Papers (First-Chunk Strategy) ==="
echo ""

remaining=()
for txt in "$TEXT_DIR"/*.txt; do
    base=$(basename "$txt" .txt)
    [[ ! "$base" =~ ^[0-9]{4}_[0-9]+$ ]] && continue
    
    find "$OUTPUT_DIR" -name "${base}_summary.txt" -type f 2>/dev/null | grep -q . && continue
    
    remaining+=("$txt")
done

echo "Remaining papers: ${#remaining[@]}"
echo ""

processed=0
failed=0

# Process in parallel (4 at a time for stability)
idx=0
total=${#remaining[@]}
while [ $idx -lt $total ]; do
    end=$((idx + 4))
    [ $end -gt $total ] && end=$total
    
    for ((j=idx; j<end; j++)); do
        process_first_chunk "${remaining[$j]}" &
    done
    wait
    
    idx=$end
    echo "Progress: $idx/$total"
    sleep 1
done

echo ""
echo "=== Complete ==="
echo "Total processed: $(find "$OUTPUT_DIR" -name "*_summary.txt" -type f | wc -l | tr -d ' ')"

# Show final breakdown
echo ""
echo "Final category breakdown:"
for dir in 01_core_methods 02_rlhf_alignment 03_multi_agent_rl 04_hierarchical_rl 05_safe_constrained_rl 07_model_based_rl; do
    count=$(ls "$OUTPUT_DIR/$dir"/*_summary.txt 2>/dev/null | wc -l | tr -d ' ')
    echo "  $dir: $count"
done
