#!/usr/bin/env zsh
# Process the 84 papers with descriptive names (not arxiv numbered)

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

process_paper() {
    local txt="$1"
    local base=$(basename "$txt" .txt)
    
    # Skip _extracted versions (duplicates)
    [[ "$base" =~ _extracted$ ]] && return 0
    
    # Skip if summary exists (check with or without .pdf extension)
    if find "$OUTPUT_DIR" -name "${base}_summary.txt" -o -name "${base%.pdf}_summary.txt" 2>/dev/null | grep -q .; then
        return 0
    fi
    
    echo "Processing: $base"
    
    local size=$(wc -c < "$txt" 2>/dev/null | tr -d ' ')
    local size_kb=$((size / 1024))
    
    # For small files, process directly
    if [ $size_kb -lt 100 ]; then
        if output=$($AICHAT -m "$MODEL" --prompt "$PROMPT" -f "$txt" --no-stream 2>&1); then
            cat=$(echo "$output" | sed -n 's/.*research.*method[^[:alnum:]]*\([0-9][0-9]_[^[:space:]]*\).*/\1/ip' | head -1 | tr -d '\r\n')
            if [[ "$cat" =~ ^0[1-8]_.* ]]; then
                # Clean up output
                final_output=$(echo "$output" | sed "s/PAPER: .*/PAPER: ${base}.pdf/")
                echo "$final_output" > "$OUTPUT_DIR/${cat}/${base}_summary.txt"
                echo "  ✓ $cat"
                return 0
            fi
        fi
        echo "  ✗ Failed: $base"
        return 1
    fi
    
    # For large files, use first chunk
    local chunk_file="/tmp/${base}_chunk.txt"
    head -1500 "$txt" > "$chunk_file"
    
    if output=$($AICHAT -m "$MODEL" --prompt "$PROMPT" -f "$chunk_file" --no-stream 2>&1); then
        cat=$(echo "$output" | sed -n 's/.*research.*method[^[:alnum:]]*\([0-9][0-9]_[^[:space:]]*\).*/\1/ip' | head -1 | tr -d '\r\n')
        if [[ "$cat" =~ ^0[1-8]_.* ]]; then
            final_output=$(echo "$output" | sed "s/PAPER: .*/PAPER: ${base}.pdf/")
            # Add note about chunking
            final_output="$final_output"'
(NOTE: Processed using first-chunk strategy due to file size)'
            echo "$final_output" > "$OUTPUT_DIR/${cat}/${base}_summary.txt"
            rm -f "$chunk_file"
            echo "  ✓ $cat (chunked)"
            return 0
        fi
    fi
    
    rm -f "$chunk_file"
    echo "  ✗ Failed: $base"
    return 1
}

export AICHAT TEXT_DIR OUTPUT_DIR MODEL PROMPT
export -f process_paper

echo "=== Processing Descriptive Name Papers ==="
echo ""

# Get all non-arxiv numbered files (excluding _extracted duplicates)
files=($(ls -1 "$TEXT_DIR"/*.txt 2>/dev/null | grep -vE '[0-9]{4}_[0-9]+\.txt$' | grep -v '_extracted\.txt$'))
total=${#files[@]}

echo "Found $total files to process"
echo ""

processed=0
failed=0

# Process in parallel (4 at a time)
idx=0
while [ $idx -lt $total ]; do
    end=$((idx + 4))
    [ $end -gt $total ] && end=$total
    
    for ((j=idx; j<end; j++)); do
        process_paper "${files[$j]}" &
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
echo "Category breakdown:"
for dir in 01_core_methods 02_rlhf_alignment 03_multi_agent_rl 04_hierarchical_rl 05_safe_constrained_rl 07_model_based_rl; do
    count=$(ls "$OUTPUT_DIR/$dir"/*_summary.txt 2>/dev/null | wc -l | tr -d ' ')
    echo "  $dir: $count"
done
