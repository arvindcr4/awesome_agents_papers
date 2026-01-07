#!/usr/bin/env zsh
# Parallel chunking strategy - process multiple papers at once

AICHAT="$HOME/.cargo/bin/aichat"
TEXT_DIR="/Users/arvind/reinforcement_learning_papers"
OUTPUT_DIR="$TEXT_DIR/categorized"
MODEL="arliai:Llama-3.3-70B-Instruct"

CHUNK_PROMPT='Analyze this RL paper section and extract:
TITLE: [title]
ARXIV_ID: [arxiv id if found]
RESEARCH_METHOD: [choose: 01_core_methods, 02_rlhf_alignment, 03_multi_agent_rl, 04_hierarchical_rl, 05_safe_constrained_rl, 06_curiosity_exploration, 07_model_based_rl, 08_imitation_learning]
METHOD_DESCRIPTION: [brief description]'

process_paper() {
    local txt="$1"
    local base=$(basename "$txt" .txt)
    
    if find "$OUTPUT_DIR" -name "${base}_summary.txt" -type f 2>/dev/null | grep -q .; then
        return 0
    fi
    
    echo "Processing: $base" >&2
    
    local size=$(wc -c < "$txt" 2>/dev/null | tr -d ' ')
    local size_kb=$((size / 1024))
    
    # Small files - process directly
    if [ $size_kb -lt 100 ]; then
        if output=$($AICHAT -m "$MODEL" --prompt "$CHUNK_PROMPT" -f "$txt" --no-stream 2>&1); then
            cat=$(echo "$output" | sed -n 's/.*research.*method[^[:alnum:]]*\([0-9][0-9]_[^[:space:]]*\).*/\1/ip' | head -1 | tr -d '\r\n')
            if [[ "$cat" =~ ^0[1-8]_.* ]]; then
                echo "$output" > "$OUTPUT_DIR/${cat}/${base}_summary.txt"
                echo "✓ $base ($cat)" >&2
                return 0
            fi
        fi
        echo "✗ $base" >&2
        return 1
    fi
    
    # Large files - chunking
    local total_lines=$(wc -l < "$txt" 2>/dev/null | tr -d ' ')
    local chunk_size=3000
    local num_chunks=$(( (total_lines + chunk_size - 1) / chunk_size ))
    
    local chunk_dir="/tmp/${base}_chunks_$$"
    rm -rf "$chunk_dir"
    mkdir -p "$chunk_dir"
    split -l $chunk_size "$txt" "$chunk_dir/chunk_"
    
    local cat=""
    local title="Unknown"
    local arxiv_id="Unknown"
    
    local chunk_num=1
    for chunk in "$chunk_dir"/chunk_*; do
        if output=$($AICHAT -m "$MODEL" --prompt "$CHUNK_PROMPT" -f "$chunk" --no-stream 2>&1); then
            if [ $chunk_num -eq 1 ]; then
                title=$(echo "$output" | grep -i "TITLE:" | head -1 | sed 's/.*TITLE:[[:space:]]*//' | sed 's/\*\*//g' | tr -d '\r\n')
                arxiv_id=$(echo "$output" | grep -i "ARXIV_ID:" | head -1 | sed 's/.*ARXIV_ID:[[:space:]]*//' | sed 's/\*\*//g' | tr -d '\r\n')
            fi
            cat=$(echo "$output" | sed -n 's/.*research.*method[^[:alnum:]]*\([0-9][0-9]_[^[:space:]]*\).*/\1/ip' | head -1 | tr -d '\r\n')
            [[ "$cat" =~ ^0[1-8]_.* ]] && break
        fi
        ((chunk_num++))
    done
    
    rm -rf "$chunk_dir"
    
    if [[ "$cat" =~ ^0[1-8]_.* ]]; then
        cat > "$OUTPUT_DIR/${cat}/${base}_summary.txt" << SUMMARY
PAPER: ${base}.pdf
TITLE: ${title}
ARXIV_ID: ${arxiv_id}
RESEARCH_METHOD: $cat
METHOD_DESCRIPTION: Extracted from chunked analysis due to large file size ($size_kb KB, $num_chunks chunks).
KEY_CONTRIBUTIONS:
- Paper processed using chunking strategy due to size constraints
- Analyzed across $num_chunks sections to determine category
- Full detailed analysis requires manual review
SUMMARY
        echo "✓ $base ($cat - chunked)" >&2
        return 0
    fi
    
    echo "✗ $base (no category extracted)" >&2
    return 1
}

export AICHAT TEXT_DIR OUTPUT_DIR MODEL CHUNK_PROMPT
export -f process_paper

# Group papers by size
echo "=== Parallel Chunking Strategy ==="
echo ""

small_files=()
medium_files=()
large_files=()

for txt in "$TEXT_DIR"/*.txt; do
    base=$(basename "$txt" .txt)
    [[ ! "$base" =~ ^[0-9]{4}_[0-9]+$ ]] && continue
    
    find "$OUTPUT_DIR" -name "${base}_summary.txt" -type f 2>/dev/null | grep -q . && continue
    
    size=$(wc -c < "$txt" 2>/dev/null | tr -d ' ')
    size_kb=$((size / 1024))
    
    if [ $size_kb -lt 100 ]; then
        small_files+=("$txt")
    elif [ $size_kb -lt 150 ]; then
        medium_files+=("$txt")
    else
        large_files+=("$txt")
    fi
done

echo "Papers to process:"
echo "  Small (<100KB): ${#small_files[@]}"
echo "  Medium (100-150KB): ${#medium_files[@]}"
echo "  Large (>150KB): ${#large_files[@]}"
echo ""

processed=0
failed=0

# Process small files in parallel (6 at once)
echo "Processing small files (6 parallel)..."
for ((i=0; i<${#small_files[@]}; i+=6)); do
    batch=("${small_files[@]:i:6}")
    for file in "${batch[@]}"; do
        process_paper "$file" &
    done
    wait
done

# Process medium files in parallel (4 at once)
echo "Processing medium files (4 parallel)..."
for ((i=0; i<${#medium_files[@]}; i+=4)); do
    batch=("${medium_files[@]:i:4}")
    for file in "${batch[@]}"; do
        process_paper "$file" &
    done
    wait
done

# Process large files in parallel (2 at once)
echo "Processing large files (2 parallel)..."
for ((i=0; i<${#large_files[@]}; i+=2)); do
    batch=("${large_files[@]:i:2}")
    for file in "${batch[@]}"; do
        process_paper "$file" &
    done
    wait
done

echo ""
echo "=== Complete ==="
echo "Total processed: $(find "$OUTPUT_DIR" -name "*_summary.txt" -type f | wc -l | tr -d ' ')"
