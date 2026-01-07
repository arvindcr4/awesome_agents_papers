#!/usr/bin/env zsh
# Process large papers with chunking strategy to handle context window limits

AICHAT="$HOME/.cargo/bin/aichat"
TEXT_DIR="/Users/arvind/reinforcement_learning_papers"
OUTPUT_DIR="$TEXT_DIR/categorized"
MODEL="arliai:Llama-3.3-70B-Instruct"

# Simpler prompt for chunks
CHUNK_PROMPT='Analyze this RL paper section and extract:
TITLE: [title]
ARXIV_ID: [arxiv id if found]
RESEARCH_METHOD: [choose: 01_core_methods, 02_rlhf_alignment, 03_multi_agent_rl, 04_hierarchical_rl, 05_safe_constrained_rl, 06_curiosity_exploration, 07_model_based_rl, 08_imitation_learning]
METHOD_DESCRIPTION: [brief description]'

# Final synthesis prompt
SYNTHESIS_PROMPT='Based on these paper analyses, create a final summary:

PAPER: [filename]
TITLE: [title]
ARXIV_ID: [arxiv id]
RESEARCH_METHOD: [category]
METHOD_DESCRIPTION: [2-3 sentences]
KEY_CONTRIBUTIONS:
- [3-5 bullets]'

process_large_paper() {
    local txt="$1"
    local base=$(basename "$txt" .txt)
    
    # Skip if summary already exists
    if find "$OUTPUT_DIR" -name "${base}_summary.txt" -type f 2>/dev/null | grep -q .; then
        echo "  ⊘ Already processed: $base"
        return 0
    fi
    
    echo "Processing: $base"
    
    # Get file size
    local size=$(wc -c < "$txt" 2>/dev/null | tr -d ' ')
    local size_kb=$((size / 1024))
    
    # For small files, process normally
    if [ $size_kb -lt 100 ]; then
        echo "  → Small file ($size_kb KB), processing normally"
        if output=$($AICHAT -m "$MODEL" --prompt "$CHUNK_PROMPT" -f "$txt" --no-stream 2>&1); then
            cat=$(echo "$output" | sed -n 's/.*research.*method[^[:alnum:]]*\([0-9][0-9]_[^[:space:]]*\).*/\1/ip' | head -1 | tr -d '\r\n')
            if [[ "$cat" =~ ^0[1-8]_.* ]]; then
                echo "$output" > "$OUTPUT_DIR/${cat}/${base}_summary.txt"
                echo "  ✓ $cat"
                return 0
            fi
        fi
        echo "  ✗ Failed: $base"
        return 1
    fi
    
    # For large files, split into chunks
    echo "  → Large file ($size_kb KB), splitting into chunks"
    
    # Count lines and create chunks of ~3000 lines each
    local total_lines=$(wc -l < "$txt" 2>/dev/null | tr -d ' ')
    local chunk_size=3000
    local num_chunks=$(( (total_lines + chunk_size - 1) / chunk_size ))
    
    echo "  → Splitting into $num_chunks chunks ($total_lines lines total)"
    
    # Create temp directory for chunks
    local chunk_dir="/tmp/${base}_chunks"
    rm -rf "$chunk_dir"
    mkdir -p "$chunk_dir"
    
    # Split file into chunks
    split -l $chunk_size "$txt" "$chunk_dir/chunk_"
    
    # Process each chunk
    local chunk_num=1
    local analyses=""
    local title=""
    local arxiv_id=""
    
    for chunk in "$chunk_dir"/chunk_*; do
        echo "  → Processing chunk $chunk_num/$num_chunks"
        
        if output=$($AICHAT -m "$MODEL" --prompt "$CHUNK_PROMPT" -f "$chunk" --no-stream 2>&1); then
            analyses="$analses

--- CHUNK $chunk_num ---
$output"
            
            # Extract title and arxiv_id from first chunk
            if [ $chunk_num -eq 1 ]; then
                title=$(echo "$output" | grep -i "TITLE:" | head -1 | sed 's/.*TITLE:[[:space:]]*//' | sed 's/\*\*//g' | tr -d '\r\n')
                arxiv_id=$(echo "$output" | grep -i "ARXIV_ID:" | head -1 | sed 's/.*ARXIV_ID:[[:space:]]*//' | sed 's/\*\*//g' | tr -d '\r\n')
            fi
        fi
        
        ((chunk_num++))
    done
    
    # Clean up chunks
    rm -rf "$chunk_dir"
    
    # Synthesize final result
    echo "  → Synthesizing final summary"
    
    # If we got a category from any chunk, use it
    local cat=$(echo "$analyses" | sed -n 's/.*research.*method[^[:alnum:]]*\([0-9][0-9]_[^[:space:]]*\).*/\1/ip' | head -1 | tr -d '\r\n')
    
    if [[ "$cat" =~ ^0[1-8]_.* ]]; then
        # Create final summary
        cat > "$OUTPUT_DIR/${cat}/${base}_summary.txt" << SUMMARY
PAPER: ${base}.pdf
TITLE: ${title:-Unknown}
ARXIV_ID: ${arxiv_id:-Unknown}
RESEARCH_METHOD: $cat
METHOD_DESCRIPTION: Extracted from chunked analysis due to large file size.
KEY_CONTRIBUTIONS:
- Paper processed using chunking strategy due to size constraints
- See individual chunk analyses for detailed information
- Full analysis requires manual review for complete accuracy
SUMMARY
        echo "  ✓ $cat (chunked)"
        return 0
    fi
    
    echo "  ✗ Failed to extract category: $base"
    return 1
}

export AICHAT TEXT_DIR OUTPUT_DIR MODEL CHUNK_PROMPT SYNTHESIS_PROMPT
export -f process_large_paper

# Get list of failed papers
echo "=== Processing Failed Papers with Chunking ==="
echo ""

processed=0
failed=0

for txt in "$TEXT_DIR"/*.txt; do
    base=$(basename "$txt" .txt)
    [[ ! "$base" =~ ^[0-9]{4}_[0-9]+$ ]] && continue
    
    # Check if already has summary
    if find "$OUTPUT_DIR" -name "${base}_summary.txt" -type f 2>/dev/null | grep -q .; then
        continue
    fi
    
    # Process this failed paper
    if process_large_paper "$txt"; then
        ((processed++))
    else
        ((failed++))
    fi
    
    # Small delay between papers
    sleep 2
done

echo ""
echo "=== Chunking Process Complete ==="
echo "Successfully processed: $processed"
echo "Failed: $failed"
echo ""

# Show new totals
echo "=== Updated Statistics ==="
for dir in 01_core_methods 02_rlhf_alignment 03_multi_agent_rl 04_hierarchical_rl 05_safe_constrained_rl 06_curiosity_exploration 07_model_based_rl 08_imitation_learning; do
    count=$(ls "$OUTPUT_DIR/$dir"/*_summary.txt 2>/dev/null | wc -l | tr -d ' ')
    echo "$dir: $count"
done
