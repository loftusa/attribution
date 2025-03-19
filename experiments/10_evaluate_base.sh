#!/bin/bash

# Create output directory if it doesn't exist
OUTPUT_DIR="../data/cruxeval_results_base_revisions"
mkdir -p $OUTPUT_DIR

# Function to clear HuggingFace cache
clear_hf_cache() {
    echo "Clearing HuggingFace cache..."
    rm -rf ~/.cache/huggingface/hub/models--allenai--OLMo-2-1124-7B*/snapshots/*
    echo "Cache cleared"
}

# Function to process a batch of revisions
echo "TESTING"
process_batch() {
    local start=$1
    local end=$2
    local batch_num=$3
    
    echo "Processing batch $batch_num (revisions $start to $end)"
    
    # Process each revision in the batch
    for ((i=start; i<=end && i<${#REVISIONS[@]}; i++)); do
        revision="${REVISIONS[$i]}"
        echo "====================================================="
        echo "Evaluating revision: $revision ($(($i + 1))/${#REVISIONS[@]})"
        echo "====================================================="
        
        # Run the evaluation script
        python 07_evaluate_cruxeval.py \
            --model "$model" \
            --revision "$revision" \
            --num_problems 1 \
            --batch_size 1 \
            --output_dir "$OUTPUT_DIR"
        
        # Check if evaluation was successful
        if [ $? -eq 0 ]; then
            echo "Completed evaluation for $model, revision $revision"
        else
            echo "Error evaluating $model, revision $revision"
        fi
        
        echo "====================================================="
        echo ""
        
        # Clear CUDA cache
        python -c "import torch; torch.cuda.empty_cache()"
    done
    
    # Clear HuggingFace cache after batch
    clear_hf_cache
}

REVISIONS=(
    "stage1-step150-tokens1B"
    "stage1-step15000-tokens63B"
    "stage1-step35000-tokens147B"
    "stage1-step55000-tokens231B"
    "stage1-step75000-tokens315B"
    "stage1-step95000-tokens399B"
    "stage1-step116000-tokens487B"
    "stage1-step136000-tokens571B"
    "stage1-step156000-tokens655B"
    "stage1-step177000-tokens743B"
    "stage1-step197000-tokens827B"
    "stage1-step217000-tokens911B"
    "stage1-step237000-tokens995B"
    "stage1-step257000-tokens1078B"
    "stage1-step277000-tokens1162B"
    "stage1-step297000-tokens1246B"
    "stage1-step318000-tokens1334B"
    "stage1-step338000-tokens1418B"
    "stage1-step358000-tokens1502B"
    "stage1-step378000-tokens1586B"
    "stage1-step398000-tokens1670B"
    "stage1-step418000-tokens1754B"
    "stage1-step438000-tokens1838B"
    "stage1-step458000-tokens1921B"
    "stage1-step478000-tokens2005B"
    "stage1-step498000-tokens2089B"
    "stage1-step518000-tokens2173B"
    "stage1-step539000-tokens2261B"
    "stage1-step559000-tokens2345B"
    "stage1-step579000-tokens2429B"
    "stage1-step599000-tokens2513B"
    "stage1-step620000-tokens2601B"
    "stage1-step641000-tokens2689B"
    "stage1-step661000-tokens2773B"
    "stage1-step682000-tokens2861B"
    "stage1-step702000-tokens2945B"
    "stage1-step722000-tokens3029B"
    "stage1-step742000-tokens3113B"
    "stage1-step762000-tokens3197B"
    "stage1-step782000-tokens3280B"
    "stage1-step802000-tokens3364B"
    "stage1-step822000-tokens3448B"
    "stage1-step842000-tokens3532B"
    "stage1-step862000-tokens3616B"
    "stage1-step882000-tokens3700B"
    "stage1-step902000-tokens3784B"
    "stage1-step922000-tokens3868B"
    "stage2-ingredient3-step1000-tokens5B"
    "stage2-ingredient2-step1000-tokens5B"
    "stage2-ingredient1-step1000-tokens5B"
    "stage2-ingredient3-step2000-tokens9B"
    "stage2-ingredient2-step2000-tokens9B"
    "stage2-ingredient1-step2000-tokens9B"
    "stage2-ingredient3-step3000-tokens13B"
    "stage2-ingredient2-step3000-tokens13B"
    "stage2-ingredient1-step3000-tokens13B"
    "stage2-ingredient3-step4000-tokens17B"
    "stage2-ingredient2-step4000-tokens17B"
    "stage2-ingredient1-step4000-tokens17B"
    "stage2-ingredient3-step5000-tokens21B"
    "stage2-ingredient2-step5000-tokens21B"
    "stage2-ingredient1-step5000-tokens21B"
    "stage2-ingredient3-step6000-tokens26B"
    "stage2-ingredient2-step6000-tokens26B"
    "stage2-ingredient1-step6000-tokens26B"
    "stage2-ingredient3-step7000-tokens30B"
    "stage2-ingredient2-step7000-tokens30B"
    "stage2-ingredient1-step7000-tokens30B"
    "stage2-ingredient3-step8000-tokens34B"
    "stage2-ingredient2-step8000-tokens34B"
    "stage2-ingredient1-step8000-tokens34B"
    "stage2-ingredient3-step9000-tokens38B"
    "stage2-ingredient2-step9000-tokens38B"
    "stage2-ingredient1-step9000-tokens38B"
    "stage2-ingredient3-step10000-tokens42B"
    "stage2-ingredient2-step10000-tokens42B"
    "stage2-ingredient1-step10000-tokens42B"
    "stage2-ingredient3-step11000-tokens47B"
    "stage2-ingredient2-step11000-tokens47B"
    "stage2-ingredient1-step11000-tokens47B"
    "stage2-ingredient3-step11931-tokens50B"
    "stage2-ingredient2-step11931-tokens50B"
    "stage2-ingredient1-step11931-tokens50B"
)

REVISIONS=(
    "stage1-step150-tokens1B"
    "stage1-step15000-tokens63B"
)
# Batch size (number of revisions per batch)
BATCH_SIZE=10

# Checkpoint file to track progress
CHECKPOINT_FILE="$OUTPUT_DIR/batch_checkpoint.txt"

# Get the starting batch from checkpoint if it exists
START_BATCH=0
if [ -f "$CHECKPOINT_FILE" ]; then
    START_BATCH=$(cat "$CHECKPOINT_FILE")
    echo "Resuming from batch $START_BATCH"
fi

# Calculate total number of batches
TOTAL_REVISIONS=${#REVISIONS[@]}
TOTAL_BATCHES=$(( (TOTAL_REVISIONS + BATCH_SIZE - 1) / BATCH_SIZE ))

# Loop through each batch
model="allenai/OLMo-2-1124-7B"
for ((batch=START_BATCH; batch<TOTAL_BATCHES; batch++)); do
    start=$((batch * BATCH_SIZE))
    end=$((start + BATCH_SIZE - 1))
    
    echo "Processing batch $((batch + 1)) of $TOTAL_BATCHES"
    process_batch $start $end $((batch + 1))
    
    # Save checkpoint
    echo $((batch + 1)) > "$CHECKPOINT_FILE"
done

# Remove checkpoint file after successful completion
rm -f "$CHECKPOINT_FILE"

echo "All evaluations complete!"
echo "Results are saved in $OUTPUT_DIR"

# Create a summary of results
echo "Creating summary of results..."
PYTHON_REVISIONS=$(printf "'%s', " "${REVISIONS[@]}" | sed 's/, $//')

python - <<EOF
import json
import os
import pandas as pd

output_dir = "$OUTPUT_DIR"
model = "$model"
revisions = [$PYTHON_REVISIONS]

results = []
for revision in revisions:
    try:
        result_file = os.path.join(output_dir, f"cruxeval_results_{model}_{revision}.json")
        if os.path.exists(result_file):
            with open(result_file, "r") as f:
                data = json.load(f)
            
            correct_count = sum(1 for r in data if r["is_correct"])
            total_count = len(data)
            accuracy = correct_count / total_count if total_count > 0 else 0
            
            results.append({
                "model": model,
                "revision": revision,
                "correct": correct_count,
                "total": total_count,
                "accuracy": accuracy
            })
        else:
            print(f"Results file for {model}, revision {revision} not found: {result_file}")
    except Exception as e:
        print(f"Error processing results for {model}, revision {revision}: {str(e)}")

if results:
    df = pd.DataFrame(results)
    df["accuracy_pct"] = df["accuracy"] * 100
    
    # Save as CSV
    csv_path = os.path.join(output_dir, f"{model}_summary.csv")
    df.to_csv(csv_path, index=False)
    
    # Print summary
    print("\nSummary of Results:")
    print(df[["model", "revision", "correct", "total", "accuracy_pct"]].to_string(index=False))
    print(f"\nSummary saved to {csv_path}")
else:
    print("No results found to summarize")
EOF

echo "Done!" 