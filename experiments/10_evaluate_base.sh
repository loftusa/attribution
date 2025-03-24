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

# Function to clear all current blobs
clear_blobs() {
    echo "Clearing all current blobs..."
    rm -rf ~/.cache/huggingface/hub/models--allenai--OLMo-2-1124-7B*/blobs/*
    echo "Blobs cleared"
}

# Function to process a batch of revisions
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
            --num_problems 800 \
            --batch_size 512 \
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
    clear_blobs
}

REVISIONS=(
    "stage2-ingredient3-step11000-tokens47B"
    "stage2-ingredient3-step10000-tokens42B"
    "stage2-ingredient3-step9000-tokens38B"
    "stage2-ingredient3-step8000-tokens34B"
    "stage2-ingredient3-step7000-tokens30B"
    "stage2-ingredient3-step6000-tokens26B"
    "stage2-ingredient3-step5000-tokens21B"
    "stage2-ingredient3-step4000-tokens17B"
    "stage2-ingredient3-step3000-tokens13B"
    "stage2-ingredient3-step2000-tokens9B"
    "stage2-ingredient3-step1000-tokens5B"
    "stage2-ingredient2-step1000-tokens5B"
    "stage2-ingredient2-step2000-tokens9B"
    "stage2-ingredient2-step3000-tokens13B"
    "stage2-ingredient2-step4000-tokens17B"
    "stage2-ingredient2-step5000-tokens21B"
    "stage2-ingredient2-step6000-tokens26B"
    "stage2-ingredient2-step7000-tokens30B"
    "stage2-ingredient2-step8000-tokens34B"
    "stage2-ingredient2-step9000-tokens38B"
    "stage2-ingredient2-step10000-tokens42B"
    "stage2-ingredient2-step11000-tokens47B"
    "stage2-ingredient1-step1000-tokens5B"
    "stage2-ingredient1-step2000-tokens9B"
    "stage2-ingredient1-step3000-tokens13B"
    "stage2-ingredient1-step4000-tokens17B"
    "stage2-ingredient1-step5000-tokens21B"
    "stage2-ingredient1-step6000-tokens26B"
    "stage2-ingredient1-step7000-tokens30B"
    "stage2-ingredient1-step8000-tokens34B"
    "stage2-ingredient1-step9000-tokens38B"
    "stage2-ingredient1-step10000-tokens42B"
    "stage2-ingredient1-step11000-tokens47B"
    "stage2-ingredient3-step11931-tokens50B"
    "stage2-ingredient2-step11931-tokens50B"
    "stage2-ingredient1-step11931-tokens50B"
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

echo "Done!" 