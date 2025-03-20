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
    # "stage1-step150-tokens1B"
    # "stage1-step15000-tokens63B"
    # "stage1-step35000-tokens147B"
    # "stage1-step55000-tokens231B"
    # "stage1-step75000-tokens315B"
    # "stage1-step95000-tokens399B"
    # "stage1-step116000-tokens487B"
    # "stage1-step136000-tokens571B"
    # "stage1-step156000-tokens655B"
    # "stage1-step177000-tokens743B"
    # "stage1-step197000-tokens827B"
    # "stage1-step217000-tokens911B"
    # "stage1-step237000-tokens995B"
    # "stage1-step257000-tokens1078B"
    # "stage1-step277000-tokens1162B"
    # "stage1-step297000-tokens1246B"
    # "stage1-step318000-tokens1334B"
    # "stage1-step338000-tokens1418B"
    # "stage1-step358000-tokens1502B"
    # "stage1-step378000-tokens1586B"
    # "stage1-step398000-tokens1670B"
    # "stage1-step418000-tokens1754B"
    # "stage1-step438000-tokens1838B"
    # "stage1-step458000-tokens1921B"
    # "stage1-step478000-tokens2005B"
    # "stage1-step498000-tokens2089B"
    # "stage1-step518000-tokens2173B"
    # "stage1-step539000-tokens2261B"
    # "stage1-step559000-tokens2345B"
    # "stage1-step579000-tokens2429B"
    # "stage1-step599000-tokens2513B"
    # "stage1-step620000-tokens2601B"
    # "stage1-step641000-tokens2689B"
    # "stage1-step661000-tokens2773B"
    # "stage1-step682000-tokens2861B"
    # "stage1-step702000-tokens2945B"
    # "stage1-step722000-tokens3029B"
    # "stage1-step742000-tokens3113B"
    # "stage1-step762000-tokens3197B"
    # "stage1-step782000-tokens3280B"
    # "stage1-step802000-tokens3364B"
    # "stage1-step822000-tokens3448B"
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

# Convert array to Python list string
PYTHON_REVISIONS=$(printf "'%s', " "${REVISIONS[@]}" | sed 's/, $//')

python - <<EOF
import json
import os
import pandas as pd
from pathlib import Path
import re

output_dir = "$OUTPUT_DIR"
model = "$model"
revisions = [$PYTHON_REVISIONS]

results = []
for revision in revisions:
    try:
        # Look for results in revision-specific directory
        revision_dir = os.path.join(output_dir, revision)
        result_file = os.path.join(revision_dir, f"cruxeval_results_{model}_{revision}.json")
        
        if os.path.exists(result_file):
            with open(result_file) as f:
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
            print(f"Results file not found: {result_file}")
    except Exception as e:
        print(f"Error processing results for {revision}: {str(e)}")

if results:
    df = pd.DataFrame(results)
    df["accuracy_pct"] = df["accuracy"] * 100
    
    # Extract stage and tokens for sorting
    def extract_stage_and_tokens(revision):
        stage_match = re.search(r'stage(\d+)', revision)
        tokens_match = re.search(r'tokens(\d+)B', revision)
        ingredient_match = re.search(r'ingredient(\d+)', revision)
        
        stage = int(stage_match.group(1)) if stage_match else 0
        tokens = float(tokens_match.group(1)) if tokens_match else 0
        ingredient = int(ingredient_match.group(1)) if ingredient_match else 0
        
        return stage, ingredient, tokens
    
    # Add sorting columns
    df[["stage", "ingredient", "tokens"]] = pd.DataFrame(
        [extract_stage_and_tokens(r) for r in df["revision"]], 
        index=df.index
    )
    
    # Sort hierarchically
    df = df.sort_values(["stage", "ingredient", "tokens"])
    
    # Save as CSV
    csv_path = os.path.join(output_dir, "summary.csv")
    df.to_csv(csv_path, index=False)
    
    # Print summary
    print("\nSummary of Results:")
    summary_cols = ["model", "revision", "correct", "total", "accuracy_pct"]
    print(df[summary_cols].to_string(index=False))
    print(f"\nSummary saved to {csv_path}")
    
    # Create plots
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(df["tokens"], df["accuracy_pct"], marker="o")
        plt.xlabel("Training Tokens (Billions)")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Model Performance vs Training Tokens by Stage\n{Path(model).name}")
        plt.grid(True)
        plt.legend()
        
        # Save plot
        plot_path = os.path.join(output_dir, "performance_by_stage.png")
        plt.savefig(plot_path)
        print(f"\nPerformance plot saved to {plot_path}")
        
        # Plot 2: Performance by stage and ingredient
        if df["ingredient"].max() > 0:  # Only create if we have ingredient data
            plt.figure(figsize=(15, 8))
            for stage in sorted(df["stage"].unique()):
                stage_data = df[df["stage"] == stage]
                for ingredient in sorted(stage_data["ingredient"].unique()):
                    if ingredient > 0:  # Skip non-ingredient data
                        data = stage_data[stage_data["ingredient"] == ingredient]
                        label = f"Stage {stage} Ingredient {ingredient}"
                        plt.plot(data["tokens"], data["accuracy_pct"], 
                                marker="o", label=label)
            
            plt.xlabel("Training Tokens (Billions)")
            plt.ylabel("Accuracy (%)")
            plt.title(f"Model Performance vs Training Tokens by Stage and Ingredient\n{Path(model).name}")
            plt.grid(True)
            plt.legend()
            
            # Save plot
            plot_path = os.path.join(output_dir, "performance_by_ingredient.png")
            plt.savefig(plot_path)
            print(f"\nIngredient performance plot saved to {plot_path}")
        
    except Exception as e:
        print(f"\nError creating plots: {str(e)}")
else:
    print("No results found to summarize")
EOF

echo "Done!" 