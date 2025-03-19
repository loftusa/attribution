#!/bin/bash

# Create output directory if it doesn't exist
OUTPUT_DIR="../data/cruxeval_results"
mkdir -p $OUTPUT_DIR

REVISIONS=(
    "step_360" "step_300" "step_240" "step_180" "step_120" "step_60" "main"
)

# Loop through each revision and run the evaluation
echo "=====================TEST RUN: ONLY ONE PROBLEM!!!================================================"
model="allenai/OLMo-2-1124-7B-Instruct"
for revision in "${REVISIONS[@]}"; do
    echo "====================================================="
    echo "Evaluating revision: $revision"
    echo "====================================================="
    
    # Run the evaluation script with the current model
    python 07_evaluate_cruxeval.py \
        --model "$model" \
        --revision "$revision" \
        --num_problems 1 \
        --batch_size 1 \
        --output_dir "$OUTPUT_DIR"
    
    # Check if the evaluation was successful
    if [ $? -eq 0 ]; then
        echo "Completed evaluation for $model"
    else
        echo "Error evaluating $model"
    fi
    
    echo "====================================================="
    echo ""
    
    # Clear CUDA cache between models
    python -c "import torch; torch.cuda.empty_cache()"
    
done

echo "All evaluations complete!"
echo "Results are saved in $OUTPUT_DIR"

# Create a summary of results
echo "Creating summary of results..."
python - <<EOF
import json
import os
import pandas as pd

output_dir = "$OUTPUT_DIR"
models = ["base", "sft", "dpo", "instruct", "rm"]

results = []
for model in models:
    try:
        result_file = os.path.join(output_dir, f"cruxeval_results_{model}.json")
        if os.path.exists(result_file):
            with open(result_file, "r") as f:
                data = json.load(f)
            
            correct_count = sum(1 for r in data if r["is_correct"])
            total_count = len(data)
            accuracy = correct_count / total_count if total_count > 0 else 0
            
            results.append({
                "model": model,
                "correct": correct_count,
                "total": total_count,
                "accuracy": accuracy
            })
        else:
            print(f"Results file for {model} not found: {result_file}")
    except Exception as e:
        print(f"Error processing results for {model}: {str(e)}")

if results:
    df = pd.DataFrame(results)
    df["accuracy_pct"] = df["accuracy"] * 100
    
    # Save as CSV
    csv_path = os.path.join(output_dir, "summary.csv")
    df.to_csv(csv_path, index=False)
    
    # Print summary
    print("\nSummary of Results:")
    print(df[["model", "correct", "total", "accuracy_pct"]].to_string(index=False))
    print(f"\nSummary saved to {csv_path}")
else:
    print("No results found to summarize")
EOF

echo "Done!" 