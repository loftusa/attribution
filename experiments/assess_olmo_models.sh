#!/bin/bash

# Array of models to evaluate
MODELS=(
    "allenai/OLMo2-7B-1124"
    "allenai/OLMo-2-1124-7B-SFT"
    "allenai/OLMo-2-1124-7B-DPO"
    "allenai/OLMo-2-1124-7B-Instruct"
    "allenai/OLMo-2-1124-7B-RM"
)

# Loop through each model and run the assessment
for model in "${MODELS[@]}"; do
    echo "Evaluating model: $model"
    echo "----------------------------------------"
    
    uv run 04_assess_multiple_benchmarks.py \
        --model "$model" \
        --generate_new
    
    echo "Completed evaluation for $model"
    echo "----------------------------------------"
done

echo "All evaluations complete!" 