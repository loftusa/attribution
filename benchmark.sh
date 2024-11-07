# Define variables with defaults that can be overridden
TASK=${TASK:-"humanevalplus"}
MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-2-7b"}

# Create directories if they don't exist
mkdir -p results/${MODEL_NAME}

accelerate launch  ../bigcode-evaluation-harness/main.py \
  --model ${MODEL_NAME} \
  --tasks ${TASK} \
  --max_length_generation 512 \
  --temperature 0.2 \
  --do_sample True \
  --n_samples 200 \
  --batch_size 10 \
  --precision bf16 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path "results/${MODEL_NAME}/${TASK}_generations.json" \
  --limit 100 \
  --save_every_k_tasks 10 \
  --metric_output_path "results/${MODEL_NAME}/${TASK}_evaluation_results.json"
