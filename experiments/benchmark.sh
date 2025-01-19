if [ -z "${HUGGING_FACE_HUB_TOKEN}" ]; then
  echo "Error: HUGGING_FACE_HUB_TOKEN environment variable is not set"
  echo "Please set it with: export HUGGING_FACE_HUB_TOKEN='your_token'"
  exit 1
fi

# Create directories if they don't exist
mkdir -p results/${MODEL_NAME}

# Common variables
MODEL="allenai/OLMo-2-1124-7B"
TASK="humanevalplus"
TEMPERATURE="0.2"
N_SAMPLES="10"  # number of generations for each problem
LIMIT="10"  # number of problems
BATCH_SIZE="8"

accelerate launch  ../bigcode-evaluation-harness/main.py \
  --model ${MODEL} \
  --tasks ${TASK} \
  --max_length_generation 650 \
  --temperature ${TEMPERATURE} \
  --do_sample True \
  --n_samples ${N_SAMPLES} \
  --batch_size ${BATCH_SIZE} \
  --allow_code_execution \
  --save_generations \
  --save_generations_path "results/${MODEL_NAME}/${TASK}_generations.json" \
  --save_references \
  --save_references_path "results/${MODEL_NAME}/${TASK}_references.json" \
  --limit ${LIMIT} \
  --save_every_k_tasks 10 \
  --metric_output_path "results/${MODEL_NAME}/${TASK}_evaluation_results.json" \
  --use_auth_token \
  --trust_remote_code \