# Define the revisions array
REVISIONS=(
  "r-multiplt-epoch1"
  "r-multiplt-epoch2"
  "r-multiplt-epoch3"
  "r-multiplt-epoch4"
  "r-multiplt-epoch5"
  "r-multiplt-epoch6"
)

# Common variables
MODEL="nuprl/MultiPL-T-StarCoderBase_15b"
TASK="multiple-r"
TEMPERATURE="0.8"
ALLOW_CODE_EXECUTION="--allow_code_execution"
SAVE_LOCATION="/home/lofty/code_llm/attribution/results/nuprl/MultiPL-T-StarCoderBase_15b_R_QUESTIONS_ATTEMPT_4"
N_SAMPLES="200"  # number of generations for each problem
LIMIT="200"  # number of problems
BATCH_SIZE="8"

# First process the base model (no revision)
echo "Processing base model"
GENERATIONS_PATH="$SAVE_LOCATION/COMMIT_BASE"
mkdir -p $GENERATIONS_PATH
cd $GENERATIONS_PATH

# Run accelerate command for base model
accelerate launch /home/lofty/code_llm/bigcode-evaluation-harness/main.py \
  --model $MODEL \
  --tasks $TASK \
  --max_length_generation 650 \
  --temperature $TEMPERATURE \
  --do_sample True \
  --batch_size $BATCH_SIZE \
  --trust_remote_code \
  --generation_only \
  $ALLOW_CODE_EXECUTION \
  --save_generations \
  --save_generations_path generations.json \
  --precision bf16 \
  --limit $LIMIT \
  --n_samples $N_SAMPLES

# Run docker command for base model
sudo -E docker run -v $(pwd):/app/generations -it evaluation-harness-multiple \
  --model $MODEL \
  --tasks $TASK \
  --load_generations_path /app/generations/generations_multiple-r.json \
  --allow_code_execution \
  --temperature $TEMPERATURE \
  --metric_output_path /app/generations/evaluation_results.json \
  --save_every_k_tasks 1 \
  --precision bf16 \
  --n_samples $N_SAMPLES \
  --limit $LIMIT

echo "Completed base model"

# Loop through each revision
for REVISION in "${REVISIONS[@]}"; do
  echo "Processing revision: $REVISION"
  
  # Create folder for this revision
  GENERATIONS_PATH="$SAVE_LOCATION/COMMIT_${REVISION##*epoch}"
  mkdir -p $GENERATIONS_PATH
  cd $GENERATIONS_PATH

  # Run accelerate command
  accelerate launch /home/lofty/code_llm/bigcode-evaluation-harness/main.py \
    --model $MODEL \
    --tasks $TASK \
    --max_length_generation 650 \
    --temperature $TEMPERATURE \
    --do_sample True \
    --batch_size $BATCH_SIZE \
    --trust_remote_code \
    --generation_only \
    $ALLOW_CODE_EXECUTION \
    --save_generations \
    --save_generations_path generations.json \
    --precision bf16 \
    --revision $REVISION \
    --limit $LIMIT \
    --n_samples $N_SAMPLES

  # Run docker command
  sudo -E docker run -v $(pwd):/app/generations -it evaluation-harness-multiple \
    --model $MODEL \
    --tasks $TASK \
    --load_generations_path /app/generations/generations_multiple-r.json \
    --allow_code_execution \
    --temperature $TEMPERATURE \
    --metric_output_path /app/generations/evaluation_results.json \
    --save_every_k_tasks 1 \
    --precision bf16 \
    --revision $REVISION \
    --n_samples $N_SAMPLES \
    --limit $LIMIT

  echo "Completed revision: $REVISION"
done
