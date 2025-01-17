#!/bin/bash

# Directory setup
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODEL_NAME="nuprl/MultiPL-T-StarCoderBase_15b"
# for COMMIT in {2..6}; do
RESULTS_PATH="/home/locallofty/code_llm/attribution/results/nuprl/MultiPL-T-StarCoderBase_15b_R_QUESTIONS_ATTEMPT_2/COMMIT_6"

# Run MultiPL-E evaluation
# echo "Evaluating results in ${RESULTS_PATH}"
# docker run --rm \
#   --network none \
#   -v "${RESULTS_PATH}:/out:rw" \
#   ghcr.io/nuprl/multipl-e-evaluation \
#   --dir /out \
#   --output-dir /out

# Run pass@k analysis for each commit
for COMMIT in {1..6}; do
  echo "Running pass@k analysis for COMMIT_${COMMIT}"
  RESULTS_PATH="/home/locallofty/code_llm/attribution/results/nuprl/MultiPL-T-StarCoderBase_15b_R_QUESTIONS_ATTEMPT_2/COMMIT_${COMMIT}"
  python /home/locallofty/code_llm/MultiPL-E/pass_k.py "${RESULTS_PATH}" -k 100
  echo "------------------------"
done
# done