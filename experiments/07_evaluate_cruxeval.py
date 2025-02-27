#%% [markdown]

# Run causal tracing on {input, function, output} triplets.
# Use olmo's DPO model. 
# First, I need to find a function in CruxEval that follows the following two properties:
#  1. Olmo2's DPO model gets the correct output given the input. I will use the `guidance` library to build code that checks this. Filter down to all functions for which this is true.
#  2. There is an obvious output token that I can trace on with causal tracing. Given the filtering in 1, further filter here, and then run causal tracing.

# For the causal tracing, here are the steps:
#  1. grab a (prompt, input, output) triplet from the CruxEvalUtil class
#  2. run a forward pass on a CruxEval function that gets the correct output given the input. Save the activations.
#  3. find the set of input tokens and the set of output tokens


# 800 problems total in CruxEval. Examples of input/
#%%
# %load_ext autoreload
# %autoreload
from dataclasses import dataclass
import polars as pl
import os
import json
import torch
import einops
import plotly.express as px
from pathlib import Path
from nnsight import LanguageModel
from bigcode_eval.tasks import get_task
from bigcode_eval.tasks.cruxeval import CruxEval
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from tqdm import tqdm
import argparse
import sys

from attribution.utils import CruxEvalUtil, CausalTracingInput, causal_trace, format_template

from guidance import models, gen, guidance

# Parse command line arguments
parser = argparse.ArgumentParser(description='Evaluate CruxEval problems with a specified model')
parser.add_argument('--model', type=str, default="allenai/OLMo-2-1124-7B-DPO", 
                    help='Model to use for evaluation (default: allenai/OLMo-2-1124-7B-DPO)')
parser.add_argument('--num_problems', type=int, default=800,
                    help='Number of problems to evaluate (default: 800)')
parser.add_argument('--batch_size', type=int, default=8,
                    help='Batch size for evaluation (default: 8)')
parser.add_argument('--output_dir', type=str, default="../data",
                    help='Directory to save results (default: ../data)')

# Check if running as script or in interactive mode
if not sys.argv[0].endswith('ipykernel_launcher.py'):
    args = parser.parse_args()
else:
    # Default values for interactive mode
    args = parser.parse_args([])

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
# Clear CUDA memory if it's in use
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # Additional forceful memory cleanup
    with torch.no_grad():
        torch.cuda.synchronize()
    
    # Print available memory for debugging
    free_memory: float = torch.cuda.mem_get_info()[0] / 1024**3  # Convert to GB
    total_memory: float = torch.cuda.mem_get_info()[1] / 1024**3  # Convert to GB
    print(f"GPU memory: {free_memory:.2f}GB free / {total_memory:.2f}GB total")


# Model paths
MODELS = {
    "base": "allenai/OLMo2-7B-1124",
    "sft": "allenai/OLMo-2-1124-7B-SFT",
    "dpo": "allenai/OLMo-2-1124-7B-DPO",
    "instruct": "allenai/OLMo-2-1124-7B-Instruct",
    "rm": "allenai/OLMo-2-1124-7B-RM",
}

# Get model name from command line or use default
model_name = args.model
if model_name in MODELS:
    model_name = MODELS[model_name]

# Extract model short name for file naming
if '/' in model_name:
    model_short_name = model_name.split('/')[-1]
else:
    # If it's already a short name, use it directly
    model_short_name = model_name

# For predefined models, use the key for simplicity
for key, value in MODELS.items():
    if model_name == value:
        model_short_name = key
        break

# Initialize CruxEvalUtil and model
ce = CruxEvalUtil()

print(f"Loading model: {model_name}")

# Load tokenizer with optimized settings
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=True,  # Use fast tokenizer
    padding_side="left"  # Pad on the left for more efficient generation
)

# Initialize model with optimized settings
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Automatically distribute across available GPUs
    torch_dtype=torch.bfloat16,  # Use bfloat16 for faster computation
    low_cpu_mem_usage=True,  # Optimize CPU memory usage
    use_cache=True  # Enable KV cache for faster generation
)

# Set model to evaluation mode
model.eval()

# Print GPU memory usage after loading model
if torch.cuda.is_available():
    free_memory: float = torch.cuda.mem_get_info()[0] / 1024**3  # Convert to GB
    total_memory: float = torch.cuda.mem_get_info()[1] / 1024**3  # Convert to GB
    used_memory: float = total_memory - free_memory
    print(f"GPU memory after loading model: {used_memory:.2f}GB used / {total_memory:.2f}GB total ({free_memory:.2f}GB free)")

#%%
# Function to evaluate a single problem using standard Transformers
def evaluate_problem(model, tokenizer, problem_id: int, max_tokens: int = 100) -> tuple[bool, str, str]:
    """
    Evaluate a single CruxEval problem using standard Transformers.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        problem_id: The problem ID to evaluate
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Tuple of (is_correct, generated_output, true_output)
    """
    try:
        # Get problem
        prompt, true_in, true_out = ce.output_full(problem_id)
        
        # Format prompt with input but leave output empty
        formatted_prompt = prompt.format(true_in, "")
        
        # Tokenize the prompt
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Generate output with optimized settings
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,  # Deterministic generation
                num_beams=1,      # No beam search for speed
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,   # Use KV cache for faster generation
                early_stopping=True
            )
        
        # Get only the newly generated tokens
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        generated = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Check if output matches
        is_correct = generated.strip() == true_out.strip()
        
        # Only print detailed info for correct outputs or every 50th problem
        if is_correct or problem_id % 50 == 0:
            print(f"\nProblem ID: {problem_id}")
            print(f"Input: {repr(true_in)}")
            print(f"Generated: {generated}")
            print(f"Expected: {true_out}")
            print(f"Correct: {is_correct}")
        
        return is_correct, generated, true_out
    except Exception as e:
        print(f"Error evaluating problem {problem_id}: {str(e)}")
        return False, str(e), f"Error: {str(e)}"

#%%
# Evaluate all problems with batch processing for better performance
num_problems = args.num_problems
batch_size = args.batch_size  # Process multiple problems at once
results = []
checkpoint_file = os.path.join(args.output_dir, f"cruxeval_checkpoint_{model_short_name}.json")

# Check if checkpoint exists to resume from previous run
try:
    with open(checkpoint_file, "r") as f:
        checkpoint_data = json.load(f)
        results = checkpoint_data["results"]
        start_idx = len(results)
        print(f"Resuming from checkpoint with {start_idx} problems already processed")
except FileNotFoundError:
    start_idx = 0
    print("Starting fresh evaluation")

print(f"Evaluating {num_problems} problems in batches of {batch_size}...")
start_time = time.time()

# Process problems in batches
for batch_start in tqdm(range(start_idx, num_problems, batch_size)):
    batch_end = min(batch_start + batch_size, num_problems)
    batch_results = []
    
    # Process each problem in the current batch
    for i in range(batch_start, batch_end):
        is_correct, generated, true_out = evaluate_problem(model, tokenizer, i)
        
        batch_results.append({
            "problem_id": i,
            "is_correct": is_correct,
            "generated": generated,
            "true_output": true_out
        })
    
    # Add batch results to overall results
    results.extend(batch_results)
    
    # Save checkpoint after each batch
    with open(checkpoint_file, "w") as f:
        json.dump({"results": results}, f)
    
    # Print progress after each batch
    processed_count = len(results)
    correct_count = sum(r["is_correct"] for r in results)
    print(f"Progress: {processed_count}/{num_problems}, Correct: {correct_count}/{processed_count} ({correct_count/processed_count*100:.2f}%)\n")
    
    # Calculate and print time estimates
    elapsed_time = time.time() - start_time
    problems_per_second = processed_count / elapsed_time
    remaining_problems = num_problems - processed_count
    estimated_remaining_time = remaining_problems / problems_per_second
    
    print(f"Speed: {problems_per_second:.2f} problems/second")
    print(f"Estimated time remaining: {estimated_remaining_time/60:.1f} minutes ({estimated_remaining_time/3600:.1f} hours)\n")

end_time = time.time()
total_time = end_time - start_time
print(f"\n\nEvaluation completed in {total_time:.2f} seconds ({total_time/60:.1f} minutes)")

# Calculate final accuracy
correct_count = sum(r["is_correct"] for r in results)
accuracy = correct_count / num_problems
print(f"Final accuracy: {correct_count}/{num_problems} ({accuracy*100:.2f}%)")

# Save final results
results_file = os.path.join(args.output_dir, f"cruxeval_results_{model_short_name}.json")
with open(results_file, "w") as f:
    json.dump(results, f)
print(f"Results saved to {results_file}")

# Find problems where the model gets the correct output
correct_problems = [r["problem_id"] for r in results if r["is_correct"]]
print(f"Found {len(correct_problems)} problems with correct outputs")

# These problems can be used for causal tracing
print("First 10 problems with correct outputs:")
for i in correct_problems[:10]:
    print(f"Problem {i}")

# Clean up checkpoint file after successful completion
if os.path.exists(checkpoint_file):
    os.remove(checkpoint_file)
    print(f"Removed checkpoint file: {checkpoint_file}")

# If running as a script, exit here
if not sys.argv[0].endswith('ipykernel_launcher.py'):
    sys.exit(0)

#%%
# The rest of your notebook remains unchanged for causal tracing
# When you're ready to use the correct problems for causal tracing,
# you can access them from the correct_problems list

# For example:
if len(correct_problems) > 0:
    problem_id = correct_problems[0]
    prompt, true_in, true_out = ce.output_full(problem_id)
    print(f"Using problem {problem_id} for causal tracing")
    print(f"Input: {true_in}")
    print(f"Expected output: {true_out}")
    
    # Your causal tracing code would go here
    # ...

# The remaining cells can be kept as they are
