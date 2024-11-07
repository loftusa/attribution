import subprocess


args = """accelerate launch  ../bigcode-evaluation-harness/main.py \
  --model NTQAI/Nxcode-CQ-7B-orpo \
  --tasks humanevalplus \
  --limit 100 \
  --max_length_generation 512 \
  --temperature 0.2 \
  --do_sample True \
  --n_samples 200 \
  --batch_size 10 \
  --precision bf16 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path results/generations_humanevalplus.json
"""

def benchmark(model_name="NTQAI/Nxcode-CQ-7B-orpo"):
  command = args.replace("NTQAI/Nxcode-CQ-7B-orpo", model_name)
  
  # Run the command with updated model name
  result = subprocess.run(command, shell=True, capture_output=True, text=True)
  print(result.stdout)
  if result.stderr:
    print("Error:", result.stderr)

if __name__ == "__main__":
  benchmark()
