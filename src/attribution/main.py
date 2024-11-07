#%%
from transformers import AutoModelForCausalLM, AutoTokenizer
import lovely_tensors as lt; lt.monkey_patch()
from benchmark import benchmark
from tune import tune

prompt = """
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
"""
messages = [
    {"role": "user", "content": prompt}
]

def load_model(name="NTQAI/Nxcode-CQ-7B-orpo"):
  model = AutoModelForCausalLM.from_pretrained(name, torch_dtype='auto', device_map='auto')
  tokenizer = AutoTokenizer.from_pretrained(name)
  return model, tokenizer


def main():
  # load model and tokenizer
  name = "allenai/OLMo-1B-0724-hf"
  model, tok = load_model(name)

  # run generation
  inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
  outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tok.eos_token_id)
  res = tok.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
  print(res)

  # benchmark before tuning
  print("Benchmarking before tuning")
  initial_metrics = benchmark(model, tok)

  # run tuning
  print("Running tuning")
  tuned_model = tune(model, tok)

  # benchmark after tuning
  print("Benchmarking after tuning")
  final_metrics = benchmark(tuned_model, tok)
  

if __name__ == "__main__":
    main()
