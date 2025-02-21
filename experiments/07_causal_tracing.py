#%%
%load_ext autoreload
%autoreload
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

from attribution.utils import CruxEvalUtil, CausalTracingInput, causal_trace, format_template

# Model paths
MODELS = {
    "base": "allenai/OLMo2-7B-1124",
    "dpo": "allenai/OLMo-2-1124-7B-DPO",
}

ce = CruxEval()
print(ce.output_full(0)[0])
# ce.df.filter(pl.col('id') == 'sample_0').select('code').item()
#%%
# Set up device and precision
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
model_name = MODELS['dpo']

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# Put model in eval mode
model.eval()

# Load CruxEval dataset
df = pl.read_ndjson('hf://datasets/cruxeval-org/cruxeval/test.jsonl')

# %%
prompt, true_out = ce.output_full(0)
# Run the model on the prompt
inputs = tokenizer(prompt, return_tensors="pt").to(device)
prompt_length = inputs.input_ids.shape[1]

with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

#%%
pln = prompt_length
print(tokenizer.decode(outputs[0][pln:], skip_special_tokens=True))

# %%
from typing import NamedTuple
from torchtyping import TensorType
from nnsight import LanguageModel
from functools import partial



# (subject, relation, object) --> (input, function, output)
# subject tokens should be the last token of the input string (newline token)
# target tokens should be the last token of the output string (newline token)

output_prompt = tokenizer.decode(outputs[0], skip_special_tokens=True)
subject_token = output_prompt.find('output:') -1

# the target is the last token of the output
target_token = len(output_prompt)-1

# CausalTracingInput(prompt=inputs.input_ids, subject_idx=torch.tensor([prompt_length]), target_id=torch.tensor([prompt_length + 1]))
# lm = LanguageModel(model, tokenizer=tokenizer)


#%%
for i in range(5):
    print(ce.output_full(i)[0])

#%%
from nnsight import LanguageModel
lm = LanguageModel(model, tokenizer=tokenizer)

prompt, true_out = ce.output_full(0)


format_template(lm.tokenizer, context_templates=[prompt], words=true_out, subtoken="last")

# cfg = CausalTracingInput(
#     prompt=lm.tokenizer.encode(prompt),
#     subject_idxs=torch.tensor([len(prompt)]),
#     target_idxs=torch.tensor([len(prompt) + 1])
# )

# results = causal_trace(model, cfg)


# %%
