#%%
from huggingface_hub import list_repo_refs

repo = "allenai/OLMo-2-1124-7B-DPO"
refs = list_repo_refs(repo)
branches = [ref.name for ref in refs.branches]
print(branches)
# %%
from huggingface_hub import list_repo_refs
out = list_repo_refs("allenai/OLMo-2-1124-7B")
branches = [b.name for b in out.branches]
print(branches)
#%%
from huggingface_hub import list_repo_refs
out = list_repo_refs("allenai/OLMo-2-1124-7B-Instruct")
branches = [b.name for b in out.branches]
print(branches)

#%%
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "allenai/OLMo-2-1124-7B-Instruct"
revision = "step_360"
tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)
