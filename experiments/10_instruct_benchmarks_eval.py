#%%
from huggingface_hub import list_repo_refs
import pandas as pd

out = list_repo_refs("allenai/OLMo-2-1124-7B")
branches = [b.name for b in out.branches]
s = pd.Series(branches)

for b in branches:
    print(b)