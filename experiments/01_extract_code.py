from attribution.utils import extract_code_snippets_from_text
from fastcore.all import Path

text = Path("results/allenai/OLMo-7B-0724-Instruct-hf/evaluation_results.json").read_text()
snippets = extract_code_snippets_from_text(text)

print(snippets)