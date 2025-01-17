import gzip
import os
import re
import json
from fastcore.all import Path

def load_results(path: Path) -> dict:
    """Load results from either .json or .json.gz file"""
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return json.load(f)
    else:
        return json.loads(path.read_text())

def extract_code_snippets_from_text(text):
    """
    Extracts code snippets from a text with a JSON-like structure.

    Args:
    - text (str): The input garbled text.

    Returns:
    - List of code snippets (list of str).
    """
    snippets = []
    
    # A regex pattern to find blocks of code in "code": [ " snippet_here " ] format.
    pattern = re.compile(r'"code":\s*\[\s*"(.*?)"\s*\]', re.DOTALL)
    
    matches = pattern.findall(text)
    for match in matches:
        # Replace escape sequences
        clean_snippet = match.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
        snippets.append(clean_snippet)

    return snippets

def save_snippets_to_files(snippets, output_dir='output_snippets'):
    """
    Saves each code snippet to a separate file.

    Args:
    - snippets (list of str): The list of code snippets.
    - output_dir (str): The directory where snippets will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for idx, snippet in enumerate(snippets, start=1):
        file_path = os.path.join(output_dir, f'snippet_{idx}.py')
        with open(file_path, 'w') as f:
            f.write(snippet)

def main():
    input_file = 'garbled_file.json'
    
    with open(input_file, 'r') as file:
        text = file.read()
    
    snippets = extract_code_snippets_from_text(text)
    
    save_snippets_to_files(snippets)
    
    print(f"Extracted {len(snippets)} code snippets and saved to {os.path.abspath('output_snippets')}")

if __name__ == "__main__":
    main()
