#%%
import gzip
import os
import re
import json
from fastcore.all import Path
import polars as pl
from typing import List, Literal
from copy import deepcopy

from transformers import AutoTokenizer
from nnsight import LanguageModel
from torchtyping import TensorType
from dataclasses import dataclass
from functools import partial
import torch
from datasets import load_dataset


PROMPT_TEMPLATE = '''\
Given a function and an input, provide ONLY the output value that would result from executing the function with the given input.
function: {code}
input: {input}
output: {output}'''

class CruxEvalUtil:
    def __init__(self, df: pl.DataFrame = None):
        if df is None:
            dataset = load_dataset("cruxeval-org/cruxeval")
            df = dataset['test'].to_polars()
            self.dataset = dataset
        self.df = df
        self.prompt = PROMPT_TEMPLATE

    def get_problem(self, problem_id: int) -> tuple[str, str]:
        """
        Get the problem with the given id.
        """
        return self.df.filter(pl.col('id') == f"sample_{problem_id}")


    def output_full(self, problem_id: int) -> tuple[str, str, str]:
        """
        taking id as input and using the polars df already loaded, output both the prompt, true input, and true output as a tuple.

        Returns:
            tuple[str, str, str]: A tuple containing (prompt, true_input, true_output)
            
        The prompt format is:
        '''
        function: {function}
        input: {true_input}
        output: {true_output}
        '''

        examples:

        (
        ''' 
        function: def f(text):
            new_text = list(text)
            for i in '+':
                if i in new_text:
                    new_text.remove(i)
            return ''.join(new_text)

        input: 'hbtofdeiequ' 
        output: 
        ''' 
        ), ('hbtofdeiequ', 'hbtofdeiequ')

        """ 
        # get the prompt and true output
        row = self.df.filter(pl.col('id') == f"sample_{problem_id}")
        code = row.select('code').item()
        true_input = row.select('input').item()
        true_output = row.select('output').item()
        
        # Create a fixed prompt template for this problem
        # Replace named placeholders with actual content, but leave {} for input and output
        fixed_prompt = PROMPT_TEMPLATE.replace('{code}', code)
        
        return fixed_prompt, true_input, true_output


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


STDEV = 0.094

@dataclass
class CausalTracingInput:
    prompt: TensorType["seq"]
    subject_idxs: TensorType["seq"] 
    target_idxs: TensorType["seq"]

def causal_trace(model, cfg: CausalTracingInput, n_iters: int = 5):
    # Arange prompts for token-wise interventions
    n_toks = len(cfg.prompt)
    n_layers = len(model.model.layers)
    batch = cfg.prompt.repeat(n_toks, 1)

    # Declare envoys
    mlps = [layer.mlp for layer in model.model.layers]
    model_in = model.model.embed_tokens

    def _window(layer, n_layers, window_size):
        return max(0, layer - window_size), min(n_layers, layer + window_size + 1)

    window = partial(_window, n_layers=n_layers, window_size=4)

    # Create noise
    noise = torch.randn(1, len(cfg.subject_idxs), 1600) * STDEV

    # Initialize results
    results = torch.zeros((len(model.model.layers), n_toks), device=model.device)

    for _ in range(n_iters):
        with torch.no_grad():
            with model.trace(cfg.prompt):
                clean_states = [
                    mlps[layer_idx].output.cpu().save() for layer_idx in range(n_layers)
                ]

            clean_states = torch.cat(clean_states, dim=0)

            with model.trace(cfg.prompt):
                model_in.output[:, cfg.subject_idxs] += noise

                corr_logits = model.lm_head.output.softmax(-1)[
                    :, -1, cfg.target_id
                ].save()

            for layer_idx in range(n_layers):
                s, e = window(layer_idx)
                with model.trace(batch):
                    model_in.output[:, cfg.subject_idxs] += noise

                    for token_idx in range(n_toks):
                        s, e = window(layer_idx)
                        for l in range(s, e):
                            mlps[l].output[token_idx, token_idx, :] = clean_states[
                                l, token_idx, :
                            ]

                    restored_logits = model.lm_head.output.softmax(-1)[
                        :, -1, cfg.target_id
                    ]

                    diff = restored_logits - corr_logits

                    diff.save()

                results[layer_idx, :] += diff.value

    results = results.detach().cpu()

    return results

def sample_k(
    model: LanguageModel, 
    tokenizer: AutoTokenizer,
    n_prompts: int,
    **generation_kwargs
) -> List[str]:
    
    batch = ['<|endoftext|>'] * n_prompts
    with model.generate(batch, **generation_kwargs):

        results = model.generator.output.save()

    # Return everything after <|endoftext|>
    samples = tokenizer.batch_decode(results[:,1:])
    return [sample + ". {}" for sample in samples]


def format_template(
    tok: AutoTokenizer, 
    context_templates: List[str], 
    words: str, 
    subtoken: Literal["last", "all"] = "last", 
) -> int:
    """
    Given list of template strings, each with *one* format specifier
    (e.g. "{} plays basketball"), and words to be substituted into the
    template, computes the post-tokenization index of their last tokens.
    """

    assert all(
        tmp.count("{}") == 1 for tmp in context_templates
    ), "Multiple fill-ins not supported."

    # assert subtoken == "last", "Only last token retrieval supported."

    # Compute prefixes and suffixes of the tokenized context
    prefixes, suffixes = _split_templates(context_templates)
    _words = deepcopy(words)

    # Compute lengths of prefixes, words, and suffixes
    prefixes_len, words_len, suffixes_len = \
        _get_split_lengths(tok, prefixes, _words, suffixes)
    
    # Format the prompts bc why not
    input_tok = tok(
        [
            template.format(word)
            for template, word in zip(context_templates, words)
        ],
        return_tensors="pt",
        padding=True
    )

    size = input_tok['input_ids'].size(1)
    padding_side = tok.padding_side


    if subtoken == "all":

        word_idxs = [
            [
                prefixes_len[i] + _word_len
                for _word_len in range(words_len[i])
            ]
            for i in range(len(prefixes))
        ]

        return input_tok, word_idxs

    # Compute indices of last tokens
    elif padding_side == "right":

        word_idxs = [
            prefixes_len[i] + words_len[i] - 1
            for i in range(len(prefixes))
        ]

        return input_tok, word_idxs
    
    elif padding_side == "left":

        word_idxs = [
            size - suffixes_len[i] - 1
            for i in range(len(prefixes))
        ]

        return input_tok, word_idxs
    
def _get_split_lengths(tok, prefixes, words, suffixes):
    # Pre-process tokens to account for different 
    # tokenization strategies
    for i, prefix in enumerate(prefixes):
        if len(prefix) > 0:
            assert prefix[-1] == " "
            prefix = prefix[:-1]

            prefixes[i] = prefix
            words[i] = f" {words[i].strip()}"

    # Tokenize to determine lengths
    assert len(prefixes) == len(words) == len(suffixes)
    n = len(prefixes)
    batch_tok = tok([*prefixes, *words, *suffixes])

    prefixes_tok, words_tok, suffixes_tok = [
        batch_tok[i : i + n] for i in range(0, n * 3, n)
    ]
    prefixes_len, words_len, suffixes_len = [
        [len(el) for el in tok_list]
        for tok_list in [prefixes_tok, words_tok, suffixes_tok]
    ]

    return prefixes_len, words_len, suffixes_len


def _split_templates(context_templates):
    # Compute prefixes and suffixes of the tokenized context
    fill_idxs = [tmp.index("{}") for tmp in context_templates]
    prefixes = [tmp[: fill_idxs[i]] for i, tmp in enumerate(context_templates)]
    suffixes = [tmp[fill_idxs[i] + 2 :] for i, tmp in enumerate(context_templates)]

    return prefixes, suffixes

def main():
    input_file = 'garbled_file.json'
    
    with open(input_file, 'r') as file:
        text = file.read()
    
    snippets = extract_code_snippets_from_text(text)
    
    save_snippets_to_files(snippets)
    
    print(f"Extracted {len(snippets)} code snippets and saved to {os.path.abspath('output_snippets')}")

if __name__ == "__main__":
    main()
