import json
from pprint import pprint

def print_generations(generations, results, task, problem_index=None, prompt_only=False):
    # Convert results keys to strings for consistent access
    results = {str(k): v for k, v in results.items()}
    
    # Get dataset for prompts
    dataset = task.get_dataset()
    
    # Determine which problems to process
    problems = [generations[problem_index]] if problem_index is not None else generations
    problem_indices = [str(problem_index)] if problem_index is not None else [str(i) for i in range(len(generations))]
    
    # Process each problem
    for prob_idx, problem in zip(problem_indices, problems):
        if prompt_only:
            # Get prompt directly from dataset
            prompt = dataset['prompt'][int(prob_idx)]
            print(f'Problem {prob_idx} Prompt:')
            print(prompt)
            print('\n' + '='*100 + '\n')
            continue
            
        # Print full generation details
        for i, result in enumerate(problem):
            print(f'# Result {i+1}:')
            test_result = results[prob_idx][i][1]
            print(f'# Test result: {test_result["result"]}')
            print(f'# Passed: {test_result["passed"]}')
            
            # Print code with original formatting
            print(result)
            print('\n' + '='*100 + '\n')


def get_pass_rates(results):
    """Calculate pass rates for each problem in results"""
    pass_rates = []
    # Convert string keys to integers
    results = {int(k): v for k, v in results.items()}
    problem_ids = sorted(results.keys())
    for problem_id in problem_ids:
        problem_results = results[problem_id]
        # Count number of passed tests
        passed = sum(1 for _, result_dict in problem_results if result_dict['passed'])
        total = len(problem_results)
        pass_rate = (passed / total) * 100
        pass_rates.append(pass_rate)
    return problem_ids, pass_rates

def save_pass_rates(problem_ids, pass_rates, save_path):
    data = {
        'problem_ids': problem_ids,
        'pass_rates': pass_rates
    }
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_pass_rates(load_path):
    with open(load_path) as f:
        data = json.load(f)
    return data['problem_ids'], data['pass_rates']

def plot_pass_rates(results, save_path=None, model_name=''):
    import matplotlib.pyplot as plt
    import numpy as np
    
    problem_ids, pass_rates = get_pass_rates(results)
    
    # Create the plot
    plt.figure(figsize=(35, 5))
    # Use range(len(pass_rates)) for x-axis positioning
    plt.bar(range(len(pass_rates)), pass_rates)
    
    # Set integer ticks on x-axis with problem IDs as labels
    plt.xticks(range(len(pass_rates)), problem_ids)
    plt.margins(x=0.01)
    
    # Customize the plot
    plt.xlabel('Problem ID')
    plt.ylabel('Pass Rate (%)')
    plt.title(f'Pass Rate by Problem (model = {model_name})')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of each bar
    for i, rate in enumerate(pass_rates):
        plt.text(i, rate + 1, f'{rate:.1f}%', 
                ha='center', va='bottom')
    
    # Set y-axis limits to go slightly above 100%
    plt.ylim(0, max(105, max(pass_rates) + 5))
    
    # Adjust layout to remove empty space
    plt.tight_layout()
    
    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path)
        plt.close()
    
    return plt


  