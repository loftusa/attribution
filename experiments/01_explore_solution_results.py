#%%
from tabnanny import check
from fastcore.all import Path
from pprint import pprint

from bigcode_eval.tasks.humanevalplus import create_task
from bigcode_eval.tasks.humaneval import GeneralHumanEval

def test_all_solutions(function_name, check_func):
  # Get all functions that start with "below_zero_llama_"
  solutions = [
    (name, func) for name, func in globals().items() 
    if name.startswith(function_name)
  ]
  
  for name, func in solutions:
    try:
      check_func(func)
      print(f"{name}: SUCCESS")
    except Exception as e:
      print(f"{name}: failed")
      # Optionally print the error:
      # print(f"  Error: {str(e)}")

# model = "allenai/OLMo-7B-0724-Instruct-hf"
model = "meta-llama/Llama-2-7b-hf"
path = Path(f"../results/{model}/humanevalplus_generations_humanevalplus.json")
results: list[list[str]] = path.read_json()


def get_sol(problem_idx:int, solution_idx:int):
  return results[problem_idx][solution_idx]

# doc = task.get_dataset()[0]

task = GeneralHumanEval(strip_prompt=True, k=[1, 10, 100], num_workers=16, timeout=20.0)
# pprint(task.get_dataset()[64]['prompt'])

pprint(results[64][0])
# tests = task.get_reference(doc)["tests"]
# print(tests)


#%% problem 64

def vowels_count_alex(s):
  """Write a function vowels_count which takes a string representing
  a word as input and returns the number of vowels in the string.
  Vowels in this case are 'a', 'e', 'i', 'o', 'u'. Here, 'y' is also a
  vowel, but only when it is at the end of the given word.

  Example:
  >>> vowels_count("abcde")
  2
  >>> vowels_count("ACEDY")
  3
  """
  from collections import Counter
  vowels = list("aeiou")
  s = s.lower()
  if s.endswith("y"):
    vowels += "y"

  c = Counter(s)
  return sum([c[val] for val in vowels])



# pprint(task.get_dataset()[64]['test'])

def check_64(candidate):
  # Check some simple cases
  assert candidate("abcde") == 2, "Test 1"
  assert candidate("Alone") == 3, "Test 2"
  assert candidate("key") == 2, "Test 3"
  assert candidate("bye") == 1, "Test 4"
  assert candidate("keY") == 2, "Test 5"
  assert candidate("bYe") == 1, "Test 6"
  assert candidate("ACEDY") == 3, "Test 7"

  # Check some edge cases that are easy to work out by hand.
  assert True, "This prints if this assert fails 2 (also good for debugging!)"

check_64(vowels_count_alex)

#%%
for i in range(20):
  pprint(get_sol(0,i))
  print()
  print()

#%%
from typing import List

# problem 0 in humanevalplus
# my own solution
def has_close_elements(numbers: list[float], threshold: float) -> bool:
  """Check if in given list of numbers, are any two numbers closer to each other than given threshold.
  >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
  False
  >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
  True
  """
  for i in numbers:
    for j in numbers:
      if i != j:
        if abs(i-j) < threshold:
          print(i,j)
          return True
  return False

# canonical solution
def has_close_elements_canonical(numbers: list[float], threshold: float) -> bool:
    """Check if in given list of numbers, are any two numbers closer to each other than given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
    for idx, elem in enumerate(numbers):
      for idx2, elem2 in enumerate(numbers):
        if idx != idx2:
          distance = abs(elem - elem2)
          if distance < threshold:
            return True
    return False

# test case from humanevalplus
def check0(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True
    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True
    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False

#%%
# Llama-2-7b-hf

# solution 20 llama2-7b-hf - incorrect - only looking at pairs
# this solution repeats 11 times in the first 20 results
def has_close_elements20(numbers: list[float], threshold: float) -> bool:
    return any(abs(x - y) < threshold for x, y in zip(numbers, numbers[1:]))

# solution 19 llama2-7b-hf - incorrect - also only looking at pairs
def has_close_elements19(numbers: List[float], threshold: float) -> bool:
  if len(numbers) < 2:
    return False

  for i in range(1, len(numbers)):
    if abs(numbers[i] - numbers[i - 1]) < threshold:
      return True

  return False

# incorrect - still only looking at pairs of numbers
def has_close_elements15(numbers: List[float], threshold: float) -> bool:
  return any(
    (
      abs(numbers[i] - numbers[i + 1])
      < threshold
      for i in range(len(numbers) - 1)
    )
  )


# check(has_close_elements20)
# check(has_close_elements19)
# check0(has_close_elements15)

#%%
# codellama-7b-python-hf
model = "codellama/CodeLlama-7b-Python-hf"
path = Path(f"../results/{model}/humanevalplus_generations_humanevalplus.json")
results: list[list[str]] = path.read_json()


def get_sol(problem_idx: int, solution_idx: int):
    return results[problem_idx][solution_idx]


for i in range(20):
    pprint(get_sol(0, i))
    print()
    print()

#%%
# correct *20
def has_close_elements20_codellama(numbers: List[float], threshold: float) -> bool:
  for i in range(len(numbers)):
    for j in range(i + 1, len(numbers)):
      if abs(numbers[i] - numbers[j]) < threshold:
        return True
  return False

check0(has_close_elements20_codellama)

#%%
# pprint(task.get_dataset()[1]["prompt"])
# pprint(task.get_dataset()[1]["test"])

from typing import List

def check_1(candidate):
  assert candidate('(()()) ((())) () ((())()())') == [
    '(()())', '((()))', '()', '((())()())'
  ]
  assert candidate('() (()) ((())) (((())))') == [
    '()', '(())', '((()))', '(((())))'
  ]
  assert candidate('(()(())((())))') == [
    '(()(())((())))'
  ]
  assert candidate('( ) (( )) (( )( ))') == ['()', '(())', '(()())']


def separate_paren_groups_alex(paren_string: str) -> List[str]:
  """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
  separate those group into separate strings and return the list of those.
  Separate groups are balanced (each open brace is properly closed) and not nested within each other
  Ignore any spaces in the input string.
  >>> separate_paren_groups('( ) (( )) (( )( ))')
  ['()', '(())', '(()())']
  """
  paren_string = paren_string.replace(' ', '')
  out = []
  current_str = []
  stopper = 0
  for chr in paren_string:
    if chr=='(':
      stopper += 1
    elif chr==')':
      stopper -= 1
    
    current_str.append(chr)
    if stopper==0:
      out.append(''.join(current_str))
      current_str = []
  return out

check_1(separate_paren_groups_alex)

def separate_paren_groups_canonical(paren_string: str) -> list[str]:
    """Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
    paren_string = paren_string.replace(" ", "")
    result = []
    current = []
    count = 0

    for char in paren_string:
        if char == "(":
            count += 1
        elif char == ")":
            count -= 1
        current.append(char)

        if count == 0 and current:
            result.append("".join(current))
            current = []

    return result

check_1(separate_paren_groups_canonical)
#%% # llama-7-2b solutions problem 2
model = "meta-llama/Llama-2-7b-hf"
# model = "codellama/CodeLlama-7b-Python-hf"
path = Path(f"../results/{model}/humanevalplus_generations_humanevalplus.json")
results: list[list[str]] = path.read_json()
for i in range(20):
    pprint(get_sol(1, i))
    print()
    print()
#%%
def separate_paren_groups_llama_20(paren_string: str) -> List[str]:
  paren_string = paren_string.replace(' ', '')
  paren_string = paren_string.replace('(', '(')
  paren_string = paren_string.replace(')', ')')
  paren_string = paren_string.replace('(', '(')
  paren_string = paren_string.replace(')', ')')
  paren_string = paren_string.replace('(', '(')
  paren_string = paren_string.replace(')', ')')
  paren_string = paren_string.replace('(', '(')
  paren_string = paren_string.replace(')', ')')
  paren_string = paren_string.replace('(', '(')
  paren_string = paren_string.replace(')', ')')
  paren_string = paren_string.replace('(', '(')
  paren_string = paren_string.replace(')', ')')
  paren_string = paren_string.replace('(', '(')
  paren_string = paren_string.replace(')', ')')
  paren_string = paren_string.replace('(', '(')
  paren_string = paren_string.replace(')', ')')
  paren_string = paren_string.replace('(', '(')
  paren_string = paren_string.replace(')', ')')
  paren_string = paren_string.replace('(', '(')
  paren_string = paren_string.replace(')', ')')


# Incorrect. Tried to append weird things, tried to add 
# to a str with item assignment, unecessary `enumerate`.
def separate_paren_groups_llama_18(paren_string: str) -> List[str]:
  paren_groups = []
  for i, c in enumerate(paren_string):
    if c == '(':
      paren_groups.append('')
    elif c == ')':
      paren_groups.append(''.join(paren_groups))
    else:
      paren_groups[-1] += c
  print(paren_groups)
  return paren_groups

# incorrect. Still trying to add to str objects, 
# not appending lists properly.
def separate_paren_groups_llama_17(paren_string: str) -> List[str]:
  paren_groups = []
  for i in range(len(paren_string)):
    if paren_string[i] == '(':
      paren_groups.append('')
    elif paren_string[i] == ')':
      paren_groups[-1] += ')'
    elif paren_string[i] == '(':
      paren_groups[-1] += '('
    elif paren_string[i] == ')':
      paren_groups[-1] += ')'
  return paren_groups


# incorrect understanding of `split`, and neglected '(' in inner groups.
def separate_paren_groups_llama_14(paren_string: str) -> List[str]:
  groups = paren_string.split('(')
  result = []
  for group in groups:
    if group.startswith('(') and group.endswith(')'):
      result.append(group)
    else:
      result.append(group)
  return result


# incorrect. Hallucinated { and [, and didn't solve the actual problem.
def separate_paren_groups_llama_12(paren_string: str) -> List[str]:
  paren_string = paren_string.strip()
  if not paren_string:
    return []

  stack = []
  for char in paren_string:
    if char == '(':
      stack.append('(')
    elif char == ')':
      if len(stack) == 0:
        return []
      else:
        stack.pop()
    elif char == '[':
      stack.append('[')
    elif char == ']':
      if len(stack) == 0:
        return []
      else:
        stack.pop()
    elif char == '{':
      stack.append('{')
    elif char == '}':
      if len(stack) == 0:
        return []
      else:
        stack.pop()
    else:
      pass

  return [str(stack)]

# incorrect. Just shows an incorrect understanding of `split`.
def separate_paren_groups_llama_11(paren_string: str) -> List[str]:
  paren_groups = paren_string.split(')')
  return [paren_group.split('(')[0] for paren_group in paren_groups]


# incorrect. Looks like just a random list comprehension, not much to do with the solution prompt.
def separate_paren_groups_llama_10(paren_string: str) -> List[str]:
  return [paren_string[i:i + 2] for i in range(len(paren_string) - 1)]


# check_1(separate_paren_groups_llama_12)

#%% codellama solutions problem 2
# model = "meta-llama/Llama-2-7b-hf"
model = "codellama/CodeLlama-7b-Python-hf"
path = Path(f"../results/{model}/humanevalplus_generations_humanevalplus.json")
results: list[list[str]] = path.read_json()


import re
from typing import List

def extract_function(code: str) -> str:
  pattern = r'def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*(?:->\s*[^:]+)?:\s*(?:[^#\n]*(?:\n\s+[^#\n]+)*)'
  match = re.search(pattern, code)
  if match:
    return match.group(0)
  return ""


for i in range(20):
    pprint(get_sol(1, i))
    print()
    print()


#%%
# incorrect x15
def separate_paren_groups_codellama_20(paren_string: str) -> List[str]:
  # TODO: implement your function here
  return []

def separate_paren_groups_codellama_19(paren_string: str) -> List[str]:
  # TODO: implement this function
  raise NotImplementedError


# first correct solution! Although a bit verbose.
def separate_paren_groups_codellama_17(paren_string: str) -> List[str]:
  # if there is no parentheses in the string, return the string
  if paren_string.find('(') == -1:
    return [paren_string]

  # initialize the list of strings
  paren_strings = []

  # initialize the current string
  current_string = ''

  # initialize the number of open parentheses
  num_open_paren = 0

  # loop through the string
  for char in paren_string:
    # if the character is an open parenthesis
    if char == '(':
      # increment the number of open parentheses
      num_open_paren += 1

      # append the character to the current string
      current_string += char

    # if the character is a close parenthesis
    elif char == ')':
      # decrement the number of open parentheses
      num_open_paren -= 1

      # append the character to the current string
      current_string += char

      # if the number of open parentheses is 0
      if num_open_paren == 0:
        # append the current string to the list of strings
        paren_strings.append(current_string)

        # reset the current string
        current_string = ''

    # if the character is a space
    elif char == ' ':
      # ignore the space
      pass

    # otherwise
    else:
      # append the character to the current string
      current_string += char

  # return the list of strings
  return paren_strings

check_1(separate_paren_groups_codellama_17)


# Incorrect. Not building gorups.
def separate_paren_groups_codellama_16(paren_string: str) -> List[str]:
  # initialize a stack
  stack = []
  # initialize a list to hold the results
  result = []
  # loop through each character in the string
  for char in paren_string:
    # if the character is an open brace, append it to the stack
    if char == '(':
      stack.append(char)
    # if the character is a close brace, pop the last open brace from the stack
    elif char == ')':
      stack.pop()
    # if the stack is empty, append the current character to the result list
    if not stack:
      result.append(char)
  return result

def separate_paren_groups_codellama_15(paren_string: str) -> List[str]:
    # The idea is to use a stack to keep track of the open parentheses and their indices.
    # Whenever we encounter a closing brace, we pop the last element from the stack and check if the
    # index of the opening brace is the same as the index of the closing brace. If so, we have found
    # a complete group. We then add the group to the list of groups.
    # If the index of the closing brace is not the same as the index of the opening brace, we add the
    # closing brace to the stack.
    # Whenever we encounter an opening brace, we add it to the stack.
    # Whenever we encounter a closing brace, we check if the index of the opening brace is the same as
    # the index of the closing brace. If so, we have found a complete group. We then add the group to
    # the list of groups.
    # If the index of the closing brace is not the same as the index of the opening brace, we add the
    # closing brace to the stack.
    # Whenever we encounter an opening brace, we add it to the stack.
    # Whenever we encounter a closing brace, we check if the index of the opening brace is the same as
    # the index of the closing brace. If so, we have found a complete group. We then add the group to
    # the list of groups.
    # If the index of the closing brace is not the same as the index of the opening brace, we add the
    # closing brace to the stack.
    # Whenever we encounter an opening brace, we add it
    pass

# didn't seem to understand the problem
def separate_paren_groups_codellama_13(paren_string: str) -> List[str]:
    # The idea is to use a stack to keep track of the opening parentheses
    # and then pop them off when we encounter a closing parentheses
    # If we encounter a closing parentheses before any opening parentheses,
    # then we know that the parentheses are unbalanced and we should return an empty list
    stack = []
    paren_groups = []
    for char in paren_string:
      if char == "(":
        stack.append(char)
      elif char == ")":
        if len(stack) == 0:
          return []
        stack.pop()
      else:
        continue
    # If we reach the end of the string and the stack is not empty, then we know that the parentheses are unbalanced
    if len(stack) != 0:
      return []
    # If we reach the end of the string and the stack is empty, then we know that the parentheses are balanced
    # and we can return the list of groups
    for i in range(len(paren_string)):
      if paren_string[i] == "(":
        paren_groups.append(paren_string[i:])
        break
    return paren_groups


# pretty close! But one logical error in the control flow
def separate_paren_groups_codellama_10(paren_string: str) -> List[str]:
    # Initialize the list to hold the separated groups
    separated_groups = []

    # Initialize the list to hold the current group
    current_group = []

    # Initialize the stack to hold the parentheses
    stack = []

    # Iterate through the string
    for char in paren_string:
        # If the character is an opening brace, append it to the stack and the current group
        if char == "(":
            stack.append(char)
            current_group.append(char)

        # If the character is a closing brace, pop the topmost opening brace from the stack
        # and append it to the current group
        elif char == ")":
            stack.pop()
            current_group.append(char)

        # If the character is a space, ignore it
        elif char == " ":
            continue

        # If the character is a non-parenthesis character, append it to the current group
        else:
            current_group.append(char)

        # If the stack is empty, append the current group to the separated groups and reset the current group
        if len(stack) == 0:
            separated_groups.append("".join(current_group))
            current_group = []

    return separated_groups

#%% problem 3

# pprint(task.get_dataset()[2]["prompt"])
# pprint(task.get_dataset()[2]["test"])
# pprint(task.get_dataset()[2]["canonical_solution"])

def check2(candidate):
  assert candidate(3.5) == 0.5
  assert abs(candidate(1.33) - 0.33) < 1e-6
  assert abs(candidate(123.456) - 0.456) < 1e-6

def truncate_number_alex(number: float) -> float:
  """ Given a positive floating point number, it can be decomposed into
  and integer part (largest integer smaller than given number) and decimals
  (leftover part always smaller than 1).

  Return the decimal part of the number.
  >>> truncate_number(3.5)
  0.5
  """
  return number - int(number)

def truncate_number_canonical(number: float) -> float:
  return number % 1.0


check2(truncate_number_alex)
check2(truncate_number_canonical)

#%%
model = "meta-llama/Llama-2-7b-hf"
# model = "codellama/CodeLlama-7b-Python-hf"
path = Path(f"../results/{model}/humanevalplus_generations_humanevalplus.json")
results: list[list[str]] = path.read_json()


for i in range(20):
    pprint(get_sol(2, i))
    print()
    print()

def truncate_number_llama_1(number: float) -> float:
    if number < 0:
        raise ValueError("Number must be positive")
    if number == 0:
        return 0
    if number == 1:
        return 1
    if number < 1:
        return 0
    if number < 10:
        return number
    if number < 100:
        return number // 10
    if number < 1000:
        return number // 100
    if number < 10000:
        return number // 1000
    if number < 100000:
        return number // 10000
    if number < 1000000:
        return number // 100000
    if number < 10000000:
        return number // 1000000
    if number < 100000000:
        return number // 10000000
    if number < 1000000000:
        return number // 100000000
    if number < 10000000000:
        return number // 1000000000
    if number < 100000000000:
        return number // 10000000000
    if number < 1000000000000:
        return number // 100000000000
    if number < 10000000000000:
        return number // 1000000000000

def truncate_number_llama_2(number: float) -> float:
    if number < 0:
        raise ValueError("Number must be positive")

    if number == 0:
        return 0

    if number < 1:
        return 0

    if number < 10:
        return number

    if number < 100:
        return number // 10

    if number < 1000:
        return number // 100

    if number < 10000:
        return number // 1000

    if number < 100000:
        return number // 10000

    if number < 1000000:
        return number // 100000

    if number < 10000000:
        return number // 1000000

    if number < 100000000:
        return number // 10000000

    if number < 1000000000:
        return number // 100000000

    if number < 10000000000:
        return number // 1000000000

    if number < 100000000000:
        return number // 10000000000

    if number < 1000000000000:
        return number // 100000000000

    if number < 10000000000000:
        return number // 100000000000

def truncate_number_llama_3(number: float) -> float:
    if number < 0:
        raise ValueError("Number must be positive")
    if number == 0:
        return 0
    if number == 1:
        return 1
    if number == 2:
        return 2
    if number == 3:
        return 3
    if number == 4:
        return 4
    if number == 5:
        return 5
    if number == 6:
        return 6
    if number == 7:
        return 7
    if number == 8:
        return 8
    if number == 9:
        return 9
    if number == 10:
        return 10
    if number == 11:
        return 11
    if number == 12:
        return 12
    if number == 13:
        return 13
    if number == 14:
        return 14
    if number == 15:
        return 15
    if number == 16:
        return 16
    if number == 17:
        return 17
    if number == 18:
        return 18
    if number == 19:
        return 19
    if number == 20:
        return 20
    if number == 21:
        return 21
    if number == 22:
        return 22
    if number == 23:
        return 23
    if number == 24:
        return 24
    if number == 25:
        return 25
    if number == 26:
        return 26
    if number == 27:
        return 27

def truncate_number_llama_4(number: float) -> float:
    if number < 0:
        raise ValueError("Cannot truncate negative numbers")

    if number == 0:
        return 0

    if number < 1:
        return 0

    if number < 10:
        return number

    if number < 100:
        return number // 10

    if number < 1000:
        return number // 100

    if number < 10000:
        return number // 1000

    if number < 100000:
        return number // 10000

    if number < 1000000:
        return number // 100000

    if number < 10000000:
        return number // 1000000

    if number < 100000000:
        return number // 10000000

    if number < 1000000000:
        return number // 100000000

    if number < 10000000000:
        return number // 1000000000

    if number < 100000000000:
        return number // 10000000000

    if number < 1000000000000:
        return number // 100000000000

    if number < 10000000000000:
        return number // 10000000000

def truncate_number_llama_5(number: float) -> float:
    return number - int(number)

def truncate_number_llama_6(number: float) -> float:
    if number < 0:
        raise ValueError("Cannot truncate negative numbers")

    if number == 0:
        return 0

    if number < 1:
        return 0

    if number < 10:
        return number

    if number < 100:
        return number // 10

    if number < 1000:
        return number // 100

    if number < 10000:
        return number // 1000

    if number < 100000:
        return number // 10000

    if number < 1000000:
        return number // 100000

    if number < 10000000:
        return number // 1000000

    if number < 100000000:
        return number // 10000000

    if number < 1000000000:
        return number // 100000000

    if number < 10000000000:
        return number // 1000000000

    if number < 100000000000:
        return number // 10000000000

    if number < 1000000000000:
        return number // 100000000000

    if number < 10000000000000:
        return number // 10000000000

def truncate_number_llama_7(number: float) -> float:
    if number < 0:
        raise ValueError("Number must be positive")
    if number == 0:
        return 0
    if number == 1:
        return 1
    if number == 2:
        return 2
    if number == 3:
        return 3
    if number == 4:
        return 4
    if number == 5:
        return 5
    if number == 6:
        return 6
    if number == 7:
        return 7
    if number == 8:
        return 8
    if number == 9:
        return 9
    if number == 10:
        return 10
    if number == 11:
        return 11
    if number == 12:
        return 12
    if number == 13:
        return 13
    if number == 14:
        return 14
    if number == 15:
        return 15
    if number == 16:
        return 16
    if number == 17:
        return 17
    if number == 18:
        return 18
    if number == 19:
        return 19
    if number == 20:
        return 20
    if number == 21:
        return 21
    if number == 22:
        return 22
    if number == 23:
        return 23
    if number == 24:
        return 24
    if number == 25:
        return 25
    if number == 26:
        return 26
    if number == 27:
        return 27

def truncate_number_llama_8(number: float) -> float:
    return number - int(number)

def truncate_number_llama_9(number: float) -> float:
    if number < 0:
        raise ValueError("Number must be positive")
    if number == 0:
        return 0
    if number < 1:
        return 0
    return number - int(number)

def truncate_number_llama_10(number: float) -> float:
    if number < 0:
        raise ValueError("Number must be positive")
    if number == 0:
        return 0
    if number == 1:
        return 1
    if number == 2:
        return 2
    if number == 3:
        return 3
    if number == 4:
        return 4
    if number == 5:
        return 5
    if number == 6:
        return 6
    if number == 7:
        return 7
    if number == 8:
        return 8
    if number == 9:
        return 9
    if number == 10:
        return 10
    if number == 11:
        return 11
    if number == 12:
        return 12
    if number == 13:
        return 13
    if number == 14:
        return 14
    if number == 15:
        return 15
    if number == 16:
        return 16
    if number == 17:
        return 17
    if number == 18:
        return 18
    if number == 19:
        return 19
    if number == 20:
        return 20
    if number == 21:
        return 21
    if number == 22:
        return 22
    if number == 23:
        return 23
    if number == 24:
        return 24
    if number == 25:
        return 25
    if number == 26:
        return 26
    if number == 27:
        return 27


def truncate_number_llama_12(number: float) -> float:
    if number < 0:
        raise ValueError("Number must be positive")
    if number == 0:
        return 0
    if number == 1:
        return 1
    if number == 2:
        return 2
    if number == 3:
        return 3
    if number == 4:
        return 4
    if number == 5:
        return 5
    if number == 6:
        return 6
    if number == 7:
        return 7
    if number == 8:
        return 8
    if number == 9:
        return 9
    if number == 10:
        return 10
    if number == 11:
        return 11
    if number == 12:
        return 12
    if number == 13:
        return 13
    if number == 14:
        return 14
    if number == 15:
        return 15
    if number == 16:
        return 16
    if number == 17:
        return 17
    if number == 18:
        return 18
    if number == 19:
        return 19
    if number == 20:
        return 20
    if number == 21:
        return 21
    if number == 22:
        return 22
    if number == 23:
        return 23
    if number == 24:
        return 24
    if number == 25:
        return 25

def truncate_number_llama_13(number: float) -> float:
    if number < 0:
        raise ValueError("Number must be positive")
    if number == 0:
        return 0
    if number == 1:
        return 1
    if number == 2:
        return 2
    if number == 3:
        return 3
    if number == 4:
        return 4
    if number == 5:
        return 5
    if number == 6:
        return 6
    if number == 7:
        return 7
    if number == 8:
        return 8
    if number == 9:
        return 9
    if number == 10:
        return 10
    if number == 11:
        return 11
    if number == 12:
        return 12
    if number == 13:
        return 13
    if number == 14:
        return 14
    if number == 15:
        return 15
    if number == 16:
        return 16
    if number == 17:
        return 17
    if number == 18:
        return 18
    if number == 19:
        return 19
    if number == 20:
        return 20
    if number == 21:
        return 21
    if number == 22:
        return 22
    if number == 23:
        return 23
    if number == 24:
        return 24
    if number == 25:
        return 25
    if number == 26:
        return 26
    if number == 27:
        return 27

def truncate_number_llama_14(number: float) -> float:
    return number - int(number)

def truncate_number_llama_15(number: float) -> float:
    if number < 0:
        raise ValueError("Cannot truncate negative numbers")
    if number == 0:
        return 0
    if number < 1:
        return 0
    if number < 10:
        return number
    if number < 100:
        return number // 10
    if number < 1000:
        return number // 100
    if number < 10000:
        return number // 1000
    if number < 100000:
        return number // 10000
    if number < 1000000:
        return number // 100000
    if number < 10000000:
        return number // 1000000
    if number < 100000000:
        return number // 10000000
    if number < 1000000000:
        return number // 100000000
    if number < 10000000000:
        return number // 1000000000
    if number < 100000000000:
        return number // 10000000000
    if number < 1000000000000:
        return number // 100000000000
    if number < 10000000000000:
        return number // 10000000000

def truncate_number_llama_16(number: float) -> float:
    if number < 0:
        raise ValueError("Number must be positive")
    if number == 0:
        return 0
    if number == 1:
        return 1
    if number == 2:
        return 2
    if number == 3:
        return 3
    if number == 4:
        return 4
    if number == 5:
        return 5
    if number == 6:
        return 6
    if number == 7:
        return 7
    if number == 8:
        return 8
    if number == 9:
        return 9
    if number == 10:
        return 10
    if number == 11:
        return 11
    if number == 12:
        return 12
    if number == 13:
        return 13
    if number == 14:
        return 14
    if number == 15:
        return 15
    if number == 16:
        return 16
    if number == 17:
        return 17
    if number == 18:
        return 18
    if number == 19:
        return 19
    if number == 20:
        return 20
    if number == 21:
        return 21
    if number == 22:
        return 22
    if number == 23:
        return 23
    if number == 24:
        return 24
    if number == 25:
        return 25
    if number == 26:
        return 26
    if number == 27:
        return 27

def truncate_number_llama_17(number: float) -> float:
    return number - int(number)

def truncate_number_llama_18(number: float) -> float:
    if number < 0:
        raise ValueError("Number must be positive")
    if number == 0:
        return 0
    if number < 1:
        return 0
    return number - int(number)

def truncate_number_llama_19(number: float) -> float:
    if number < 0:
        raise ValueError("Number must be positive")
    if number == 0:
        return 0
    if number == 1:
        return 1
    if number == 2:
        return 2
    if number == 3:
        return 3
    if number == 4:
        return 4
    if number == 5:
        return 5
    if number == 6:
        return 6
    if number == 7:
        return 7
    if number == 8:
        return 8
    if number == 9:
        return 9
    if number == 10:
        return 10
    if number == 11:
        return 11
    if number == 12:
        return 12
    if number == 13:
        return 13
    if number == 14:
        return 14
    if number == 15:
        return 15
    if number == 16:
        return 16
    if number == 17:
        return 17
    if number == 18:
        return 18
    if number == 19:
        return 19
    if number == 20:
        return 20
    if number == 21:
        return 21
    if number == 22:
        return 22
    if number == 23:
        return 23
    if number == 24:
        return 24
    if number == 25:
        return 25
    if number == 26:
        return 26
    if number == 27:
        return 27

test_all_solutions('truncate_number_llama_', check2)
#%%
# model = "meta-llama/Llama-2-7b-hf"
model = "codellama/CodeLlama-7b-Python-hf"
path = Path(f"../results/{model}/humanevalplus_generations_humanevalplus.json")
results: list[list[str]] = path.read_json()


for i in range(20):
    pprint(get_sol(2, i))
    print()
    print()

def truncate_number_codellama_1(number: float) -> float:
  return number - int(number)


def truncate_number_codellama_2(number: float) -> float:
  return number - int(number)


def truncate_number_codellama_3(number: float) -> float:
  return number - int(number)


def truncate_number_codellama_4(number: float) -> float:
  return number - int(number)


def truncate_number_codellama_5(number: float) -> float:
  return number - int(number)


def truncate_number_codellama_6(number: float) -> float:
  integer_part = int(number)
  return number - integer_part


def truncate_number_codellama_7(number: float) -> float:
  return number - int(number)


def truncate_number_codellama_8(number: float) -> float:
  return number - int(number)


def truncate_number_codellama_9(number: float) -> float:
  return number - int(number)


def truncate_number_codellama_10(number: float) -> float:
  return number - int(number)


def truncate_number_codellama_11(number: float) -> float:
  return number - int(number)


def truncate_number_codellama_12(number: float) -> float:
  return number - int(number)


def truncate_number_codellama_13(number: float) -> float:
  return number - int(number)


def truncate_number_codellama_14(number: float) -> float:
  return number - int(number)


def truncate_number_codellama_15(number: float) -> float:
  return number - int(number)


def truncate_number_codellama_16(number: float) -> float:
  return number - int(number)


def truncate_number_codellama_17(number: float) -> float:
  return number - int(number)


def truncate_number_codellama_18(number: float) -> float:
  return number - int(number)


def truncate_number_codellama_19(number: float) -> float:
  return number - int(number)


def truncate_number_codellama_20(number: float) -> float:
  return number - int(number)


test_all_solutions('truncate_number_codellama_', check2)


#%% problem 4

# pprint(task.get_dataset()[3]["prompt"])
# pprint(task.get_dataset()[3]["test"])
# pprint(task.get_dataset()[3]["canonical_solution"])

def below_zero_alex(operations: List[int]) -> bool:
    """ You're given a list of deposit and withdrawal operations on a bank 
    account that starts with
    zero balance. Your task is to detect if at any point the balance of 
    account falls below zero, and
    at that point function should return True. Otherwise it should return 
    False.
    >>> below_zero([1, 2, 3])
    False
    >>> below_zero([1, 2, -4, 5])
    True
    """
    balance = 0
    for op in operations:
      balance += op
      if balance < 0:
        return True
    return False




def check3(candidate):
    assert candidate([]) == False
    assert candidate([1, 2, -3, 1, 2, -3]) == False
    assert candidate([1, 2, -4, 5, 6]) == True
    assert candidate([1, -1, 2, -2, 5, -5, 4, -4]) == False
    assert candidate([1, -1, 2, -2, 5, -5, 4, -5]) == True
    assert candidate([1, -2, 2, -2, 5, -5, 4, -4]) == True


check3(below_zero_alex)

#%%
model = "meta-llama/Llama-2-7b-hf"
# model = "codellama/CodeLlama-7b-Python-hf"
path = Path(f"../results/{model}/humanevalplus_generations_humanevalplus.json")
results: list[list[str]] = path.read_json()


for i in range(20):
    pprint(get_sol(3, i))
    print()
    print()

#%%
def below_zero_llama_1(operations: List[int]) -> bool:
    if operations[0] < 0:
        return True
    else:
        return False


def below_zero_llama_2(operations: List[int]) -> bool:
    if len(operations) == 0:
        return False
    if operations[0] < 0:
        return True
    return below_zero_llama_2(operations[1:])


def below_zero_llama_3(operations: List[int]) -> bool:
    balance = 0
    for operation in operations:
        if operation > 0:
            balance += operation
        else:
            balance -= operation

    return balance < 0


def below_zero_llama_4(operations: List[int]) -> bool:
    if len(operations) == 0:
        return False
    elif operations[0] > 0:
        return False
    else:
        return True


def below_zero_llama_5(operations: List[int]) -> bool:
    if len(operations) == 0:
        return False
    if operations[0] < 0:
        return True
    if operations[-1] < 0:
        return True
    return False


def below_zero_llama_6(operations: List[int]) -> bool:
    if len(operations) == 0:
        return False
    balance = 0
    for operation in operations:
        if operation > 0:
            balance += operation
        else:
            balance -= operation
    if balance < 0:
        return True
    return False


def below_zero_llama_7(operations: List[int]) -> bool:
    balance = 0
    for operation in operations:
        if operation > 0:
            balance += operation
        else:
            balance -= operation

    return balance < 0


def below_zero_llama_8(operations: List[int]) -> bool:
    return any(operations)


def below_zero_llama_9(operations: List[int]) -> bool:
    balance = 0
    for operation in operations:
        if operation > 0:
            balance += operation
        else:
            balance -= operation
        if balance < 0:
            return True
    return False


def below_zero_llama_10(operations: List[int]) -> bool:
    if len(operations) == 0:
        return False
    if operations[0] < 0:
        return True
    return below_zero_llama_10(operations[1:])


def below_zero_llama_11(operations: List[int]) -> bool:
    return any(operations)


def below_zero_llama_12(operations: List[int]) -> bool:
    if len(operations) == 0:
        return False
    balance = 0
    for operation in operations:
        if operation > 0:
            balance += operation
        else:
            balance -= operation
    return balance < 0


def below_zero_llama_13(operations: List[int]) -> bool:
    if operations[0] < 0:
        return True
    else:
        return False


def below_zero_llama_14(operations: List[int]) -> bool:
    if len(operations) == 0:
        return False
    balance = 0
    for operation in operations:
        if operation > 0:
            balance += operation
        else:
            balance -= operation
    return balance < 0


def below_zero_llama_15(operations: List[int]) -> bool:
    return sum(operations) < 0


def below_zero_llama_16(operations: List[int]) -> bool:
    if len(operations) == 0:
        return False
    if operations[0] > 0:
        return False
    if operations[-1] < 0:
        return True
    return False


def below_zero_llama_17(operations: List[int]) -> bool:
    if len(operations) == 0:
        return False
    elif operations[0] == 0:
        return False
    elif operations[0] < 0:
        return True
    else:
        return below_zero_llama_17(operations[1:])


def below_zero_llama_18(operations: List[int]) -> bool:
    if len(operations) == 0:
        return False
    if operations[0] < 0:
        return True
    if operations[-1] < 0:
        return True
    return False


def below_zero_llama_19(operations: List[int]) -> bool:
    if operations[0] < 0:
        return True
    return False


def below_zero_llama_20(operations: List[int]) -> bool:
    if len(operations) == 0:
        return False

    balance = 0
    for operation in operations:
        if operation < 0:
            balance -= operation
        else:
            balance += operation

    return balance < 0



test_all_solutions('below_zero_llama_', check3)

#%% 
# model = "meta-llama/Llama-2-7b-hf"
model = "codellama/CodeLlama-7b-Python-hf"
path = Path(f"../results/{model}/humanevalplus_generations_humanevalplus.json")
results: list[list[str]] = path.read_json()


for i in range(20):
    pprint(get_sol(3, i))
    print()
    print()

#%%
def below_zero_codellama_1(operations: List[int]) -> bool:
    balance = 0
    for operation in operations:
        balance += operation
        if balance < 0:
            return True
    return False


def below_zero_codellama_2(operations: List[int]) -> bool:
    balance = 0
    for operation in operations:
        if operation < 0:
            balance += operation
        else:
            balance -= operation
        if balance < 0:
            return True
    return False


def below_zero_codellama_3(operations: List[int]) -> bool:
    balance = 0
    for operation in operations:
        balance += operation
        if balance < 0:
            return True
    return False


def below_zero_codellama_4(operations: List[int]) -> bool:
    balance = 0
    for operation in operations:
        balance += operation
        if balance < 0:
            return True
    return False


def below_zero_codellama_5(operations: List[int]) -> bool:
    balance = 0
    for operation in operations:
        balance += operation
        if balance < 0:
            return True
    return False


def below_zero_codellama_6(operations: List[int]) -> bool:
    balance = 0
    for operation in operations:
        balance += operation
        if balance < 0:
            return True
    return False


def below_zero_codellama_7(operations: List[int]) -> bool:
    balance = 0
    for operation in operations:
        balance += operation
        if balance < 0:
            return True
    return False


def below_zero_codellama_8(operations: List[int]) -> bool:
    balance = 0
    for operation in operations:
        balance += operation
        if balance < 0:
            return True
    return False


def below_zero_codellama_9(operations: List[int]) -> bool:
    balance = 0
    for operation in operations:
        balance += operation
        if balance < 0:
            return True
    return False


def below_zero_codellama_10(operations: List[int]) -> bool:
    balance = 0
    for operation in operations:
        if operation < 0:
            balance += operation
        else:
            balance -= operation
        if balance < 0:
            return True
    return False


def below_zero_codellama_11(operations: List[int]) -> bool:
    balance = 0
    for operation in operations:
        balance += operation
        if balance < 0:
            return True
    return False


def below_zero_codellama_12(operations: List[int]) -> bool:
    balance = 0
    for operation in operations:
        if operation < 0:
            balance += operation
        else:
            balance -= operation
        if balance < 0:
            return True
    return False


def below_zero_codellama_13(operations: List[int]) -> bool:
    balance = 0
    for operation in operations:
        balance += operation
        if balance < 0:
            return True
    return False


def below_zero_codellama_14(operations: List[int]) -> bool:
    balance = 0
    for operation in operations:
        balance += operation
        if balance < 0:
            return True
    return False


def below_zero_codellama_15(operations: List[int]) -> bool:
    balance = 0
    for operation in operations:
        balance += operation
        if balance < 0:
            return True
    return False


def below_zero_codellama_16(operations: List[int]) -> bool:
    balance = 0
    for operation in operations:
        balance += operation
        if balance < 0:
            return True
    return False


def below_zero_codellama_17(operations: List[int]) -> bool:
    balance = 0
    for operation in operations:
        balance += operation
        if balance < 0:
            return True
    return False


def below_zero_codellama_18(operations: List[int]) -> bool:
    balance = 0
    for operation in operations:
        balance += operation
        if balance < 0:
            return True
    return False


def below_zero_codellama_19(operations: List[int]) -> bool:
    balance = 0
    for operation in operations:
        balance += operation
        if balance < 0:
            return True
    return False


def below_zero_codellama_20(operations: List[int]) -> bool:
    balance = 0
    for operation in operations:
        balance += operation
        if balance < 0:
            return True
    return False

test_all_solutions("below_zero_codellama_", check3)
#%% problem 5
# pprint(task.get_dataset()[4]["prompt"])
def mean_absolute_deviation_alex(numbers: List[float]) -> float:
  """ For a given list of input numbers, calculate Mean Absolute Deviation
  around the mean of this dataset.
  Mean Absolute Deviation is the average absolute difference between each
  element and a centerpoint (mean in this case):
  MAD = average | x - x_mean |
  >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])
  1.0
  """
  avg = sum(numbers) / len(numbers)
  deviations = [abs(x - avg) for x in numbers]
  return sum(deviations) / len(deviations)

def mean_absolute_deviation_canonical(numbers: List[float]) -> float:
   mean = sum(numbers) / len(numbers)
   return sum(abs(x - mean) for x in numbers) / len(numbers)


# pprint(task.get_dataset()[4]["test"])

def check4(candidate):
    assert abs(candidate([1.0, 2.0, 3.0]) - 2.0/3.0) < 1e-6
    assert abs(candidate([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6
    assert abs(candidate([1.0, 2.0, 3.0, 4.0, 5.0]) - 6.0/5.0) < 1e-6

check4(mean_absolute_deviation_alex)
# pprint(task.get_dataset()[4]["canonical_solution"])

model = "meta-llama/Llama-2-7b-hf"
# model = "codellama/CodeLlama-7b-Python-hf"
path = Path(f"../results/{model}/humanevalplus_generations_humanevalplus.json")
results: list[list[str]] = path.read_json()

#%% llama
for i in range(20):
    pprint(get_sol(4, i))
    print()
    print()

#%% 
def mean_absolute_deviation_llama_1(numbers: List[float]) -> float:
  mean = sum(numbers) / len(numbers)
  return sum(abs(x - mean) for x in numbers) / len(numbers)


def mean_absolute_deviation_llama_2(numbers: List[float]) -> float:
  mean = sum(numbers) / len(numbers)
  return sum(abs(x - mean) for x in numbers) / len(numbers)


def mean_absolute_deviation_llama_3(numbers: List[float]) -> float:
  return sum(abs(x - mean(numbers))) / len(numbers)


def mean_absolute_deviation_llama_4(numbers: List[float]) -> float:
  mean = sum(numbers) / len(numbers)
  return sum(abs(number - mean) for number in numbers) / len(numbers)


def mean_absolute_deviation_llama_5(numbers: List[float]) -> float:
  mean = sum(numbers) / len(numbers)
  return sum(abs(x - mean) for x in numbers) / len(numbers)


def mean_absolute_deviation_llama_6(numbers: List[float]) -> float:
  mean = sum(numbers) / len(numbers)
  return sum(abs(x - mean) for x in numbers) / len(numbers)


def mean_absolute_deviation_llama_7(numbers: List[float]) -> float:
  return sum(abs(x - mean(numbers))) / len(numbers)


def mean_absolute_deviation_llama_8(numbers: List[float]) -> float:
  mean = sum(numbers) / len(numbers)
  return sum(abs(x - mean) for x in numbers) / len(numbers)


def mean_absolute_deviation_llama_9(numbers: List[float]) -> float:
  mean = sum(numbers) / len(numbers)
  return sum(abs(x - mean) for x in numbers) / len(numbers)


def mean_absolute_deviation_llama_10(numbers: List[float]) -> float:
  return sum(abs(x - mean(numbers))) / len(numbers)


def mean_absolute_deviation_llama_11(numbers: List[float]) -> float:
  return sum(abs(number - mean(numbers))) / len(numbers)


def mean_absolute_deviation_llama_12(numbers: List[float]) -> float:
  mean = sum(numbers) / len(numbers)
  return sum(abs(x - mean) for x in numbers) / len(numbers)


def mean_absolute_deviation_llama_13(numbers: List[float]) -> float:
  mean = sum(numbers) / len(numbers)
  return sum(abs(x - mean) for x in numbers) / len(numbers)


def mean_absolute_deviation_llama_14(numbers: List[float]) -> float:
  return sum(abs(x - mean(numbers))) / len(numbers)


def mean_absolute_deviation_llama_15(numbers: List[float]) -> float:
  return sum(abs(x - mean(numbers))) / len(numbers)


def mean_absolute_deviation_llama_16(numbers: List[float]) -> float:
  return sum(abs(x - mean(numbers))) / len(numbers)


def mean_absolute_deviation_llama_17(numbers: List[float]) -> float:
  return sum(abs(number - mean(numbers))) / len(numbers)


def mean_absolute_deviation_llama_18(numbers: List[float]) -> float:
  mean = sum(numbers) / len(numbers)
  return sum(abs(number - mean) for number in numbers) / len(numbers)


def mean_absolute_deviation_llama_19(numbers: List[float]) -> float:
  return sum(abs(x - mean(numbers))) / len(numbers)


def mean_absolute_deviation_llama_20(numbers: List[float]) -> float:
  return sum(abs(x - mean(numbers))) / len(numbers)


test_all_solutions("mean_absolute_deviation_llama_", check4)

#%% codellama
# model = "meta-llama/Llama-2-7b-hf"
model = "codellama/CodeLlama-7b-Python-hf"
path = Path(f"../results/{model}/humanevalplus_generations_humanevalplus.json")
results: list[list[str]] = path.read_json()

for i in range(20):
    pprint(get_sol(4, i))
    print()
    print()

def mean_absolute_deviation_codellama_1(numbers: List[float]) -> float:
  mean = sum(numbers) / len(numbers)
  return sum(abs(x - mean) for x in numbers) / len(numbers)


def mean_absolute_deviation_codellama_2(numbers: List[float]) -> float:
  mean = sum(numbers) / len(numbers)
  return sum(abs(x - mean) for x in numbers) / len(numbers)


def mean_absolute_deviation_codellama_3(numbers: List[float]) -> float:
  mean = sum(numbers) / len(numbers)
  return sum(abs(x - mean) for x in numbers) / len(numbers)


def mean_absolute_deviation_codellama_4(numbers: List[float]) -> float:
  mean = sum(numbers) / len(numbers)
  return sum(abs(x - mean) for x in numbers) / len(numbers)


def mean_absolute_deviation_codellama_5(numbers: List[float]) -> float:
  mean = sum(numbers) / len(numbers)
  return sum(abs(x - mean) for x in numbers) / len(numbers)


def mean_absolute_deviation_codellama_6(numbers: List[float]) -> float:
  mean = sum(numbers) / len(numbers)
  return sum(abs(x - mean) for x in numbers) / len(numbers)


def mean_absolute_deviation_codellama_7(numbers: List[float]) -> float:
  mean = sum(numbers) / len(numbers)
  return sum(abs(x - mean) for x in numbers) / len(numbers)


def mean_absolute_deviation_codellama_8(numbers: List[float]) -> float:
  mean = sum(numbers) / len(numbers)
  return sum(abs(x - mean) for x in numbers) / len(numbers)


def mean_absolute_deviation_codellama_9(numbers: List[float]) -> float:
  return sum(abs(x - sum(numbers) / len(numbers)) for x in numbers) / len(numbers)


def mean_absolute_deviation_codellama_10(numbers: List[float]) -> float:
  return sum(abs(x - sum(numbers) / len(numbers)) for x in numbers) / len(numbers)


def mean_absolute_deviation_codellama_11(numbers: List[float]) -> float:
  return sum(abs(x - sum(numbers) / len(numbers)) for x in numbers) / len(numbers)


def mean_absolute_deviation_codellama_12(numbers: List[float]) -> float:
  return sum(abs(x - sum(numbers) / len(numbers)) for x in numbers) / len(numbers)


def mean_absolute_deviation_codellama_13(numbers: List[float]) -> float:
  return sum(abs(number - sum(numbers) / len(numbers)) for number in numbers) / len(numbers)


def mean_absolute_deviation_codellama_14(numbers: List[float]) -> float:
  return sum(abs(x - sum(numbers) / len(numbers)) for x in numbers) / len(numbers)


def mean_absolute_deviation_codellama_15(numbers: List[float]) -> float:
  mean = sum(numbers) / len(numbers)
  return sum(abs(x - mean) for x in numbers) / len(numbers)


def mean_absolute_deviation_codellama_16(numbers: List[float]) -> float:
  mean = sum(numbers) / len(numbers)
  return sum(abs(x - mean) for x in numbers) / len(numbers)


def mean_absolute_deviation_codellama_17(numbers: List[float]) -> float:
  mean = sum(numbers) / len(numbers)
  return sum(abs(x - mean) for x in numbers) / len(numbers)


def mean_absolute_deviation_codellama_18(numbers: List[float]) -> float:
  mean = sum(numbers) / len(numbers)
  return sum(abs(x - mean) for x in numbers) / len(numbers)


def mean_absolute_deviation_codellama_19(numbers: List[float]) -> float:
  return sum(abs(x - sum(numbers) / len(numbers)) for x in numbers) / len(numbers)


def mean_absolute_deviation_codellama_20(numbers: List[float]) -> float:
  mean = sum(numbers) / len(numbers)
  return sum(abs(x - mean) for x in numbers) / len(numbers)


test_all_solutions("mean_absolute_deviation_codellama_", check4)