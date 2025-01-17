# grab human eval questions and 
# human eval questions
#%%
from datasets import load_dataset
from pprint import pprint
from typing import List
import numpy as np

ds = load_dataset("evalplus/humanevalplus")['test']

def get_problem(problem_number: int, kind: str = 'prompt'):
  """'task_id', 'prompt', 'canonical_solution', 'entry_point', 'test'"""
  return ds[kind][problem_number]

def test_solution(my_function, problem_number):
    ds = load_dataset("evalplus/humanevalplus")["test"]
    test_code = ds["test"][problem_number]

    # Create testing environment with required functions/imports
    test_env = {
        "candidate": my_function,
        "np": np,
        "assertion": lambda out, exp, atol: assert_equal(out, exp, atol),
        "__name__": "__main__",
    }

    try:
        # Execute test code in our testing environment
        exec(test_code, test_env)
        # Explicitly call the check function
        test_env["check"](my_function)
        print("✓ All tests passed!")
        return True
    except AssertionError as e:
        print(f"✗ Test failed: {str(e)}")
        return False
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


def assert_equal(out, exp, atol):
    if out != exp:
        raise AssertionError(f"Expected {exp}, but got {out}")

pprint(get_problem(66, kind='prompt'))




# def get_problem(problem_number: int):
#     return list((RESULTS_PATH / "COMMIT_1").glob(f"HumanEval_{problem_number}_*.json"))[
#         0
#     ].read_json()

#%%
from evalplus.evaluate import evaluate_functional_correctness
# problem 66
def digitSum(text: str) -> int:
    """Write a function that takes a string as input and returns the sum of the
    upper characters only ASCII codes.

    Examples:
    >>> digitSum('')
    0
    >>> digitSum('abAB')
    131
    >>> digitSum('abcCd')
    67
    >>> digitSum('helloE')
    69
    >>> digitSum('woArBld')
    131
    >>> digitSum('aAaaaXa')
    153
    """
    return sum(ord(char) for char in text if char.isupper())

test_solution(digitSum, 66)
#%%
from typing import List
def intersperse(numbers: List[int], delimeter: int) -> List[int]:
  """ Insert a number 'delimeter' between every two consecutive elements 
  of input list `numbers'
  >>> intersperse([], 4)
  []
  >>> intersperse([1, 2, 3], 4)
  [1, 4, 2, 4, 3]
  """
  pass


#%%
def digitSum(text: str) -> int:
    """Write a function that takes a string as input and returns the sum of the
    upper characters only ASCII codes.

    Examples:
    >>> digitSum('')
    0
    >>> digitSum('abAB')
    131
    >>> digitSum('abcCd')
    67
    >>> digitSum('helloE')
    69
    >>> digitSum('woArBld')
    131
    >>> digitSum('aAaaaXa')
    153
    """
    return sum(ord(char) for char in text if char.isupper())

pprint(ds['prompt'][66])
pprint(ds['test'][66])

#%%
