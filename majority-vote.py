"""
Majority Vote Protocol Implementation

This script implements a system where multiple AI agents (Red Teams) analyze and solve
a given programming problem. The responses are saved to timestamped files for analysis.

The system uses OpenAI's GPT models to generate solutions and implements a basic
version of a majority vote protocol where multiple AI agents work on the same problem.
"""

# Standard library imports
import json
import subprocess
import sys
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime
from pathlib import Path
import tempfile

# Third-party imports
import openai
from dotenv import load_dotenv
from datasets import load_dataset

# Load environment variables from .env file (contains OpenAI API key)
load_dotenv()

# Initialize OpenAI client using API key from environment variables
client = openai.OpenAI()  # The API key will be automatically loaded from OPENAI_API_KEY environment variable

# System prompt for Red Teams (analytical problem solvers)
SYSTEM_PROMPT_RED = """You are a systematic and analytical Python problem solver.
Your task is to analyze and solve the following programming problem:

{problem}

Your approach:
1. Break down problems into clear components
2. Use established patterns and best practices 
3. Focus on reliability and proven solutions
4. Consider edge cases and potential issues

IMPORTANT: Please provide ONLY the Python code solution. Do not include any explanation, commentary, or markdown. 
Your response should be valid Python code that can be directly executed.
"""

# System prompt for the problem restator
SYSTEM_PROMPT_RESTATOR = """You are an expert at rephrasing and clarifying problems.
Your task:
1. Reframe the given problem in clear, precise terms
2. Identify core requirements and constraints
3. Structure the problem in a systematic way
4. Highlight key elements that need to be addressed
Provide a clear, structured restatement that both Red teams can work from."""

def load_apps_problem(problem_id: int = None) -> Tuple[str, Dict, int]:
    """
    Load a problem from the HuggingFace APPS dataset.
    
    Args:
        problem_id (int, optional): Specific problem ID to load. If None, loads the first problem.
    
    Returns:
        Tuple[str, Dict, int]: Contains the problem description, input/output tests, and problem ID
    """
    print("Loading APPS dataset...")
    
    # Load the dataset
    dataset = load_dataset("codeparrot/apps", split="test")
    
    # If no problem_id specified, use the first one
    if problem_id is None:
        sample = dataset[0]
        problem_id = sample["problem_id"]
    else:
        # Find the problem with the specified ID
        matching_samples = [s for s in dataset if s["problem_id"] == problem_id]
        if not matching_samples:
            raise ValueError(f"Problem with ID {problem_id} not found in the dataset.")
        sample = matching_samples[0]
    
    # Extract problem description
    problem_description = sample["question"]
    
    # Parse input-output tests
    input_output = json.loads(sample["input_output"]) if sample["input_output"] else {"inputs": [], "outputs": []}
    
    print(f"Loaded problem ID: {problem_id}")
    return problem_description, input_output, problem_id

def red_1(problem: str) -> str:
    """
    First Red Team AI agent implementation.
    
    This function sends the problem to GPT-3.5-turbo with specific system instructions
    to analyze and solve the problem systematically.
    
    Args:
        problem (str): The problem statement to analyze
        
    Returns:
        str: The AI's proposed solution or error message if the API call fails
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_RED.format(problem=problem)},
                {"role": "user", "content": "Solve this programming problem. Provide ONLY the Python code solution."}
            ],
            temperature=0,  # Use temperature 0 for more consistent responses
            max_tokens=1500  # Increase token limit to allow for complete solutions
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting AI response: {e}")
        return "Error: Failed to get AI response"

def red_2(problem: str) -> str:
    """
    Second Red Team AI agent implementation.
    
    Functions identically to red_1 but operates as an independent agent.
    This allows for comparison of solutions between different instances.
    
    Args:
        problem (str): The problem statement to analyze
        
    Returns:
        str: The AI's proposed solution or error message if the API call fails
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_RED.format(problem=problem)},
                {"role": "user", "content": "Solve this programming problem. Provide ONLY the Python code solution."}
            ],
            temperature=0,
            max_tokens=1500  # Increase token limit to allow for complete solutions
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting AI response: {e}")
        return "Error: Failed to get AI response"

def problem_restator(prompt: str) -> str:
    """
    AI agent that reformulates and clarifies the given problem.
    
    Uses a higher temperature setting to allow for more creative restatements
    of the original problem.
    
    Args:
        prompt (str): The original problem statement
        
    Returns:
        str: A reformulated version of the problem or error message
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_RESTATOR},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,  # Higher temperature for more varied restatements
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting AI response: {e}")
        return "Error: Failed to get AI response"

def run_code_with_input(code: str, input_text: str, timeout: int = 10) -> Tuple[str, str, bool]:
    """
    Run Python code with the given input and return the output.
    
    Args:
        code (str): Python code to execute
        input_text (str): Input to provide to the program
        timeout (int): Maximum execution time in seconds
        
    Returns:
        Tuple[str, str, bool]: stdout, stderr, and whether execution was successful
    """
    # Create a temporary file to write the code
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as tmp_file:
        tmp_file.write(code.encode('utf-8'))
        tmp_file_path = tmp_file.name
    
    try:
        # Run the code with the given input
        proc = subprocess.Popen(
            [sys.executable, tmp_file_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        try:
            stdout, stderr = proc.communicate(input=input_text, timeout=timeout)
            success = proc.returncode == 0
            return stdout.strip(), stderr.strip(), success
        except subprocess.TimeoutExpired:
            proc.kill()
            return "", "Execution timed out", False
    except Exception as e:
        return "", f"Error executing code: {str(e)}", False
    finally:
        # Clean up the temporary file
        os.unlink(tmp_file_path)

def test_solution(code: str, input_output: Dict) -> Tuple[bool, str]:
    """
    Test a solution against a set of input-output test cases.
    
    Args:
        code (str): Python code solution to test
        input_output (Dict): Dictionary containing 'inputs' and 'outputs' lists
        
    Returns:
        Tuple[bool, str]: Whether all tests passed and an explanation
    """
    if not input_output or not input_output.get("inputs") or not input_output.get("outputs"):
        return False, "No test cases available"
    
    inputs = input_output["inputs"]
    expected_outputs = input_output["outputs"]
    
    test_results = []
    all_passed = True
    
    for i, (input_text, expected_output) in enumerate(zip(inputs, expected_outputs)):
        print(f"Running test case {i+1}...")
        actual_output, stderr, success = run_code_with_input(code, input_text)
        
        # Clean up whitespace for comparison
        expected_output = expected_output.strip()
        actual_output = actual_output.strip()
        
        test_passed = actual_output == expected_output and success
        
        if not test_passed:
            all_passed = False
        
        test_results.append({
            "test_case": i+1,
            "passed": test_passed,
            "input": input_text,
            "expected_output": expected_output,
            "actual_output": actual_output,
            "stderr": stderr
        })
    
    # Generate summary
    passed_count = sum(1 for r in test_results if r["passed"])
    explanation = f"Passed {passed_count}/{len(test_results)} test cases."
    
    if not all_passed:
        # Add details of first failed test
        failed_test = next((r for r in test_results if not r["passed"]), None)
        if failed_test:
            explanation += f"\nFailed test case {failed_test['test_case']}:"
            explanation += f"\nInput: {failed_test['input']}"
            explanation += f"\nExpected: {failed_test['expected_output']}"
            explanation += f"\nActual: {failed_test['actual_output']}"
            if failed_test["stderr"]:
                explanation += f"\nError: {failed_test['stderr']}"
    
    return all_passed, explanation

def create_results_folder() -> Path:
    """
    Creates a timestamped folder for storing the results of the current run.
    
    The folder structure will be:
    results/
        YYYY-MM-DD_HH-MM-SS/
            red_team_1.txt
            red_team_2.txt
            test_results.json
    
    Returns:
        Path: Path object pointing to the created folder
    """
    base_path = Path("/Users/sebastianbowkis/Ai-research/majority-vote-protocol/results")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_folder = base_path / current_time
    result_folder.mkdir(parents=True, exist_ok=True)
    return result_folder

def save_response_to_file(folder_path: Path, team_name: str, response: str):
    """
    Saves an AI team's response to a text file in the results folder.
    
    Args:
        folder_path (Path): Path to the results folder
        team_name (str): Name of the AI team (e.g., "Red Team 1")
        response (str): The AI's response to save
    """
    filename = f"{team_name.replace(' ', '_').lower()}.txt"
    file_path = folder_path / filename
    with open(file_path, "w") as f:
        f.write(response)

def save_test_results(folder_path: Path, results: Dict):
    """
    Saves test results to a JSON file in the results folder.
    
    Args:
        folder_path (Path): Path to the results folder
        results (Dict): Dictionary containing test results
    """
    file_path = folder_path / "test_results.json"
    with open(file_path, "w") as f:
        json.dump(results, f, indent=2)

def solve_problem_from_apps(problem_id: int = None) -> Dict[str, any]:
    """
    Main problem-solving orchestrator using APPS dataset.
    
    This function:
    1. Loads a problem from the APPS dataset
    2. Creates a new results folder
    3. Gets solutions from both Red Teams
    4. Tests the solutions
    5. Saves results to files
    6. Returns all data
    
    Args:
        problem_id (int, optional): Specific problem ID to load. If None, loads a default problem.
        
    Returns:
        Dict[str, any]: Dictionary containing problem, solutions, test results, and folder path
    """
    # Load problem from APPS dataset
    problem_description, input_output, problem_id = load_apps_problem(problem_id)
    
    # Create results folder
    results_folder = create_results_folder()
    
    print("Getting solution from Red Team 1...")
    red1_solution = red_1(problem_description)
    
    print("Getting solution from Red Team 2...")
    red2_solution = red_2(problem_description)
    
    # Save solutions to files
    save_response_to_file(results_folder, "Red Team 1", red1_solution)
    save_response_to_file(results_folder, "Red Team 2", red2_solution)
    
    # Test solutions
    print("Testing Red Team 1 solution...")
    red1_passed, red1_explanation = test_solution(red1_solution, input_output)
    
    print("Testing Red Team 2 solution...")
    red2_passed, red2_explanation = test_solution(red2_solution, input_output)
    
    # Prepare results
    results = {
        "problem_id": problem_id,
        "problem_description": problem_description,
        "red_team_1": {
            "solution": red1_solution,
            "passed_all_tests": red1_passed,
            "explanation": red1_explanation,
        },
        "red_team_2": {
            "solution": red2_solution,
            "passed_all_tests": red2_passed,
            "explanation": red2_explanation,
        },
        "majority_vote": {
            "agreement": red1_passed == red2_passed,
            "overall_result": "PASS" if red1_passed and red2_passed else "FAIL"
        },
        "results_folder": str(results_folder)
    }
    
    # Save test results
    save_test_results(results_folder, results)
    
    return results

def pretty_print_results(results: Dict[str, any]):
    """
    Formats and prints the problem-solving results to the console.
    
    Args:
        results (Dict[str, any]): Dictionary containing problem, solutions, and test results
    """
    print("\n" + "="*80 + "\n")
    print(f"PROBLEM ID: {results['problem_id']}")
    print("\n" + "-"*40 + "\n")
    print("PROBLEM DESCRIPTION:")
    print(results['problem_description'][:300] + "..." if len(results['problem_description']) > 300 else results['problem_description'])
    
    print("\n" + "-"*40 + "\n")
    print("RED TEAM 1 RESULTS:")
    print(f"Passed all tests: {results['red_team_1']['passed_all_tests']}")
    print(results['red_team_1']['explanation'])
    
    print("\n" + "-"*40 + "\n")
    print("RED TEAM 2 RESULTS:")
    print(f"Passed all tests: {results['red_team_2']['passed_all_tests']}")
    print(results['red_team_2']['explanation'])
    
    print("\n" + "-"*40 + "\n")
    print("MAJORITY VOTE RESULTS:")
    print(f"Teams in agreement: {results['majority_vote']['agreement']}")
    print(f"Overall result: {results['majority_vote']['overall_result']}")
    
    print("\n" + "-"*40 + "\n")
    print(f"Results saved to: {results['results_folder']}")
    print("\n" + "="*80 + "\n")

def main():
    """
    Main entry point of the script.
    
    Executes the problem-solving workflow:
    1. Loads a problem from APPS dataset
    2. Gets solutions from the AI teams
    3. Tests solutions against input-output examples
    4. Prints and saves the results
    """
    print("Majority Vote Protocol - APPS Problem Solver")
    print("=" * 50)
    
    # Optionally specify a problem ID (default is None, which uses the first problem)
    # problem_id = 42  # Uncomment and modify to use a specific problem
    
    try:
        results = solve_problem_from_apps()
        pretty_print_results(results)
    except Exception as e:
        print(f"Error running majority vote protocol: {e}")
        import traceback
        traceback.print_exc()

# Script entry point
if __name__ == "__main__":
    main()
