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
from openai import OpenAI  # Import the OpenAI class directly

# If you want to check the version, use this instead:
import pkg_resources
openai_version = pkg_resources.get_distribution("openai").version
print(f"OpenAI version: {openai_version}")

from dotenv import load_dotenv
from datasets import load_dataset

# Load environment variables from .env file
load_dotenv()

# Global configuration
PROBLEM_ID = 4  # Default problem ID to use

# Simplest possible initialization - should avoid the proxies issue
client = OpenAI()

# # Initialize OpenAI client using API key from environment variables
# client = openai.OpenAI()  # The API key will be automatically loaded from OPENAI_API_KEY environment variable

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
    print(f"Connecting to HuggingFace to load problem ID: {problem_id if problem_id is not None else 'None (will use first problem)'}")
    
    try:
        # Only download the specific problem we need using the streaming API
        print("Using streaming API to avoid downloading the entire dataset...")
        for idx, sample in enumerate(load_dataset("codeparrot/apps", split="test", streaming=True)):
            if problem_id is None and idx == 0:
                # First problem when no ID specified
                problem_id = sample["problem_id"]
                print(f"Successfully loaded first problem (ID: {problem_id})")
                break
            elif sample["problem_id"] == problem_id:
                # Found the problem with the specific ID
                print(f"Successfully found problem with ID: {problem_id}")
                break
            
            # Print progress every 100 problems
            if idx % 100 == 0:
                print(f"Searching... Checked {idx} problems so far")
                
        # If we reach this point without breaking, we didn't find the problem
        else:
            raise ValueError(f"Problem with ID {problem_id} not found in the dataset after checking {idx+1} problems")
    
        # Extract problem description
        problem_description = sample["question"]
        print(f"Problem title: {problem_description.split(chr(10))[0] if chr(10) in problem_description else problem_description[:50]}...")
        
        # Parse input-output tests
        input_output = json.loads(sample["input_output"]) if sample["input_output"] else {"inputs": [], "outputs": []}
        print(f"Number of test cases: {len(input_output.get('inputs', []))}")
        
        return problem_description, input_output, problem_id
        
    except Exception as e:
        print(f"Error loading problem from dataset: {e}")
        raise

def red_1(problem: str) -> str:
    """
    First Red Team AI agent implementation.
    
    This function sends the problem to GPT-4o with specific system instructions
    to analyze and solve the problem systematically.
    
    Args:
        problem (str): The problem statement to analyze
        
    Returns:
        str: The AI's proposed solution or error message if the API call fails
    """
    print("Requesting solution from Red Team 1 (GPT-3.5)...")
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_RED.format(problem=problem)},
                    {"role": "user", "content": "Solve this programming problem. Provide ONLY the Python code solution."}
                ],
                #temperature=0,  # Use temperature 0 for more consistent responses
                max_completion_tokens=1000  # Increase token limit to allow for complete solutions
            )
            print(f"Red Team 1 response received! ({len(response.choices[0].message.content)} chars)")
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error getting AI response (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                print(f"Retrying in 5 seconds...")
                import time
                time.sleep(5)
            else:
                return "Error: Failed to get AI response after multiple attempts"

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
    print("Requesting solution from Red Team 2 (GPT-3.5)...")
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_RED.format(problem=problem)},
                    {"role": "user", "content": "Solve this programming problem. Provide ONLY the Python code solution."}
                ],
                #temperature=0,
                max_completion_tokens=1000  # Increase token limit to allow for complete solutions
            )
            print(f"Red Team 2 response received! ({len(response.choices[0].message.content)} chars)")
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error getting AI response (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                print(f"Retrying in 5 seconds...")
                import time
                time.sleep(5)
            else:
                return "Error: Failed to get AI response after multiple attempts"

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
    print("Requesting problem restatement from AI...")
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_RESTATOR},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,  # Higher temperature for more varied restatements
                max_tokens=500
            )
            print("Problem restatement received!")
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error getting AI response (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                print(f"Retrying in 5 seconds...")
                import time
                time.sleep(5)
            else:
                return "Error: Failed to get AI response after multiple attempts"

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
    print(f"Running code with input (timeout: {timeout}s)...")
    
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
            print(f"Code execution {'successful' if success else 'failed'}")
            return stdout.strip(), stderr.strip(), success
        except subprocess.TimeoutExpired:
            proc.kill()
            print(f"Execution timed out after {timeout} seconds")
            return "", "Execution timed out", False
    except Exception as e:
        print(f"Error executing code: {str(e)}")
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

def majority_vote(solutions, input_text):
    """
    Implements majority voting by comparing the outputs of multiple solutions on the same input.
    Only returns an answer if all solutions produce identical outputs.
    
    Args:
        solutions (list): List of Python code solutions (already cleaned)
        input_text (str): Input to test the solutions on
        
    Returns:
        tuple: (matched, output)
            - matched: Boolean indicating if all solutions produced identical outputs
            - output: The agreed-upon output if matched is True, None otherwise
    """
    # Store outputs from each solution
    solution_outputs = []
    
    # Run each solution with the same input
    for i, solution in enumerate(solutions):
        output, stderr, success = run_code_with_input(solution, input_text)
        
        # Record the output (or error message if execution failed)
        if success:
            solution_outputs.append((i, output.strip()))
        else:
            solution_outputs.append((i, f"ERROR: {stderr}"))
    
    # Extract just the outputs for comparison (ignore the solution indices)
    outputs_only = [output for _, output in solution_outputs]
    
    # Check if all solutions produced identical outputs
    unique_outputs = set(outputs_only)
    outputs_match = len(unique_outputs) == 1
    
    # Return detailed results
    result = {
        "matched": outputs_match,
        "output": outputs_only[0] if outputs_match else None,
        "all_outputs": solution_outputs
    }
    
    return result


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
    base_path = Path("//Users/abowkis/Documents/Projects/majority-vote-protocol/results")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_folder = base_path / current_time
    result_folder.mkdir(parents=True, exist_ok=True)
    return result_folder

def clean_code_response(response: str) -> str:
    """
    Cleans AI response text by removing markdown code block delimiters.
    
    Args:
        response (str): The AI response text that may contain markdown code blocks
        
    Returns:
        str: Cleaned code without markdown delimiters
    """
    # Remove markdown code block syntax
    response = response.strip()
    
    # Remove opening code block markers like ```python, ```java, etc.
    if response.startswith("```"):
        first_line_end = response.find("\n")
        if first_line_end != -1:
            response = response[first_line_end+1:]
    
    # Remove closing code block markers
    if response.endswith("```"):
        response = response[:-3].strip()
    
    return response

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
    
    # Clean the response before saving
    cleaned_response = clean_code_response(response)
    
    with open(file_path, "w") as f:
        f.write(cleaned_response)

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
    Phase 1: Generate and test solutions
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
    
    # Clean the solutions before testing
    red1_cleaned = clean_code_response(red1_solution)
    red2_cleaned = clean_code_response(red2_solution)
    
    # Test solutions against the APPS test cases
    print("Testing Red Team 1 solution...")
    red1_passed, red1_explanation = test_solution(red1_cleaned, input_output)
    
    print("Testing Red Team 2 solution...")
    red2_passed, red2_explanation = test_solution(red2_cleaned, input_output)
    
    # Prepare results (without majority voting for now)
    results = {
        "problem_id": problem_id,
        "problem_description": problem_description,
        "red_team_1": {
            "solution": red1_solution,
            "cleaned_solution": red1_cleaned,
            "passed_all_tests": red1_passed,
            "explanation": red1_explanation,
        },
        "red_team_2": {
            "solution": red2_solution,
            "cleaned_solution": red2_cleaned,
            "passed_all_tests": red2_passed,
            "explanation": red2_explanation,
        },
        "results_folder": str(results_folder)
    }
    
    # Save test results
    save_test_results(results_folder, results)
    
    return results

def apply_majority_voting(results, test_input=None):
    """
    Phase 2: Apply majority voting to solutions that passed testing
    
    Args:
        results (dict): Results from solve_problem_from_apps
        test_input (str, optional): Input to use for testing. If None, use the first input from APPS
    
    Returns:
        dict: Updated results with majority voting information
    """
    red1_passed = results["red_team_1"]["passed_all_tests"]
    red2_passed = results["red_team_2"]["passed_all_tests"]
    
    # Initialize majority voting results
    majority_vote_results = {
        "performed": False,
        "matched": False,
        "output": None,
        "all_outputs": []
    }
    
    # Only proceed to majority voting if both pass all tests
    if red1_passed and red2_passed:
        print("Both solutions passed tests. Proceeding to majority voting...")
        
        # Get the solutions
        red1_cleaned = results["red_team_1"]["cleaned_solution"]
        red2_cleaned = results["red_team_2"]["cleaned_solution"]
        
        # If no test input provided, try to get one from the original problem
        if test_input is None:
            # Load the problem again to get inputs
            _, input_output, _ = load_apps_problem(results["problem_id"])
            test_input = input_output["inputs"][0] if input_output.get("inputs") else "Sample input"
        
        # Pass both answers to the majority voter
        print(f"Running majority vote with input: {test_input[:50]}...")
        vote_result = majority_vote([red1_cleaned, red2_cleaned], test_input)
        
        majority_vote_results = {
            "performed": True,
            "matched": vote_result["matched"],
            "output": vote_result["output"],
            "all_outputs": vote_result["all_outputs"]
        }
        
        if vote_result["matched"]:
            print("Majority vote PASSED: Both solutions produced identical outputs")
            print(f"Agreed output: {vote_result['output'][:100]}..." if len(vote_result["output"]) > 100 else f"Agreed output: {vote_result['output']}")
        else:
            print("Majority vote FAILED: Solutions produced different outputs")
    else:
        print("Skipping majority vote since one or both solutions failed their tests")
    
    # Update results with majority voting information
    results["majority_vote"] = majority_vote_results
    
    # Save updated results
    save_test_results(Path(results["results_folder"]), results)
    
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
    print(f"Results saved to: {results['results_folder']}")
    print("\n" + "="*80 + "\n")

    print("\n" + "-"*40 + "\n")
    print("MAJORITY VOTE RESULTS:")
    if "majority_vote" in results:
        if results["majority_vote"].get("performed", False):
            print(f"Outputs match: {results['majority_vote']['matched']}")
            if results['majority_vote']['matched']:
                print(f"Agreed output: {results['majority_vote']['output']}")
            else:
                print("Solutions produced different outputs")
                # Optionally show what each solution produced
                for i, output in results['majority_vote']['all_outputs']:
                    print(f"  Solution {i+1} output: {output[:50]}..." if len(output) > 50 else output)
        else:
            print("Majority vote not performed (solutions failed tests)")
    else:
        print("Majority vote phase not executed yet")

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
    
    # Use the global PROBLEM_ID
    global PROBLEM_ID
    print(f"Using problem ID: {PROBLEM_ID}")
    
    try:
        results = solve_problem_from_apps(PROBLEM_ID)
        pretty_print_results(results)
        if results["red_team_1"]["passed_all_tests"] and results["red_team_2"]["passed_all_tests"]:
            # Load the problem again to get the first test input
            _, input_output, _ = load_apps_problem(PROBLEM_ID)
            test_input = input_output["inputs"][0]  # Use the first test input
            
            print(f"Using test input from unit tests: {test_input[:50]}..." if len(test_input) > 50 else test_input)
            
            # Apply majority voting with this test input
            updated_results = apply_majority_voting(results, test_input)
            
            # See updated results
            pretty_print_results(updated_results)

    except Exception as e:
        print(f"Error running majority vote protocol: {e}")
        import traceback
        traceback.print_exc()
# Script entry point
if __name__ == "__main__":
    main()
