"""The oMLX twin of ollama_function_support.py: the same function-calling scorecard (10 prompts x 10 runs per model,
8 tools, chained calls), but it tests the chat models served by oMLX's OpenAI-compatible API instead of Ollama's.

Needs oMLX running on :8000 with some chat models downloaded, and OMLX_API_KEY in the environment or the repo-root
.env. Needs `pip install openai requests tabulate python-dotenv`.

    python omlx_function_support.py                       # every suitable chat model (<= 10 GB)
    python omlx_function_support.py mlx-community--Qwen3.5-4B-MLX-4bit mlx-community--granite-4.1-3b-4bit
    python omlx_function_support.py --include-big         # also the big ones, e.g. Qwen3.8-27B-oQ4e-mtp (slow)

OMLX_INCLUDE_BIG=1 in the environment does the same as --include-big.
"""
import argparse
import inspect
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import requests
import time
import statistics
from tabulate import tabulate
import csv
from typing import Dict, Any, List, get_type_hints
import math
from pathlib import Path

HERE = Path(__file__).parent

load_dotenv()

api_key = os.environ.get("OMLX_API_KEY")
if not api_key:
    raise SystemExit("Set OMLX_API_KEY (environment or the repo-root .env) to your oMLX API key.")
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=api_key)

# Listed by oMLX but only partly downloaded, so it can't be loaded
EXCLUDED_MODELS = {"mlx-community--gemma-4-31b-it-4bit"}


def convert_to_roman_numerals(n: int) -> str:
    """Converts a number into Roman numerals"""
    if n == 2025:
        return "MMXXV"
    elif n == 100:
        return "C"
    else:
        return "Wrong Year"


def convert_fahrenheit_to_centigrade(f: float) -> float:
    return (f-32)/9*5


convert_fahrenheit_to_centigrade_tool = {
    'type': 'function',
    'function': {
        'name': 'convert_fahrenheit_to_centigrade',
        'description': 'Converts the temperature given in Fahrenheit to Centigrade',
        'parameters': {
            'type': 'object',
            'required': ['f'],
            'properties': {
                'f': {'type': 'float', 'description': 'The temperature in Fahrenheit'},
            },
        },
    },
}


def day_of_the_week() -> str:
    """Get the current day of the week"""
    from datetime import datetime
    return datetime.now().strftime('%A')


def calc_trig_function(function_name: str, angle: int) -> float:
    """Calculate sin/cos/tan for an angle in degrees"""
    rad = math.radians(angle)

    if function_name.lower() == 'sin':
        return round(math.sin(rad), 4)
    elif function_name.lower() == 'cos':
        return round(math.cos(rad), 4)
    elif function_name.lower() == 'tan':
        return round(math.tan(rad), 4)
    else:
        raise ValueError("Function must be 'sin', 'cos', or 'tan'")


def weather_tool(location: str) -> str:
    """Find the weather for a specific location"""
    if location.lower() == 'paris':
        return "Pas Mal"
    else:
        return "Merd!"


def time_tool() -> str:
    """Return the current time"""
    return "Now"


def location_tool() -> str:
    """Returns the current location"""
    return "Here"


def distance_tool(location: str) -> int:
    """Distance to the city or town provided"""
    if location.lower() == 'berlin':
        return 12345
    else:
        return 0


JSON_TYPES = {int: 'integer', float: 'number', str: 'string', bool: 'boolean'}


def function_to_tool(fn) -> Dict[str, Any]:
    """Build the OpenAI tool schema from a Python function's signature, type hints and docstring
    (the ollama SDK did this for us when we passed it the functions themselves)."""
    hints = get_type_hints(fn)
    params = inspect.signature(fn).parameters
    return {
        'type': 'function',
        'function': {
            'name': fn.__name__,
            'description': inspect.getdoc(fn) or '',
            'parameters': {
                'type': 'object',
                'required': [name for name, p in params.items() if p.default is p.empty],
                'properties': {name: {'type': JSON_TYPES[hints[name]]} for name in params},
            },
        },
    }


def get_available_models() -> list:
    try:
        # /v1/models/status is /v1/models plus each model's type and estimated size in bytes
        response = requests.get('http://127.0.0.1:8000/v1/models/status',
                                headers={'Authorization': f'Bearer {api_key}'})
        models = response.json()['models']
        return models
    except requests.RequestException as e:
        print(f"Error listing models: {e}")
        return []


def is_suitable_model(model: Dict[str, Any], include_big: bool = False) -> bool:
    return ((include_big or model["estimated_size"] <= 10_000_000_000)
            and model["model_type"] in ("llm", "vlm")      # chat models only: no embedding/rerank/audio
            and model["id"] not in EXCLUDED_MODELS
            )


def run_single_test(prompt: str, expected_result: int, model: str):
    start_time = time.time()
    results = {
        'model': model,
        'prompt': prompt,
        'success': False,
        'error': None,
        'raw_response': None,
        'function_details': []
    }

    available_functions = {
        'convert_to_roman_numerals': convert_to_roman_numerals,
        'convert_fahrenheit_to_centigrade': convert_fahrenheit_to_centigrade,
        'day_of_the_week': day_of_the_week,
        'calc_trig_function': calc_trig_function,
        'weather_tool': weather_tool,
        'time_tool': time_tool,
        'location_tool': location_tool,
        'distance_tool': distance_tool,
    }

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            tools=[function_to_tool(f) for f in (
                convert_to_roman_numerals, convert_fahrenheit_to_centigrade,
                day_of_the_week, calc_trig_function,
                weather_tool, time_tool,
                location_tool, distance_tool,
            )],
            temperature=0.6,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        message = response.choices[0].message

        results['raw_response'] = message

        if message.tool_calls:
            last_result = None
            for tool in message.tool_calls:
                if function_to_call := available_functions.get(tool.function.name):
                    # Chain tool calls: feed previous result into the first numeric arg
                    args = json.loads(tool.function.arguments or '{}')   # a JSON string in the OpenAI API
                    if last_result is not None:
                        for key, value in args.items():
                            if isinstance(value, (int, float)):
                                args[key] = last_result
                                break

                    function_result = function_to_call(**args)
                    last_result = function_result

                    args_str = ', '.join(f"{k}={v}" for k, v in args.items())
                    results['function_details'].append({
                        'name': tool.function.name,
                        'args': args_str,
                        'result': function_result
                    })

            results['success'] = last_result == expected_result

    except Exception as e:
        results['error'] = str(e)

    execution_time = (time.time() - start_time) * 1000
    return results, execution_time


def test_model(model: str, num_runs: int = 10) -> Dict[str, Any]:
    from datetime import datetime
    current_day = datetime.now().strftime('%A')

    test_cases = [
        ('What is 2025 in Roman numerals?', "MMXXV"),
        ('What is 212 degrees F in C?', 100),
        ('What is the day of the week today?', current_day),
        ('What is the weather in the capital of France?', "Pas Mal"),
        ('Is it going to be sunny in on the Paris tomorrow?', "Pas Mal"),
        ('What is the time?', "Now"),
        ('What is my location?', "Here"),
        ('How far is it to Berlin?', 12345),
        ('What is the cosine of 60 degrees?', 0.5000),
        ('What is 212 degrees F in C converted to Roman numerals', "C"),
    ]

    print(f"\nTesting {model}...")

    print("  Performing warmup run...", end=' ', flush=True)
    warmup_result, _ = run_single_test(test_cases[0][0], test_cases[0][1], model)
    status = "PASS" if warmup_result['success'] else "FAIL"
    if warmup_result['function_details']:
        function_calls = []
        for fd in warmup_result['function_details']:
            function_calls.append(f"{fd['name']}({fd['args']}) --> {fd['result']}")
        print(f"{status}: {' -> '.join(function_calls)}")
    else:
        print(status)

    all_results = []
    execution_times = []

    for test_case in test_cases:
        prompt, expected = test_case
        print(f"Testing prompt: {prompt}")

        for i in range(num_runs):
            print(f"  Run {i + 1}/{num_runs}...", end=' ', flush=True)
            result, execution_time = run_single_test(prompt + " /no_think", expected, model)
            all_results.append(result)
            execution_times.append(execution_time)

            status = "PASS" if result['success'] else "FAIL"
            if result['function_details']:
                function_calls = []
                for fd in result['function_details']:
                    function_calls.append(f"{fd['name']}({fd['args']}) --> {fd['result']}")
                print(f"{status}: {' -> '.join(function_calls)}")
            else:
                print(status)

    success_count = sum(1 for r in all_results if r['success'])
    total_possible = len(test_cases) * num_runs

    final_result = {
        'model': model,
        'execution_time': statistics.mean(execution_times),
        'execution_time_std': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
        'success_rate': success_count,
        'total_possible': total_possible,
        'error': all_results[-1].get('error', None),
        'raw_response': all_results[-1].get('raw_response', None)
    }

    return final_result


def format_results_table(results: List[Dict[str, Any]]) -> str:
    headers = ['Model', 'Size (GB)', 'Time (ms)', 'σ Time', 'Success Rate']
    table_data = []

    for result in results:
        model_size_gb = result.get('model_size', 0) / 1_000_000_000
        success_rate = f"{result['success_rate']}/{result['total_possible']}"
        row = [
            result['model'],
            f"{model_size_gb:.2f}",
            f"{result.get('execution_time', 0):.0f}",
            f"±{result.get('execution_time_std', 0):.0f}",
            success_rate
        ]
        table_data.append(row)

    return tabulate(table_data, headers=headers, tablefmt='grid')


def write_results_csv(results: List[Dict[str, Any]], filename: str = None):
    if filename is None:
        filename = str(HERE / 'omlx_function_results.csv')
    headers = ['Model', 'Size_GB', 'Time_ms', 'Time_StdDev', 'Success_Rate', 'Total_Possible']

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for result in results:
            model_size_gb = result.get('model_size', 0) / 1_000_000_000
            row = [
                result['model'],
                f"{model_size_gb:.2f}",
                f"{result.get('execution_time', 0):.0f}",
                f"{result.get('execution_time_std', 0):.0f}",
                result['success_rate'],
                result['total_possible']
            ]
            writer.writerow(row)


def format_final_report(results: List[Dict[str, Any]]) -> str:
    report = []

    report.append("\n" + "=" * 60)
    report.append("FUNCTION SUPPORT TEST RESULTS SUMMARY")
    report.append("=" * 60)
    report.append(format_results_table(results))

    report.append("\nDETAILED STATISTICS")
    report.append("=" * 60)

    total_models = len(results)
    perfect_models = []
    other_models = []

    for result in results:
        if result['success_rate'] == result['total_possible']:
            perfect_models.append(result['model'])
        else:
            success_rate = (result['success_rate'] / result['total_possible']) * 100
            other_models.append((result['model'], success_rate))

    report.append(f"\nTotal models tested: {total_models}")
    report.append(f"Perfect score models: {len(perfect_models)}")
    report.append(f"Other models: {len(other_models)}")

    if perfect_models:
        report.append("\nModels with perfect scores:")
        for model in perfect_models:
            report.append(f"  {model}")

    if other_models:
        report.append("\nOther models:")
        for model, rate in other_models:
            report.append(f"  {model}: {rate:.1f}% success rate")

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Score oMLX chat models on function calling.")
    parser.add_argument('models', nargs='*', help="model ids to test (default: every suitable chat model)")
    parser.add_argument('--include-big', action='store_true',
                        default=os.environ.get('OMLX_INCLUDE_BIG', '').lower() in ('1', 'true', 'yes'),
                        help="also test models over 10 GB, e.g. Qwen3.8-27B-oQ4e-mtp (or set OMLX_INCLUDE_BIG=1)")
    cli = parser.parse_args()

    print("Starting function support test suite...\n")

    models = get_available_models()
    if not models:
        print("Error: No models found. Please ensure oMLX is running and models are downloaded.")
        return

    suitable_models = [m for m in models if is_suitable_model(m, cli.include_big)]
    if cli.models:
        for name in set(cli.models) - {m['id'] for m in suitable_models}:
            print(f"Skipping {name}: not a suitable chat model on this server (too big without --include-big?)")
        suitable_models = [m for m in suitable_models if m['id'] in cli.models]

    print(f"Found {len(suitable_models)} suitable models for testing.")
    print("Running tests (this may take a few minutes)...")
    print("=" * 60)

    results = []
    for model in suitable_models:
        result = test_model(model['id'])
        result['model_size'] = model['estimated_size']
        results.append(result)

    print("\n" * 2)
    report = format_final_report(results)
    print(report)

    write_results_csv(results)
    print(f"\nResults have been saved to {HERE / 'omlx_function_results.csv'}")


if __name__ == "__main__":
    main()
