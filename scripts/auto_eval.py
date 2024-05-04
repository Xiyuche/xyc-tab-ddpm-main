import csv
import json
import subprocess
import os


def ensure_directory_exists(directory):
    """Ensure that a directory exists, and if not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def run_command(command):
    """Run a command using subprocess and capture its output."""
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, text=True)
        print(f"Successfully executed: {command}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing {command}: {e}")
        return None


import re


def parse_evaluation_results(output):
    """Extract evaluation results from command output using regex after 'EVAL RESULTS:'"""
    results = {}
    eval_results_section = output.split('EVAL RESULTS:')[1] if 'EVAL RESULTS:' in output else ''
    matches = re.finditer(r'\[([a-z]+)\]\s*\{([^}]+)\}', eval_results_section)

    for match in matches:
        section = match.group(1)
        data = eval('{' + match.group(2) + '}')
        results[section] = data

    return results


import csv


def save_results_to_json(data, filename='evaluation_results.json'):
    """Save parsed evaluation results to a JSON file."""
    try:
        with open(filename, 'r+') as file:
            # First we load existing data into a dict.
            file_data = json.load(file)
            # Join new_data with file_data inside emp_details
            file_data.append(data)
            # Sets file's current position at offset.
            file.seek(0)
            # convert back to json.
            json.dump(file_data, file, indent=4)
    except FileNotFoundError:
        with open(filename, 'w') as file:
            # If not exists, create the file and write the data
            json.dump([data], file, indent=4)

def transform_resample():
    # datasets = ["abalone", "adult", "california", "buddy", "cardio", "churn2", "default", "diabetes", "fb-comments",
                # "gesture", "higgs-small", "house", "insurance", "king", "miniboone", "wilt"]
    datasets = ["abalone", "adult"]
    known_rates = [0.6, 0.9]
    u_j_combinations = [(1, 1), (2, 2), (3, 3)]

    for dataset in datasets:
        config_path = f"exp/{dataset}/ddpm_cb_best/config.toml"
        for rate in known_rates:
            dir_name = f"{dataset}-exp-{int(rate * 100):03d}"
            exp_dir = f"AutoResample/{dir_name}"
            for (u, j) in u_j_combinations:
                file_name = f"Resample_{u}u_{j}j.npy"
                file_path = os.path.join(exp_dir, file_name)
                if os.path.exists(file_path):
                    command = (f"python scripts/eval_seeds.py --config {config_path} 10 ddpm synthetic catboost 5 "
                               f"--file_path {file_path}")
                    print("Executing:", command)
                    output = run_command(command)
                    if output:
                        results = parse_evaluation_results(output)
                        data = {
                            'dataset': dataset,
                            'u': u,
                            'j': j,
                            'known_rate': rate,
                        }
                        # Dynamically adding each metric from 'val' and 'test' sections
                        for section, metrics in results.items():
                            for key, value in metrics.items():
                                data[f'{section}_{key}'] = value

                        save_results_to_json(data)


if __name__ == "__main__":
    transform_resample()
