import subprocess
import os


def ensure_directory_exists(directory):
    """Ensure that a directory exists, and if not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def run_command(command):
    """Run a command using subprocess with error handling."""
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"Successfully executed: {command}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing {command}: {e}")


def transform_resample():
    datasets = ["abalone"]
    known_rates = [0.9]  # Example known rates
    u_j_combinations = [(1, 1), (2, 2), (5, 5), (10, 10), (15, 15)]  # Example u, j values
    seeds_num = 5

    for seed in range(seeds_num):
        for dataset in datasets:
            config_path = f"exp/{dataset}/ddpm_cb_best/config.toml"
            for rate in known_rates:
                dir_name = f"{dataset}-exp-{int(rate * 100):03d}"
                exp_dir = f"multi_seed_resample/{dataset}/AutoResample_{dataset}_seed_{seed}/{dir_name}"
                for (u, j) in u_j_combinations:
                    file_name = f"Resample_{u}u_{j}j.npy"
                    file_path = os.path.join(exp_dir, file_name)
                    if os.path.exists(file_path):
                        command = f"python scripts/pipeline.py --config {config_path} --sample --file_path {file_path}"
                        print("Executing:", command)  # Printing command for debugging
                        run_command(command)


if __name__ == "__main__":
    transform_resample()
