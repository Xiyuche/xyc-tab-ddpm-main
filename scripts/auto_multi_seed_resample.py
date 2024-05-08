import subprocess
import os
import concurrent.futures


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
        return e


def main():
    datasets = ["abalone"]
    known_rates = [0.9]  # Example known rates
    u_j_combinations = [(1, 1), (2, 2), (5, 5), (10, 10), (15, 15)]  # Example u, j values

    # List to store all commands
    commands = []
    seeds_num = 5
    for seed in range(seeds_num):
        for dataset in datasets:
            config_path = f"exp/{dataset}/ddpm_cb_best/config.toml"
            for rate in known_rates:
                dir_name = f"{dataset}-exp-{int(rate * 100):03d}"
                exp_dir = f"AutoResample_{dataset}_seed_{seed}/{dir_name}"
                ensure_directory_exists(exp_dir)
                new_mask = 1 if not os.path.exists(
                    f"{exp_dir}/Mask_{int(rate * 100):03d}.npy") else 0  # new_mask is aborted

                for (u, j) in u_j_combinations:
                    command = f"python scripts/pipeline.py --config {config_path} --sample --exp_dir {exp_dir} --u_times {u} --jump_length {j} --new_mask {new_mask} --probability_known {rate} --seed {seed}"
                    commands.append(command)

    # Use ThreadPoolExecutor or ProcessPoolExecutor depending on preference
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        # Map the run_command to all commands, execute them in parallel
        results = list(executor.map(run_command, commands))

    # Check for errors in results
    errors = [result for result in results if result is not None]
    if errors:
        print(f"Errors occurred in {len(errors)} commands.")
    else:
        print("All commands executed successfully without errors.")


if __name__ == "__main__":
    main()

