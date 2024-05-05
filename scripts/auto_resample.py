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


def main():
    datasets = ["abalone", "adult", "california", "buddy", "cardio", "churn2", "diabetes", "fb-comments",
                "gesture", "higgs-small", "house", "insurance", "king", "miniboone", "wilt"]
    known_rates = [0.6, 0.9]  # Example known rates
    u_j_combinations = [(1, 1), (2, 2)]  # Example u, j values

    for dataset in datasets:
        config_path = f"exp/{dataset}/ddpm_cb_best/config.toml"
        for rate in known_rates:
            dir_name = f"{dataset}-exp-{int(rate * 100):03d}"
            exp_dir = f"AutoResample/{dir_name}"
            ensure_directory_exists(exp_dir)

            # Assume a new mask is needed if it's the first time using this known rate for this dataset
            new_mask = 1 if not os.path.exists(f"{exp_dir}/Mask_{int(rate * 100):03d}.npy") else 0

            for (u, j) in u_j_combinations:
                command = f"python scripts/pipeline.py --config {config_path} --sample --exp_dir {exp_dir} --u_times {u} --jump_length {j} --new_mask {new_mask} --probability_known {rate}"
                print("Executing:", command)  # Printing command for debugging
                run_command(command)


if __name__ == "__main__":
    main()
