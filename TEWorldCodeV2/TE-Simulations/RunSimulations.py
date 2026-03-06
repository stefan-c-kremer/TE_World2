from concurrent.futures import ThreadPoolExecutor
from glob import glob
import subprocess

# Iterates through each TE configuration folder, running all of the experiments and stores them in trace files.
# It leverages parallel processing to take advantage of the associated performance enhancements.

N_PASS_THROUGHS = 3
EXPERIMENTS_PATH = "../../TE-Experiments/**"
MAX_PARALLEL = 48

def simulate_all_experiments() -> None:
    """
    Runs simulations for all experiments with parallel processing.
    """
    # Sorting the folders such that "low" folders appear earlier
    folder_names = sorted(glob(EXPERIMENTS_PATH), reverse=True)
        
    # Runs up to MAX_PARALLEL experiments in parallel
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        executor.map(run_experiment, folder_names)
        
def run_experiment(folder_name: str) -> None:
    """
    Runs an simulation for an indvidual experiment.
    """
    print(f"Running experiment in {folder_name}...")
    subprocess.run(['python3', "../../TEWorldCodeV2/TESim.py"], cwd=folder_name)
    print(f"Running experiment in {folder_name}.")

if __name__ == "__main__":
    for i in range(1, N_PASS_THROUGHS + 1):
        print(f"Pass through #{i} is starting...")
        simulate_all_experiments()
        print(f"Pass through #{i} is finished.")
        
    print("Completed all simulations!")
