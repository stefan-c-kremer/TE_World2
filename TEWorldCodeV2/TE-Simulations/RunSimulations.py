import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from glob import glob

# Iterates through each TE configuration folder, running all of the experiments and stores them in trace files.
# It leverages parallel processing to take advantage of the associated performance enhancements.

N_PASS_THROUGHS = 1
EXPERIMENTS_PATH = "../../TE-Experiments/**"
MAX_PARALLEL = 120

def run(args: list[str]) -> None:
    # -s indicates skipping validation prompt (for sharknet.sh batch script)
    if len(args) == 1 or args[1] != "-s":
        # Safety validation script, to prevent accidental runs
        print("Are you sure you want to run directly from the terminal (y/n)?")
        res = input("> ")
        
        if res.lower() != "y":
            return
    
    # -f for fast mode
    fast_mode = False
    
    if len(args) > 1 and args[1] == "-f":
        fast_mode = True
        
    print("Starting simulations...")
    
    simulate_all_experiments(N_PASS_THROUGHS, fast_mode)
        
    print("Completed all simulations!")


def simulate_all_experiments(n_interations: int, fast_mode: bool = False) -> None:
    """
    Runs simulations for all experiments with parallel processing.
    fast_mode: enables iterations to be run simulataneously, instead of one after another
    """
    
    # Sorting the folders such that "low" folders appear earlier
    folder_names = sorted(glob(EXPERIMENTS_PATH), reverse=True)
    
    # In fast mode, all iterations are grouped together
    if fast_mode:
        names = []
        
        # Adding all of the folder names together into an array (n times), such that we can run the iterations togeter
        for _ in range(n_interations):
            names += folder_names
            
        run_in_parallel(run_experiment, names)
    else:
        for i in range(1, 1 + n_interations):
            print(f"Starting iteration {i}/{n_interations}...")
            run_in_parallel(run_experiment, folder_names)
            print(f"Finished iteration {i}/{n_interations}.")
            
def run_in_parallel(func: Callable, names: list[str]):
    """
    Helper function to run jobs in parallel.
    """
    # Runs up to MAX_PARALLEL experiments in parallel
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        executor.map(func, names)
        
def run_experiment(folder_name: str) -> None:
    """
    Runs an simulation for an indvidual experiment.
    """
    print(f"Running experiment in {folder_name}...")
    subprocess.run(['python3', "../../TEWorldCodeV2/TESim.py"], cwd=folder_name)
    print(f"Running experiment in {folder_name}.")

if __name__ == "__main__":
    run(sys.argv)