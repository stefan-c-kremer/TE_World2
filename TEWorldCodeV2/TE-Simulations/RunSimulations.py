import sys
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from glob import glob
from ResultsAnalyzer import ResultsAnalyzer, TEResult

# Iterates through each TE configuration folder, running all of the experiments and stores them in trace files.
# It leverages parallel processing to take advantage of the associated performance enhancements.

N_PASS_THROUGHS = 1
EXPERIMENTS_PATH = "../../TE-Experiments/**"
MAX_PARALLEL = 120

def run(args: list[str]) -> None:
    # -s indicates skipping validation prompt (for sharknet.sh batch script)
    if len(args) == 1 or "-s" not in args:
        # Safety validation script, to prevent accidental runs
        print("Are you sure you want to run directly from the terminal (y/n)?")
        res = input("> ")
        
        if res.lower() != "y":
            return
        
    iter_override = None
        
    # Parse out specified iteration overrides
    if len(args) > 1:
        for i, arg in enumerate(args):
            if arg == "-i":
                iter_override = int(args[i + 1]) # assume that the next value is an integer
    
    # -f for fast mode
    fast_mode = False
    
    if len(args) > 1 and "-f" in args:
        fast_mode = True
        
    print("Starting simulations...")
    
    simulate_all_experiments(N_PASS_THROUGHS, fast_mode, iter_override)
        
    print("Completed all simulations!")


def simulate_all_experiments(n_iters: int, fast_mode: bool = False, iter: int|None = None) -> None:
    """
    Runs simulations for all experiments with parallel processing.
    n_iters: number of iterations to be run, can be overwritten by iter
    fast_mode: enables iterations to be run simulataneously, instead of one after another
    iter: override parameter to specify an explict iteration to run. In this case, it only runs experiments in that iteration that have no trace-<iter>.csv file.
    """
    
    # Sorting the folders such that "low" folders appear earlier
    folder_names = sorted(glob(EXPERIMENTS_PATH), reverse=True)
    analyzer = ResultsAnalyzer()
    total_exp_count = len(folder_names)
    
    # If `iter` has been specified, filter out all of the folders that already have a corresponding CSV file where an experiment is finished
    if iter:
        print(f"Specified iteration #{iter}. Will filter out uncompleted experiments.")
        filtered_names = []
        
        for name in folder_names:
            trace_path = f"{name}/trace-{iter:03d}.csv"
            
            # Obtain result, and mark to be re-run if error occurs
            try:
                result = analyzer.analyze_file(trace_path)["result"]
            except Exception:
                result = TEResult.OTHER
                
            
            # For experiments that have not yet been started
            if len(glob(trace_path)) == 0:
                filtered_names.append(name)
            # For experiments that were started, but not finished
            # Removes existing CSV file for trial, and replaces it with a new (truncated) one
            elif result == TEResult.OTHER:
                filtered_names.append(name)
                
        # Replace folder_names with the filtered_names value
        folder_names = filtered_names
        
    exp_run_count = total_exp_count - (total_exp_count - len(folder_names))
    
    print(f"{exp_run_count}/{total_exp_count} valid experiments will be run.")
    
    # In fast mode, all iterations are grouped together
    # Cannot be performed when a specific iteration is specified to be run
    if fast_mode and not iter:
        names = []
        
        # Adding all of the folder names together into an array (n times), such that we can run the iterations togeter
        for _ in range(n_iters):
            names += folder_names
            
        run_in_parallel(run_experiment, names)
    else:
        for i in range(1, 1 + n_iters):
            print(f"Starting iteration {i}/{n_iters}...")
            run_in_parallel(run_experiment, folder_names)
            print(f"Finished iteration {i}/{n_iters}.")
            
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