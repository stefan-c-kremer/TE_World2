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
        
    print("Starting simulations...")
    
    simulate_all_experiments(N_PASS_THROUGHS, iter_override)
        
    print("Completed all simulations!")


def simulate_all_experiments(n_iters: int, iter: int = 1) -> None:
    """
    Runs simulations for all experiments with parallel processing.
    n_iters: number of iterations to be run, can be overwritten by iter
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
            # Identify all trace files for a given run, and then pick the latest one, if available
            trace_glob_path = f"{name}/trace-{iter:03d}-???.csv"
            trace_paths = sorted(glob(trace_glob_path), reverse=True)
            trace_path = ""
            
            if len(trace_paths) > 0:
                trace_path = trace_paths[0]
            
            # Obtain result, and mark to be re-run if error occurs (i.e. file does not exist)
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
    print(f"Starting iteration #{iter}...")
    run_in_parallel(run_experiment, folder_names, iter)
    print(f"Finished iteration #{iter}.")
            
def run_in_parallel(func: Callable, names: list[str], iter: int) -> None:
    """
    Helper function to run jobs in parallel.
    """
    # Create iteration arguments
    iters = [iter for _ in range(len(names))]
    
    # Runs up to MAX_PARALLEL experiments in parallel
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        executor.map(func, names, iters)
        
def run_experiment(folder_name: str, iter: int) -> None:
    """
    Runs an simulation for an indvidual experiment.
    """
    print(f"Running experiment in {folder_name}...")
    subprocess.run(['python3', "../../TEWorldCodeV2/TESim.py", str(iter)], cwd=folder_name)
    print(f"Running experiment in {folder_name}.")

if __name__ == "__main__":
    run(sys.argv)