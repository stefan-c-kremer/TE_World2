import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from glob import glob
from ResultsAnalyzer import ResultsAnalyzer, TEResult

# Importing functions from parent directory (using different method to avoid relative path issues)
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parents[1])
sys.path.append(parent_dir)
print(parent_dir)
from TESim import output

# Iterates through each TE configuration folder, running all of the experiments and stores them in trace files.
# It leverages parallel processing to take advantage of the associated performance enhancements.

N_PASS_THROUGHS = 1
EXPERIMENTS_PATH = "../../TE-Experiments/**"
MAX_PARALLEL = 32

def run(args: list[str]) -> None:
    run = None
    max_generations = None
    
    # Parse out specified simulation run overrides
    if len(args) > 1:
        for i, arg in enumerate(args):
            if arg == "-r":
                run = int(args[i + 1]) # assume that the next value is an integer
            elif arg == "-g":
                max_generations = int(args[i + 1]) # assume that the next value is an integer
            
    # If no run is specified, write results to run #1 by default
    if not run:
        run = 1
    
    # -s indicates skipping validation prompt (for sharknet.sh batch script)
    if len(args) == 1 or "-s" not in args:

        # Safety validation script, to prevent accidental runs
        output("BULK SIM", f"Are you sure you want to run (experiment run #{run}) directly from the terminal (y/n)?")
        res = input("> ")
        
        if res.lower() != "y":
            return
        
    output("BULK SIM", "Starting simulations...")
    
    print("RESULTS THRESHOLD", max_generations)
    simulate_all_experiments(run, max_generations)
        
    output("BULK SIM", "Completed all simulations!")


def simulate_all_experiments(run: int = 1, max_generations: int|None = None) -> None:
    """
    Runs simulations for all experiments with parallel processing.
    run: override parameter to specify an explict experimental run. In this case, it only runs experiments in that experimental run that have no trace-<run>-<iteration>.csv file, or experiments that have not finished.
    """
    
    def get_generation(name):
        """
        Return the current generation of the simulation, from the Excel spreadsheet.
        """
        trace_glob_path = f"{name}/trace-{run:03d}-???.csv"
        trace_paths = sorted(glob(trace_glob_path), reverse=True)
        trace_path = ""
        
        if len(trace_paths) > 0:
            trace_path = trace_paths[0]
            
        analyzer = ResultsAnalyzer()
        generations = None
        
        try:
            generations = analyzer.analyze_file(trace_path)["generations"]
        except Exception:
            generations = 0
        
        return generations
    
    # Sorting the folders such that "low" folders appear earlier
    folder_names = sorted(glob(EXPERIMENTS_PATH), reverse=True)
    analyzer = ResultsAnalyzer()
    total_exp_count = len(folder_names)
    
    # If `run` has been specified, filter out all of the folders that already have a corresponding CSV file where an experiment is finished
    if run:
        output("BULK SIM", f"Specified experimental run #{run}. Will filter out uncompleted experiments.")
        filtered_names = []
        
        for name in folder_names:
            # Identify all trace files for a given run, and then pick the latest one, if available
            trace_glob_path = f"{name}/trace-{run:03d}-???.csv"
            trace_paths = sorted(glob(trace_glob_path), reverse=True)
            trace_path = ""
            
            if len(trace_paths) > 0:
                trace_path = trace_paths[0]
            
            # Obtain result, and mark to be re-run if error occurs (i.e. file does not exist)
            try:
                result = analyzer.analyze_file(trace_path, results_threshold=max_generations)["result"]
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
    
    # Sort based on generation
    folder_names.sort(key=get_generation)
    
    output("BULK SIM", f"{exp_run_count}/{total_exp_count} valid experiments will be run.")
    output("BULK SIM", f"Starting experimental run #{run}...")
    run_in_parallel(run_experiment, folder_names, run)
    output("BULK SIM", f"Finished experimental run #{run}.")
            
def run_in_parallel(func: Callable, names: list[str], run: int) -> None:
    """
    Helper function to run jobs in parallel.
    """
    # Create experimental run arguments
    runs = [run for _ in range(len(names))]
    
    # Runs up to MAX_PARALLEL experiments in parallel
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        executor.map(func, names, runs)
        
def run_experiment(folder_name: str, run: int) -> None:
    """
    Runs an simulation for an indvidual experiment.
    """
    output("BULK SIM", f"Running experiment in {folder_name}...")
    completed_process = subprocess.run(['python3', "../../TEWorldCodeV2/TESim.py", str(run), folder_name], cwd=folder_name)
    return_code = completed_process.returncode
    output("BULK SIM", f"Finished experiment in {folder_name} with return code: {return_code}.")

if __name__ == "__main__":
    run(sys.argv)