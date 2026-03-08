import pandas as pd
import re
from glob import glob
from RunSimulations import EXPERIMENTS_PATH
from enum import Enum

"""
Extracts results from all of the TE-Experiments folders, reporting the following results:
  - TE extinction
  - Host extinction
  - TE peristence
  - Cancelled early
  
Additionally, it creates an Excel graph for this data.
"""

EXPERIMENTS_PATH_SHORT = "../../TE-Experiments"
TE_EXTINCT_COL = " LTETOTAL"
HOST_EXTINCT_COL = " pop_size"
TE_PERSISTENCE_COL = "      gen"
MAX_GENS = 1500

class TEResult(Enum):
    TE_EXTINCTION = 1
    HOST_EXTINCTION = 2
    TE_PERSISTENCE = 3
    OTHER = 4 # normally the run was cancelled mid-way

class ResultsAnalyzer:
    def __init__(self):
        pass
    
    def analyze_file(self, path: str) -> TEResult:
        """
        Analyzes a given trace (result) file, and returns the corresponding result.
        """
        
        # Read CSV into data frame
        df = pd.read_csv(path)
        
        # Check conditions to see what result it should be classified
        if df.iloc[-1][TE_EXTINCT_COL] == 0:
            return TEResult.TE_EXTINCTION
        elif df.iloc[-1][HOST_EXTINCT_COL] == 0:
            return TEResult.HOST_EXTINCTION
        elif df.iloc[-1][TE_PERSISTENCE_COL] == MAX_GENS:
            return TEResult.TE_PERSISTENCE
        else:
            return TEResult.OTHER
    
    def analyze_experiment(self, path: str):
        """
        Analyzes an individual experiment and returns a result.
        """
        # Obtain all of the CSV files within the folder
        folder_path = f"{path}/*.csv"
        files = sorted(glob(folder_path))
        
        experiment_results = {
            "TE_EXTINCTION": 0,
            "HOST_EXTINCTION": 0,
            "TE_PERSISTENCE": 0,
            "OTHER": 0
        }
        
        for file in files:
            trial_result = self.analyze_file(file)
            
            if trial_result == TEResult.TE_EXTINCTION:
                experiment_results["TE_EXTINCTION"] += 1
            elif trial_result == TEResult.HOST_EXTINCTION:
                experiment_results["HOST_EXTINCTION"] += 1
            elif trial_result == TEResult.TE_PERSISTENCE:
                experiment_results["TE_PERSISTENCE"] += 1
            else:
                experiment_results["OTHER"] += 1
                
        return experiment_results
            
    def analyze_experiments(self):
        """
        Analyzes all experiments, and stores a representation of the results.
        """
        # These correspond to the graph names
        names = []
        top_names = []
        right_names = []
        te_extinction_counts = []
        host_extinction_counts = []
        te_persistence_counts = []
        other_counts = []
        
        name_pattern = re.compile('[HL]{10}')
        
        folder_names = sorted(glob(EXPERIMENTS_PATH), reverse=True)
        
        for folder in folder_names:
            folder_path = f"{EXPERIMENTS_PATH_SHORT}/{folder}"
            experiment_result = self.analyze_experiment(folder_path)
            
            # Obtain names of experiments, corresponding to graph
            experiment_name = name_pattern.findall(folder)[0]
            top_name = experiment_name[0:5]
            right_name = experiment_name[5:10]
            
            # Data storage mechanisms
            names.append(experiment_name)
            top_names.append(top_name)
            right_names.append(right_name)
            te_extinction_counts.append(experiment_result["TE_EXTINCTION"])
            host_extinction_counts.append(experiment_result["HOST_EXTINCTION"])
            te_persistence_counts.append(experiment_result["TE_PERSISTENCE"])
            other_counts.append(experiment_result["OTHER"])

        # Storage in data frame
        data = {
            "name": names,
            "top_name": top_names,
            "right_name": right_names,
            "te_extinction_count": te_extinction_counts,
            "host_extinction_count": host_extinction_counts,
            "te_persistence_count": te_persistence_counts,
            "other_count": other_counts
        }
        
        return pd.DataFrame(data) 
            
if __name__ == "__main__":
    extractor = ResultsAnalyzer()
    results = extractor.analyze_experiments()
    
    print(f"The results have been obtained:")
    print(results)