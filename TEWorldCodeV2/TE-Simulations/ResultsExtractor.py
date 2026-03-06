import pandas as pd
from glob import glob
from RunSimulations import EXPERIMENTS_PATH
from enum import Enum

# Extracts results from all of the TE-Experiments folders, reporting the following results:
#   - TE extinction
#   - Host extinction
#   - TE peristence
#   - Cancelled early

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

class ResultsExtractor:
    def __init__(self):
        pass
    
    # Analyzes a single file and determines the corresponding result
    def analyze_file(self, path):
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
    
    # Analyzes all files within the given folder and return results
    def analyze_folder(self, path):
        folder_names = sorted(glob(EXPERIMENTS_PATH), reverse=True)
        
        # Obtain all of the CSV files within the folder
        for folder in folder_names:
            folder_path = f"{EXPERIMENTS_PATH_SHORT}/{folder}/*.csv"
            files = sorted(glob(folder_path))
            
            experiment_results = [0, 0, 0, 0]
            
            for file in files:
                trial_result = self.analyze_file(file)
                
                if trial_result == TEResult.TE_EXTINCTION:
                    experiment_results[0] += 1
                elif trial_result == TEResult.HOST_EXTINCTION:
                    experiment_results[1] += 1
                elif trial_result == TEResult.TE_PERSISTENCE:
                    experiment_results[2] += 1
                else:
                    experiment_results[3] += 1
                    
            # TODO Store 

    
if __name__ == "__main__":
    extractor = ResultsExtractor()
    
    
    extractor.analyze_file("../../TE-Experiments/IS-LLLLLLLLLL-EXP/trace-003.csv")