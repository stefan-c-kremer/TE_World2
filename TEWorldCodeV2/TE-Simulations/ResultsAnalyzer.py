import yaml
import re
import shutil
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, PatternFill
from glob import glob
from RunSimulations import EXPERIMENTS_PATH
from enum import Enum
from CreateTraceGraphs import Trial

"""
Extracts results from all of the TE-Experiments folders, reporting the following results:
  - TE extinction
  - Host extinction
  - TE peristence
  - Cancelled early
  
Additionally, it creates an Excel graph for this data.
"""

EXPERIMENTS_PATH_SHORT = "../../TE-Experiments"
LIVE_TE_COL = " LTETOTAL"
POP_SIZE_COL = " pop_size"
GEN_COL = "      gen"
MAX_GENS = 1500
MAX_VALUE_LEN = 16

class TEResult(Enum):
    TE_EXTINCTION = 1
    HOST_EXTINCTION = 2
    TE_PERSISTENCE = 3
    OTHER = 4 # normally the run was cancelled mid-way

TE_EXTINCTION_COLOUR = "FFFF00"
HOST_EXTINCTION_COLOUR = "A02B93"
TE_PERSISTENCE_COLOUR = "00B0F0"

plots_config = [
    ( 'Live (autonomous and non-autonomous) TEs vs Generation', 'gen', ['LTEAUT', 'LTENAUT']) 
]

OUTPUT_PATH = "output/"


class ResultsAnalyzer:
    def __init__(self):
        self.results = None
    
    def analyze_file(self, path: str) -> dict:
        """
        Analyzes a given trace (result) file, and returns the corresponding result and generation count.
        """
        
        # Read CSV into data frame
        df = pd.read_csv(path)
        
        analysis = {
            
        }
        
        # Check conditions to see what result it should be classified
        try:
            result_type = None
            
            if df.iloc[-1][POP_SIZE_COL] == 0:
                result_type = TEResult.HOST_EXTINCTION
            elif df.iloc[-1][LIVE_TE_COL] == 0:
                result_type = TEResult.TE_EXTINCTION
            elif df.iloc[-1][GEN_COL] == MAX_GENS:
                result_type = TEResult.TE_PERSISTENCE
            else:
                result_type = TEResult.OTHER
                
            analysis = {
                "result": result_type,
                "generations": int(df.iloc[-1][GEN_COL])
            }
    
        # If there is some sort of file-specific exception, return OTHER
        except Exception:            
            analysis = {
                "result": TEResult.OTHER,
                "generations": 0
            }
        
        return analysis
    
    def analyze_experiment(self, path: str) -> dict:
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
            
            if trial_result["result"] == TEResult.TE_EXTINCTION:
                experiment_results["TE_EXTINCTION"] += 1
            elif trial_result["result"] == TEResult.HOST_EXTINCTION:
                experiment_results["HOST_EXTINCTION"] += 1
            elif trial_result["result"] == TEResult.TE_PERSISTENCE:
                experiment_results["TE_PERSISTENCE"] += 1
            else:
                experiment_results["OTHER"] += 1
                
        return experiment_results
            
    def analyze_experiments(self) -> None:
        """
        Analyzes all experiments, and stores a representation of the results.
        """
        # These correspond to the graph names
        names = []
        top_names = []
        right_names = []
        parasitism_names = []
        te_extinction_counts = []
        host_extinction_counts = []
        te_persistence_counts = []
        other_counts = []
        
        name_pattern = re.compile('[HL]{10}')
        
        folder_names = sorted(glob(EXPERIMENTS_PATH), reverse=True)
        
        for folder in folder_names:
            experiment_result = self.analyze_experiment(folder)
            
            # Obtain names of experiments, corresponding to graph
            experiment_name = name_pattern.findall(folder)[0]
            top_name = experiment_name[0:4]
            right_name = experiment_name[4:8]
            parasitism_name = experiment_name[8:10]
            
            # Data storage mechanisms
            names.append(experiment_name)
            top_names.append(top_name)
            right_names.append(right_name)
            parasitism_names.append(parasitism_name)
            te_extinction_counts.append(experiment_result["TE_EXTINCTION"])
            host_extinction_counts.append(experiment_result["HOST_EXTINCTION"])
            te_persistence_counts.append(experiment_result["TE_PERSISTENCE"])
            other_counts.append(experiment_result["OTHER"])

        # Storage in data frame
        data = {
            "name": names,
            "top_name": top_names,
            "right_name": right_names,
            "parasitism_names": parasitism_names,
            "te_extinction_count": te_extinction_counts,
            "host_extinction_count": host_extinction_counts,
            "te_persistence_count": te_persistence_counts,
            "other_count": other_counts
        }
        
        self.results = pd.DataFrame(data)
        
    def export_results_to_excel(self, save_path="results.xlsx", parasitism_fields="LL"):
        """
        Exports results to an excel file
        """
        
        results = self.results
        
        if parasitism_fields:
            results = self.results[self.results.parasitism_names == parasitism_fields]
        
        results.to_excel(save_path)
    
    def export_results_to_graph(self, save_path="graph.xlsx", parasitism_fields="LL") -> None:
        """
        Takes the data corresponding to the parasitism fields and exports an excel file for basic visualization, 
        similar to the original paper.
        """
        wb = Workbook()
        ws = wb.active
        
        # Set up row and column headers of worksheet
        self.set_up_worksheet(ws, parasitism_fields)
        self.insert_data_into_worksheet(ws, parasitism_fields)
        
        # Insert data into worksheet
      
        wb.save(save_path)
        
    def get_result_fill(self, row):
        """
        Computes and returns a cell fill type based on the most frequent results.
        """
        max_category = None
            
        # Determine the most frequent result
        if row.te_extinction_count > row.host_extinction_count:
            if row.te_extinction_count > row.te_persistence_count:
                max_category = TEResult.TE_EXTINCTION
            else:
                max_category = TEResult.TE_PERSISTENCE
        else:
            if row.host_extinction_count > row.te_persistence_count:
                max_category = TEResult.HOST_EXTINCTION
            else:
                max_category = TEResult.TE_PERSISTENCE
                
        if max_category == TEResult.TE_EXTINCTION:
            return PatternFill(start_color=TE_EXTINCTION_COLOUR, end_color=TE_EXTINCTION_COLOUR, fill_type='solid')
        elif max_category == TEResult.HOST_EXTINCTION:
            return PatternFill(start_color=HOST_EXTINCTION_COLOUR, end_color=HOST_EXTINCTION_COLOUR, fill_type='solid')
        else:
            return PatternFill(start_color=TE_PERSISTENCE_COLOUR, end_color=TE_PERSISTENCE_COLOUR, fill_type='solid')
        
    def get_all_te_persistence_experiments(self) -> list[str]:
        """
        Obtains all TE persistence experiments and returns their corresponding names to be identified in folders.
        """
        return self.results[self.results.te_persistence_count > 0]["name"].values
    
    def output_te_persistence_detailed_results(self, output_path: str) -> None:
        """
        Obtains all trace files corresponding with an experiment that had TE persistence, and copies them to an output path.
        Also runs the individual trace graphing functionality.
        """
        persistence_names = self.get_all_te_persistence_experiments()
        
        # Obtain all of the CSV files within all of the files that returned TE persistence results
        # Copy these corresponding files in the folder
        for name in persistence_names:
            # Create glob file path to obtain CSV files
            glob_path = f"{EXPERIMENTS_PATH_SHORT}/IS-{name}-EXP/*.csv"
            trace_files = sorted(glob(glob_path))
            
            # Copy trace files for the corresponding experiment that has at least one TE persistence result
            for i, trace_file_name in enumerate(trace_files):
                output_name = f"{name}-{i + 1}"  
                trace_output_path = f"{output_path}{output_name}.csv"     
                shutil.copy(trace_file_name, trace_output_path)
                
                # After copying the file, we will create graphs (the file needs to be in the graphing folder to create a graph)
                trial = Trial(output_name, trace_output_path, OUTPUT_PATH)
                trial.plot_all(plots_config)
                
       
    def insert_data_into_worksheet(self, ws, parasitism_fields: str) -> None:
        """
        Inserts data into the worksheet, corresponding to the parasitism fields.
        """
        # Obtain all results corresonding to the parasitism fields
        # Obtain results with mapped fields
        matched_results = self.results[
                (self.results.parasitism_names == parasitism_fields) & 
                (
                    (self.results.te_extinction_count > 0) | 
                    (self.results.host_extinction_count > 0) | 
                    (self.results.te_persistence_count > 0)
                )
            ]
        
        # Iterate through each result
        for row in matched_results.itertuples():
            col_name = "Q"
            
            parameter_size = MAX_VALUE_LEN // 2
            
            # Obtain column by traversing letters in reverse
            for letter in reversed(row.top_name):
                if letter == "H":
                    col_name = chr(ord(col_name) - parameter_size)
                    
                parameter_size //= 2
            
            parameter_size = MAX_VALUE_LEN // 2
            
            # Obtain row by traversing letters in reverse
            row_name = 20
            for letter in reversed(row.right_name):
                if letter == "H":
                    row_name -= parameter_size
                    
                parameter_size //= 2
                    
            cell_name = f"{col_name}{str(row_name)}"

            # Insert data
            ws[cell_name].fill = self.get_result_fill(row)
        
    def set_up_worksheet(self, ws, parasitism_fields: str) -> None:
        """
        Sets up column and row headers for a worksheet, and doesn't save the file (to be done in parent function).
        """
        ws.title = f"Results ({parasitism_fields})"
        center_alignment = Alignment(horizontal="center", vertical="center")
        
        # Centre-align rows by default
        for row in ws.iter_rows(min_row=1, max_row=20, min_col=1, max_col=21):
            for cell in row:
                cell.alignment = center_alignment
        
        # Create configurations
        with open("changeable-configurations.yaml", "r") as fp:
            mappings = yaml.safe_load(fp)["configuration_mappings"]
            
        graph_field_names = []
            
        for field in mappings:
            # Should only have 1 key
            for key in field.keys():
                field_name = key
            
            graph_field_names.append(field_name)
            
        top_field_names = graph_field_names[0:4]
        right_field_names = graph_field_names[4:8]
        
        # Write top field names
        for i, field_name in enumerate(top_field_names):
            ws[f"A{i + 1}"] = field_name
            
        # Write right field names
        for i, field_name in enumerate(right_field_names):
            field_letter = chr(ord("R") + i)
            start_cell_name = f"{field_letter}1"
            end_cell_name = f"{field_letter}4"
            ws[start_cell_name] = field_name
            
            # Merge cells
            ws.merge_cells(f"{start_cell_name}:{end_cell_name}")
            
        # Implement top field cell merging
        merge_size = MAX_VALUE_LEN // 2
        row = 1
        
        # Merge cells together, and add column header values
        while merge_size >= 1:
            start_cell_name = f"B{row}"
            cell_val = "H"
            
            while start_cell_name < "R1":
                end_cell_name = f"{chr(ord(start_cell_name[0]) + merge_size - 1)}{start_cell_name[1]}"
                cell_range = f"{start_cell_name}:{end_cell_name}"
                ws.merge_cells(cell_range)
                
                # Add cell value
                ws[start_cell_name] = cell_val
                
                if cell_val == "H":
                    cell_val = "L"
                else:
                    cell_val = "H"
                
                start_cell_name = f"{chr(ord(end_cell_name[0]) + 1)}{end_cell_name[1]}"
                
            row += 1
            merge_size //= 2
            
        col = 0
        merge_size = MAX_VALUE_LEN // 2
            
        # Merge cells together, and add row header values
        while merge_size >= 1:
            start_cell_name = f"{chr(ord('U') - col)}5"
            cell_val = "H"
            
            while int(start_cell_name[1:]) <= 20:
                end_cell_name = f"{start_cell_name[0]}{int(start_cell_name[1:]) + merge_size - 1}"
                cell_range = f"{start_cell_name}:{end_cell_name}"
                
                ws.merge_cells(cell_range)
                
                # Add cell value
                ws[start_cell_name] = cell_val
                
                if cell_val == "H":
                    cell_val = "L"
                else:
                    cell_val = "H"
                
                start_cell_name = f"{end_cell_name[0]}{int(end_cell_name[1:]) + 1}"
                
            col += 1
            merge_size //= 2
            
        # Adjust alignment of right headers
        for row in ws.iter_rows(1, 20, 18, 21):
            for cell in row:
                cell.alignment = Alignment(horizontal="center", vertical="center", text_rotation=180)
  
            
if __name__ == "__main__":
    extractor = ResultsAnalyzer()
    extractor.analyze_experiments()
    
    new_config_permutations = ["LL", "LH", "HL", "HH"]
    
    for permutation in new_config_permutations:
        extractor.export_results_to_excel(f"output/results-{permutation.lower()}.xlsx", permutation)
        extractor.export_results_to_graph(f"output/graph-{permutation.lower()}.xlsx", permutation)
    
    extractor.output_te_persistence_detailed_results(OUTPUT_PATH)