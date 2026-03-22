import numpy as np
import yaml
import re
import shutil
import pandas as pd
import matplotlib.axes as ax
import matplotlib.pyplot as plt
from math import sqrt
from openpyxl import Workbook
from openpyxl.styles import Alignment, PatternFill
from glob import glob
from RunSimulations import EXPERIMENTS_PATH
from enum import Enum
from CreateTraceGraphs import Trial
import matplotlib.gridspec as gs
import matplotlib.patches as pa

"""
Extracts results from all of the TE-Experiments folders, reporting the following results:
  - TE extinction
  - Host extinction
  - TE peristence
  - Cancelled early
  
Additionally, it creates an Excel graph for this data.
"""

EXPERIMENTS_PATH_SHORT = "../../TE-Experiments"
MAX_GENS = 1500
DIM = 16
HEADER_DIM = 4
N_PARAMS = int(sqrt(DIM))
GEN_COL = "      gen"
FIG_HEIGHT = 20
FIG_WIDTH = 24


class TEResult(Enum):
    TE_EXTINCTION = 1 # both autonomous and non-autonomous go extinct
    TE_NAUT_PERSISTENCE = 2 # autonomous TEs go extinct
    TE_AUT_PERSISTENCE = 3 # non-autonomous TEs go extinct
    HOST_EXTINCTION = 4
    TE_PERSISTENCE = 5 # autonomous and non-autonomous persist
    OTHER = 6 # normally the run was cancelled mid-way
    
# Stores important information needed for data extraction, storage and graph relevant to all TE scenarios
SCENARIO_MAPPINGS = {
    TEResult.TE_EXTINCTION.value: {
            "column": " LTETOTAL",
            "count_name": "te_extinction_count",
            "colour": "FFFF00"
        },
    TEResult.TE_NAUT_PERSISTENCE.value: {
            "column": "   LTEAUT",
            "count_name": "te_naut_persistence_count",
            "colour": "FF0000",
        },
    TEResult.TE_AUT_PERSISTENCE.value: {
            "column": "  LTENAUT",
            "count_name": "te_aut_persistence_count",
            "colour": "FF6F00"
        },
    TEResult.HOST_EXTINCTION.value: {
            "column": " pop_size",
            "count_name": "host_extinction_count",
            "colour": "A02B93"
        },
    TEResult.TE_PERSISTENCE.value: {
            "column": GEN_COL,
            "count_name": "te_persistence_count",
            "colour": "00B0F0"
    }
}

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
        
        # Check conditions to see what result it should be classified
        try:
            result_type = None
            
            tes_persisted = df.iloc[-1][SCENARIO_MAPPINGS[TEResult.TE_PERSISTENCE.value]["column"]] == MAX_GENS
            
            # The order of conditional statements matters (i.e. host extinction should be checked first)
            if df.iloc[-1][SCENARIO_MAPPINGS[TEResult.HOST_EXTINCTION.value]["column"]] == 0:
                result_type = TEResult.HOST_EXTINCTION
            elif df.iloc[-1][SCENARIO_MAPPINGS[TEResult.TE_EXTINCTION.value]["column"]] == 0:
                result_type = TEResult.TE_EXTINCTION
            elif df.iloc[-1][SCENARIO_MAPPINGS[TEResult.TE_NAUT_PERSISTENCE.value]["column"]] == 0 and tes_persisted:
                result_type = TEResult.TE_NAUT_PERSISTENCE
            elif df.iloc[-1][SCENARIO_MAPPINGS[TEResult.TE_AUT_PERSISTENCE.value]["column"]] == 0 and tes_persisted:
                result_type = TEResult.TE_AUT_PERSISTENCE
            elif tes_persisted: # both types of TEs persisted
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
            "TE_NAUT_PERSISTENCE": 0,
            "TE_AUT_PERSISTENCE": 0,
            "HOST_EXTINCTION": 0,
            "TE_PERSISTENCE": 0,
            "OTHER": 0
        }
        
        for file in files:
            trial_result = self.analyze_file(file)
            
            if trial_result["result"] == TEResult.TE_EXTINCTION:
                experiment_results["TE_EXTINCTION"] += 1
            elif trial_result["result"] == TEResult.TE_NAUT_PERSISTENCE:
                experiment_results["TE_NAUT_PERSISTENCE"] += 1
            elif trial_result["result"] == TEResult.TE_AUT_PERSISTENCE:
                experiment_results["TE_AUT_PERSISTENCE"] += 1
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
        te_naut_persistence_counts = []
        te_aut_persistence_counts = []
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
            te_naut_persistence_counts.append(experiment_result["TE_NAUT_PERSISTENCE"])
            te_aut_persistence_counts.append(experiment_result["TE_AUT_PERSISTENCE"])
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
            "te_naut_persistence_count": te_naut_persistence_counts,
            "te_aut_persistence_count": te_aut_persistence_counts,
            "host_extinction_count": host_extinction_counts,
            "te_persistence_count": te_persistence_counts,
            "other_count": other_counts
        }
        
        self.results = pd.DataFrame(data)
        
    def export_results_to_excel(self, save_path="results.xlsx", parasitism_names="LL"):
        """
        Exports results to an excel file
        """
        
        results = self.results
        
        if parasitism_names:
            results = self.results[self.results.parasitism_names == parasitism_names]
        
        results.to_excel(save_path)
        
    def export_results_to_plot(self, save_path="graph.png",  parasitism_names="LL") -> None:
        """
        Takes the data and outputs the graph to a PNG file, similar to the original paper.
        """
        
        # Create HEIGHT * WIDTH grid of sub-figures of fixed sizes (DIM + HEADER_DIM additional cells for headers, etc.)
        fig = plt.figure(figsize=(20, 20))
        grid = gs.GridSpec(FIG_HEIGHT, FIG_WIDTH)
        ax_main = fig.add_subplot(grid[HEADER_DIM:FIG_HEIGHT, HEADER_DIM:FIG_WIDTH - HEADER_DIM])
        ax_top_left = fig.add_subplot(grid[0:HEADER_DIM, 0:HEADER_DIM])
        ax_top = fig.add_subplot(grid[0:HEADER_DIM, HEADER_DIM:FIG_WIDTH - HEADER_DIM])
        ax_top_right = fig.add_subplot(grid[0:HEADER_DIM, FIG_WIDTH - HEADER_DIM:FIG_WIDTH])
        ax_right = fig.add_subplot(grid[HEADER_DIM:FIG_HEIGHT, FIG_WIDTH - HEADER_DIM:FIG_WIDTH])
        
        # Hide the background of the label areas so they look like empty space
        ax_top_left.axis('off')
        ax_top.axis('off')
        ax_top_right.axis('off')
        ax_right.axis('off')
        
        ax_main.set_xlim(0, DIM) 
        ax_main.set_ylim(0, DIM)
        x_labels, y_labels = self.get_plot_tick_labels()
        label_pos = [i for i in range(DIM)]

        # Set tick values
        ax_main.set_xticks(label_pos, x_labels)
        ax_main.set_yticks(label_pos, y_labels)
        
        # Obtain results and fill in graph
        self.fill_in_plot_graph_header_titles(ax_top_left, ax_top_right)
        self.fill_in_plot_graph_headers(ax_top, ax_right)
        matched_results = self.get_relevant_results(parasitism_names)
        self.fill_in_plot_graph(fig, ax_main, matched_results, parasitism_names)
        
        # Save figure
        fig.savefig(save_path)
        
    def get_plot_tick_labels(self) -> tuple[np.array, np.array]:
        """
        Get x/y tick labels.
        """
        x_labels = []
        
        for row in range(DIM):
            x_labels.append(self.convert_number_to_partial_experiment_name(row))
            
        y_labels = list(reversed(x_labels))
            
        # y-labels are the same as the x-labels, but in opposite order
        return np.array(x_labels), np.array(y_labels)
    
    def fill_in_plot_graph_headers(self, ax_top: ax.Axes, ax_right: ax.Axes) -> None:
        """
        Fill in header (H/L) values.
        """
        # Set dimensions
        ax_top.set_xlim(0, DIM)
        ax_top.set_ylim(0, HEADER_DIM)
        ax_right.set_xlim(0, HEADER_DIM)
        ax_right.set_ylim(0, DIM)
        
        # Create top H/L boxes
        width = int(DIM // 2)
        row = 3

        while width >= 1:
            col = 0
            label = "H"
            
            while col < DIM:
                rect = pa.Rectangle((col, row), width, 1, edgecolor='black', facecolor='none')
                ax_top.add_patch(rect)
                
                # Add text in the centere of the box
                ax_top.text(col + width/2, row + 0.5, label, ha='center', va='center')
                
                # Set-up for next column
                col += width
                if label == "H":
                    label = "L"
                else:
                    label = "H"
                    
            width = int(width // 2)
            row -= 1
            
        height = int(DIM // 2)
        col = 3

        while height >= 1:
            row = 0
            label = "L"
            
            while row < DIM:
                rect = pa.Rectangle((col, row), 1, height, edgecolor='black', facecolor='none')
                ax_right.add_patch(rect)
                
                # Add text in the centere of the box
                ax_right.text(col + 0.5, row + height/2, label, ha='center', va='center', rotation=270)
                
                # Set-up for next column
                row += height
                if label == "H":
                    label = "L"
                else:
                    label = "H"
                    
                
                    
            height = int(height // 2)
            col -= 1
        
    def fill_in_plot_graph_header_titles(self, ax_top: ax.Axes, ax_right: ax.Axes) -> None:
        """
        Fills in headers of plot graph.
        """
        # Fill in row headers
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
        
        # Adjust axes to better align with the headers
        ax_top.set_xlim(0, 1)
        ax_top.set_ylim(0, HEADER_DIM)
        
        ax_right.set_xlim(0, HEADER_DIM)
        ax_right.set_ylim(0, 1)
        
        # Add a bunch of subplots
        for i, name in enumerate(top_field_names):
            ax_top.text(0.25, i, name)
            
        for i, name, in enumerate(right_field_names):
            ax_right.text(i, 0.25, name, rotation=270)
        
    def fill_in_plot_graph(self, fig, axs, matched_results: pd.DataFrame, parasitism_names="LL") -> None:
        """
        Fills in sub-figures of graph with their corresponding experimental results.
        """
        for row in range(DIM):
            for col in range(DIM):
                # Row adjustment is to align with matplotlib
                name = self.convert_number_to_experiment_name(DIM - row - 1, col, parasitism_names)
                
                # Fill in the boxes with respect to their proportions
                result_proportions = self.get_scenario_proportions(name, matched_results)
                self.fill_in_fig_with_stacked_bars(fig, axs, row, col, result_proportions)
                
    def fill_in_fig_with_stacked_bars(self, fig, axs, row: int, col: int, props: dict) -> None:
        """
        Fills in individual sub-figures with stacked bar graphs to represent proportional results.
        """
        bottom = row
        
        # Iterate through all the scenario mappings with an ordered dictionary, filling in the corresponding colours like a bar chart
        for key, scenario_items in SCENARIO_MAPPINGS.items():
            prop = props[key]
            
            axs.bar(col + 0.5, prop, bottom=bottom, width=0.8, color=f"#{scenario_items['colour']}")
            bottom += prop
    
    def get_scenario_proportions(self, name: str, matched_results: pd.DataFrame) -> dict:
        """
        Obtains proportions of each scenario for each entry.
        """
        # Dictionaries in Python are now sorted, so this can be traversed through consistently
        props = {
            TEResult.HOST_EXTINCTION.value: 0,
            TEResult.TE_AUT_PERSISTENCE.value: 0,
            TEResult.TE_EXTINCTION.value: 0,
            TEResult.TE_NAUT_PERSISTENCE.value: 0,
            TEResult.TE_PERSISTENCE.value: 0
        }
        
        total_count = 0
        
        # Add counts, before turning into proportion
        for key, scenario_values in SCENARIO_MAPPINGS.items():
            try:
                count_value = matched_results.loc[matched_results.name == name, scenario_values["count_name"]].item()
                props[key] = count_value
                total_count += count_value
            # Error handling in case no value was obtained
            except ValueError:
                pass
            
        # If there were proper results, obtain the proportion
        if total_count > 0:
            for key in props.keys():
                props[key] /= total_count
        
        return props
                
    def convert_number_to_experiment_name(self, row: int, col: int, parasitism_names: str) -> str:
        return self.convert_number_to_partial_experiment_name(col) + self.convert_number_to_partial_experiment_name(row) + parasitism_names
                
    def convert_number_to_partial_experiment_name(self, pos: int) -> str:
        """
        Used the row/col integer to return the corresponding experiment name (row-wise or column-wise).
        """
        name = ""
        modulus = 2
        
        while modulus <= DIM:
            half_modulus = int(modulus // 2)
            res = pos % modulus
            
            if res < half_modulus:
                name += "H"
            else:
                name += "L"
            
            modulus *= 2
        
        return name
        
    def get_all_te_persistence_experiments(self) -> list[str]:
        """
        Obtains all TE persistence experiments and returns their corresponding names to be identified in folders.
        """
        return self.results[(self.results.te_persistence_count > 0) | (self.results.te_naut_persistence_count > 0) | (self.results.te_aut_persistence_count > 0)]["name"].values
    
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
                
       
    def get_relevant_results(self, parasitism_names: str) -> pd.DataFrame:
        """
        Get results that have at least one non-TEResult.OTHER result, and match the parasitism fields.
        """
        return self.results[
                        (self.results.parasitism_names == parasitism_names) & 
                        (
                            (self.results.te_extinction_count > 0) | 
                            (self.results.host_extinction_count > 0) | 
                            (self.results.te_persistence_count > 0) | 
                            (self.results.te_naut_persistence_count > 0) |
                            (self.results.te_aut_persistence_count > 0)
                        )
                    ]
        
if __name__ == "__main__":
    extractor = ResultsAnalyzer()
    extractor.analyze_experiments()
    
    new_config_permutations = ["LL", "LH", "HL", "HH"]
    
    for permutation in new_config_permutations:
        extractor.export_results_to_excel(f"output/results-{permutation.lower()}.xlsx", permutation)
        extractor.export_results_to_plot(f"output/graph-{permutation.lower()}.png", permutation)
    
    extractor.output_te_persistence_detailed_results(OUTPUT_PATH)