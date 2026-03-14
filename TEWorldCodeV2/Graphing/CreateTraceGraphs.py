import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt

csv2key = {
    'pop_size': 'hosts', 'LTETOTAL': 'TEs', 'LTEAUT': 'Autonomous TEs', 
    'LTENAUT': 'Non-autonomous TEs', 'LETHAL_J': 'Lethal', 'DELETE_J': 'Delet.', 
    'NEUTRA_J': 'Neutral', 'BENEFI_J': 'Benfit.', 'TEDEATH': 'TEs', 
    'COLLISIO': 'TEs', 'TOTAL_JU': 'TEs'
}

csv2lt = {
    'pop_size': '#FF0000', 'LTETOTAL': '#FF0000', 'LTEAUT': "#00FF26", 
    'LTENAUT': "#C300FF", 'DELETE_J': '#00FF00', 'NEUTRA_J': '#0000FF', 
    'BENEFI_J': '#000000', 'LETHAL_J': '#FF0000'
}

plots_config = [
    ( 'Host Population vs Generation', 'gen', ['pop_size'] ),
    ( 'Total Live TEs vs Generation', 'gen', ['LTETOTAL'] ),
    ( 'Live TE Percentiles vs Generation', 'gen', [ 'LTE100pe',
                           'LTE075pe', 'LTE050pe', 'LTE025pe', 'LTE000pe' ] ),
    ( 'Total Dead TEs vs Generation', 'gen', ['DTETOTAL'] ),
    ( 'Dead TE Percentiles vs Generation', 'gen', [ 'DTE100pe',
                           'DTE075pe', 'DTE050pe', 'DTE025pe', 'DTE000pe' ] ),
    ( 'Fitness Percentiles vs Generation', 'gen', [ 'FIT100pe',
                           'FIT075pe', 'FIT050pe', 'FIT025pe', 'FIT000pe' ] ),
    ( 'TE Deaths vs Generation', 'gen', [ 'TEDEATH' ] ),
    ( 'TE Collisions vs Generation', 'gen', [ 'COLLISIO' ] ),
    ( 'TE Jumps vs Generation', 'gen', [ 'TOTAL_JU' ] ),
    ( 'TE Jump Effects vs Generation', 'gen', ['LETHAL_J', 'DELETE_J', 'NEUTRA_J', 'BENEFI_J' ] ),
    ( 'TE and Gene Locations', 'gen', ['GSIZE100','GSIZE075','GSIZE050','GSIZE025','GSIZE000', 'GELOC100','GELOC075','GELOC050','GELOC025','GELOC000', 'TELOC100','TELOC075','TELOC050','TELOC025','TELOC000' ] ),
    ( 'Live (autonomous and non-autonomous) TEs vs Generation', 'gen', ['LTEAUT', 'LTENAUT']) 
]

class Trial:
    def __init__(self, trial_id):
        self.trial_id = trial_id
        self.file_name = f"{trial_id}.csv" # Assuming implicitly that the trace files are in the current directory
        self.df = pd.read_csv(self.file_name)
        
        # Standardize column names (strip whitespace)
        self.df.columns = [c.strip() for c in self.df.columns]

    def plot_all(self, configs):
        for title, x_col, y_cols in configs:
            self.create_plot(title, x_col, y_cols)

    def create_plot(self, title, x_col, y_cols):
        # Check if required columns exist in this specific CSV
        available_y = [y for y in y_cols if y in self.df.columns]
        if not available_y or x_col not in self.df.columns:
            return

        # Create figure (mimicking your 800,200 size)
        plt.figure(figsize=(8, 2)) 
        
        for y in available_y:
            plt.plot(
                self.df[x_col], 
                self.df[y], 
                label=csv2key.get(y, y), 
                color=csv2lt.get(y, '#000000'),
                linewidth=1
            )

        plt.title(title, fontsize=10, family='sans-serif')
        plt.legend(loc='upper left', fontsize=8)
        plt.tight_layout()
        
        # Save as SVG
        output_name = f"{title}-{self.trial_id}.svg"
        plt.savefig(output_name, format='svg')
        plt.close()

def main():
    # Find files like trace-001.csv
    files = sys.argv[1:]
    
    print(f"{len(files)} will be processed and graphed. Please wait...")
    
    # Allows for multiple files via command line arguments
    for file_name in files:
        trial_id = file_name.split(".")[0]
        print(f"Processing file {file_name}")
        trial = Trial(trial_id)
        trial.plot_all(plots_config)

        # Generate HTML Summary
        with open(f"graphs-{trial_id}.html", "w") as html:
            html.write(f"<html><body><h1> {trial_id} </h1>\n")
            for title, _, _ in plots_config:
                img_name = f"{title}-{trial_id}.svg"
                if os.path.exists(img_name):
                    html.write(f'<p><img src="{img_name}"/></p>\n')
            html.write("</body></html>")
            
    print("Finished creating graphs for all input files.")

if __name__ == "__main__":
    main()