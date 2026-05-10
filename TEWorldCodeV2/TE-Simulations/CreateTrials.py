import sys
import yaml
import os
from glob import glob

"""
Creates Python configuration files for TE experiments, with high and low parameters.
"""

EXPERIMENTS_PATH = "../../TE-Experiments"

# These are configurations that stay the same for all trials
unchanged_fields = """
from TEUtil import *;

seed = None;   # if seed = None, the random number generator's initial state is set "randomly"

output = {
	"SPLAT":		False,
	"SPLAT FITNESS":	False,
	"INITIALIZATION":	False,
	"GENERATION":		True,
	"HOST EXTINCTION":	True,
	"TE EXTINCTION":	True,
        "TRIAL NO":		True,
        "GENE INIT":		False,
	"TE INIT":		False,
	"BULK SIM":     True
	};

Gene_length = 1000;
TE_length = lambda autonomous : 6000 if autonomous else 300;
						 
Append_gene = True;	# True: when the intialization routine tries to place 
			# a gene inside another gene, it instead appends it
			# at the end of the original gene (use this with small
			# amounts of Junk_BP).
			# False:  when the intialization routine tries to place
			# a gene inside another gene, try to place it somewhere
			# else again (don't use theis option with samll amounts
			# of Junk_BP).

Initial_Aut_TEs = 1;

TE_excision_rate = 0.0; # set to zero as LINE and SINE are retrotransposons (copy/paste)

Host_start_fitness = 1.0;
		
Host_reproduction_rate = 1;  # how many offspring each host has

Host_survival_rate = lambda propfit: min( Carrying_capacity * propfit, 0.95 );
    # propfit = proportion of fitness owned by this individual

Maximum_generations = 1500;
Terminate_no_TEs = True;	# end simulation if there are no TEs left

save_frequency = 50;    # Frequency with with which to save state of experiment
"""

def generate_num_strings(n):
    """
    Generates all numeric stirngs of length n.
    This is not technically binary as 2 now represents the zero mapping.
    """
    permutations = []
    bit_format = "0{}b".format(n)
    
    # Generate binary strings
    for i in range(2**n):
        permutations.append(format(i, bit_format))
        
    short_bit_format = bit_format = "0{}b".format(n - 1) # 1 less bit than the bit_format variable
        
    # Generate all possible n - 1 binary strings, and then add zero at the end for special case (initial non-autonomous TEs)
    for i in range(2**(n - 1)):
        permutations.append(f"{format(i, short_bit_format)}2")
    
    return permutations

def is_high(permutation, i):
    return permutation[i] == "1"

def is_low(permutation, i):
    return permutation[i] == "0"

def get_saved_field(name: str, run: int|None = None):
    """
    Returns the configured field for `saved`
    """
    state_file_name = None
    
    # If an experimental run override is specified, we want to obtain the latest state file associated with that run
    if run:
        state_glob_path = f"{EXPERIMENTS_PATH}/IS-{name}-EXP/state-{run:03d}-???-???????.gz"
        
        # Implicitly sorts in descending order, first by run, than by the latest state file
        state_files = sorted(glob(state_glob_path), reverse=True)
        
        # Take the most recent state file, and use it for future runs
        if len(state_files) > 0:
            state_file_name = state_files[0].split("/")[-1]

    if state_file_name:
        return f"saved = '{state_file_name}'"
    
    return "saved = None"

def generate_configurations(run: int|None = None):
    """
    Generates all configurations and returns them as strings.
    """
    
    configuration_mappings = get_configuration_mappings()["configurations"]
    n_changeable_configurations = len(configuration_mappings)
    mapping_split = n_changeable_configurations # for the purposes of replicating the graph
    permutations = generate_num_strings(n_changeable_configurations)
    configurations = []
    
    # Will use each binary string to turn the respective configuration on/off
    for permutation in permutations:
        configuration_name = ""
        configuration_body = unchanged_fields
        
        configuration_body += "\n# ********************************************"
        configuration_body += "\n# TRIAL FIELDS\n"

        # Obtain the configuration, and append the corresponding value, based on the flag
        for i, config in enumerate(configuration_mappings):
            # Update name
            # If halfway through the dictionary, add a hyphen to mark the difference between columns and rows
            if i == mapping_split:
                configuration_name += "-"
        
            if is_high(permutation, i):
                configuration_name += "H"
            elif is_low(permutation, i):
                configuration_name += "L"
            else:
                configuration_name += "Z"
                
            # Add comment
            configuration_body += f"# Parameter: {config['name']}\n"
                          
            # Iterate through 
            for parameter in config["parameters"]:
                configuration_body += f"{parameter['id']} = "
                
                # Adding changeable configuration values, based on configured configuration_mappings
                if is_high(permutation, i):
                    configuration_body += str(parameter["high"])
                elif is_low(permutation, i):
                    configuration_body += str(parameter["low"])
                else:
                    configuration_body += str(parameter["zero"])
                    
                configuration_body += "\n"
                
            configuration_body += "\n"
     
        # Add `saved` field, after name has been determined
        configuration_body += get_saved_field(configuration_name, run)
     
        # Add comment about dynamic configuration generation, for debugging purposes
        configuration_body += "\n\n# This configuration file was programmatically generated."
        configuration_body += "\n# Used permutation '{}', which corresponds to '{}'. Reference the configuration_mappings in the configuration file to determine what is 'high' and what is 'low'".format(permutation, configuration_name)
                
        configuration = {
            "name": configuration_name,
            "body": configuration_body
        }
                
        configurations.append(configuration)
        
    return configurations

def create_configuration_files(run: int|None = None):
    """
    Creates configuration files.
    """
    print("Generating configurations...")
    configurations = generate_configurations(run)
    print("Finished generating configurations.")
    
    
    # Create TE-Experiments folder, if it does not exist
    if not os.path.exists(EXPERIMENTS_PATH):
        os.mkdir(EXPERIMENTS_PATH)
    
    # Write all of the configurations to files in the TE-Experiments directory
    print("Writing trial configurations to files...")
    for configuration in configurations:
        name = configuration["name"]
        body = configuration["body"]
        dir_path = f"{EXPERIMENTS_PATH}/IS-{name}-EXP"
        config_path = f"{dir_path}/parameters.py"
        
        print("Writing trial configuration to {}...".format(config_path))
        
        # If the directory does not exist, create it
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        
        with open(config_path, "w") as fp:
            fp.write(body)
            
        print("Wrote trial configuration to {}.".format(config_path))
            
    print("Finished writing trial configurations.")
            

def get_configuration_mappings():
    """
    Obtains configuration configuration_mappings from the YAML file, and returns them as a dictionary.
    """
    configuration_mappings = None
    
    with open("parameters.yaml", "r") as fp:
        configuration_mappings = yaml.safe_load(fp)
    
    return configuration_mappings
        
if __name__ == "__main__":
    run_override = None
    
    if len(sys.argv) > 1 and "-r" in sys.argv:
        run_override = int(sys.argv[2])
        
    create_configuration_files(run=run_override)