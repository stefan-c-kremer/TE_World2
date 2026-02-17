import yaml
import os
import shutil

"""
Creates 256 Python configuration files for TE experiments, with high and low parameters.
"""

"""
TODO: Plan:

1. Set all of the unchangeable parameters in a file, and store them in a literal.
2.  Create a list of all permutations of 8 1s and 0s, to indicate high and low.
3.  For each configuration:
        a. To enable higher configura
        b. Create a new string that appends the unchanged literal to the list of changeable configurations.
        c. Write this to a python file.
"""

# These are configurations that stay the same for all trials
unchanged_fields = """
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
	};

Gene_length = 1000;
TE_length = 1000;
						 
Initial_genes = 500;
Append_gene = True;	# True: when the intialization routine tries to place 
			# a gene inside another gene, it instead appends it
			# at the end of the original gene (use this with small
			# amounts of Junk_BP).
			# False:  when the intialization routine tries to place
			# a gene inside another gene, try to place it somewhere
			# else again (don't use theis option with samll amounts
			# of Junk_BP).

Initial_TEs = 1;

Host_start_fitness = 1.0;
Host_mutation = ProbabilityTable( 0.40, lambda fit: 0.0,
                                  0.30, lambda fit: fit - random.random()*0.01,
                                  0.15, lambda fit: fit,
                                  0.15, lambda fit: fit + random.random()*0.01
                                  );
	
# what happens when a TA hits a gene
Insertion_effect = ProbabilityTable(0.30, lambda fit: 0.0,
                                    0.20, lambda fit: fit - random.random()*0.01,
                                    0.30, lambda fit: fit,
                                    0.20, lambda fit: fit + random.random()*0.01
                                    );
		
Host_reproduction_rate = 1;  # how many offspring each host has

Host_survival_rate = lambda propfit: min( Carrying_capacity * propfit, 0.95 );
    # propfit = proportion of fitness owned by this individual

Maximum_generations = 1500;
Terminate_no_TEs = True;	# end simulation if there are no TEs left

seed = None;   # if seed = None, the random number generator's initial state is
               # set "randomly"

save_frequency = 50;    # Frequency with with which to save state of experiment

saved = None;   # if saved = None then we start a new simulation from scratch
                # if saves = string, then we open that file and resume a simulation
"""

def generate_binary_strings(n):
    """
    Generates all binary stirngs of length n.
    """
    permutations = []
    bit_format = "0{}b".format(n)
    
    for i in range(2**n):
        permutations.append(format(i, bit_format))
        
    return permutations

def is_high(permutation, i):
    return permutation[i] == "1"

def get_configuration_name(binary_string):
    """
    Takes a binary string and obtains a name with it, corresponding to high and low fields.
    TODO: Align this with Dr. Kremer's research paper.
    """
    name = ""
    
    # First half is for the column values
    for i in range(len(binary_string)):
        if is_high(binary_string, i):
            name += "H"
        else:
            name += "L"

    return name

def generate_configurations():
    """
    Generates all configurations and returns them as strings.
    """
    
    mappings = get_configuration_mappings()["configuration_mappings"]
    n_changeable_configurations = len(mappings)
    permutations = generate_binary_strings(n_changeable_configurations)
    configurations = []
    
    # Will use each binary string to turn the respective configuration on/off
    for permutation in permutations:
        configuration_body = unchanged_fields
        
        # Obtain the configuration, and append the corresponding value, based on the flag
        for i, field in enumerate(mappings):
            field_name = field.keys()[0]
            configuration_body += "\n\n"
            configuration_body += "{} = ".format(field_name)
            
            # Adding changeable configuration values, based on configured mappings
            if is_high(permutation, i):
                configuration_body += str(field[field_name]["High"])
            else:
                configuration_body += str(field[field_name]["Low"])
                
        configuration = {
            "name": get_configuration_name(permutation),
            "body": configuration_body
        }
                
        configurations.append(configuration)
        
    return configurations



def create_configuration_files():
    """
    Creates configuration files.
    """
    print("Generating configurations...")
    configurations = generate_configurations()
    print("Finished generating configurations.")
    
    # Write all of the configurations to files in the TE-Experiments directory
    print("Writing trial configurations to files...")
    for configuration in configurations:
        
        name = configuration["name"]
        print(name)
        body = configuration["body"]
        dir_path = "../../TE-Experiments/IS-{}-Exp".format(name)
        config_path = "{}/parameters.py".format(dir_path)
        
        print("Writing trial configuration to {}...".format(config_path))
        
        # If the directory already exists, overwrite it
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            
        os.mkdir(dir_path)
        
        with open(config_path, "w") as fp:
            fp.write(body)
            
        print("Wrote trial configuration to {}.".format(config_path))
            
    print("Finished writing trial configurations.")
            

def get_configuration_mappings():
    """
    Obtains configuration mappings from the YAML file, and returns them as a dictionary.
    """
    mappings = None
    
    with open("changeable-configurations.yaml", "r") as fp:
        mappings = yaml.safe_load(fp)
    
    return mappings
        
if __name__ == "__main__":
    create_configuration_files()