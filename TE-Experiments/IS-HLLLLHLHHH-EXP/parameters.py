
from TEUtil import *;

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

# ********************************************
# TRIAL FIELDS
# Parameter: Insertion Bias
TE_Insertion_Distribution = Triangle( pmax=0, pzero=3.0/3.0 )
Gene_Insertion_Distribution = Triangle( pzero=1.0/3.0, pmax=1 )

# Parameter: TE Death Rate
TE_death_rate = 0.005

# Parameter: TE Excision Rate
TE_excision_rate = 0.1

# Parameter: TE Progeny
TE_progeny = ProbabilityTable( 0.15, 0, 0.55, 1, 0.30, 2 )

# Parameter: Carrying Capacity
Carrying_capacity = 30

# Parameter: Mutation Effect
Host_Mutation = ProbabilityTable( 0.40, lambda fit: 0.0, 0.30, lambda fit: fit - random.random()*0.1, 0.15, lambda fit: fit, 0.15, lambda fit: fit + random.random()*0.1 )
Insertion_effect = ProbabilityTable( 0.40, lambda fit: 0.0, 0.30, lambda fit: fit - random.random()*0.1, 0.15, lambda fit: fit, 0.15, lambda fit: fit + random.random()*0.1 )

# Parameter: Non-coding Base Pairs
Junk_BP = 1400000

# Parameter: Corrected Mutation Rate
Initial_genes = 5000
Host_mutation_rate = 0.3

# Parameter: Total Non-autonomous TE
Total_NAut_TE = int(Initial_TEs * Carrying_capacity * 3)

# Parameter: Kidnapping Frequency
Kidnapping_Frequency = lambda live_aut, live_naut : 1 - 1/(1 + 0.01 * live_naut)



# This configuration file was programmatically generated.
# Used permutation '1000010111', which corresponds to 'HLLLLHLHHH'. Reference the configuration_mappings in the configuration file to determine what is 'high' and what is 'low'