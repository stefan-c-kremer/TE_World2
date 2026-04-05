
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
TE_length = lambda autonomous : 6000 if autonomous else 300;
						 
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

Host_start_fitness = 1.0;
Host_mutation_rate = 0.03;

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

# ********************************************
# TRIAL FIELDS


Insertion_bias = TE_Insertion_Distribution = Triangle( pmax=0, pzero=3.0/3.0 ); Gene_Insertion_Distribution = Triangle( pzero=1.0/3.0, pmax=1 );

TE_death_rate = lambda autonomous : 0.00008 if autonomous else 0.002

TE_excision_rate = 0.5

TE_progeny = ProbabilityTable( 0.00, 0, 0.55, 1, 0.30, 2, 0.15, 3 )

Carrying_capacity = 300

Mutation_effect = 0.1

Junk_BP = 14000000

Corrected_mutation_rate = 0.3

Total_NAut_TE = int(Initial_TEs * Carrying_capacity)

Kidnapping_Frequency = lambda live_aut, live_naut : 1 - 1/(1 + 0.01 * live_naut)

# This configuration file was programmatically generated.
# Used permutation '1011111101', which corresponds to 'HLHHHHHHLH'. Reference the mappings in the configuration file to determine what is 'high' and what is 'low'