
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

# ********************************************
# TRIAL FIELDS
# Parameter: Corrected Mutation Rate
Initial_genes = 5000
Host_mutation_rate = 0.3

# Parameter: Non-coding Base Pairs
Junk_BP = 14000000

# Parameter: Mutation Effect
Host_mutation = ProbabilityTable( 0.40, lambda fit: 0.0, 0.30, lambda fit: fit - random.random()*0.1, 0.15, lambda fit: fit, 0.15, lambda fit: fit + random.random()*0.1 )
Insertion_effect = ProbabilityTable(0.30, lambda fit: 0.0, 0.20, lambda fit: fit - random.random()*0.1, 0.30, lambda fit: fit, 0.20, lambda fit: fit + random.random()*0.1 )

# Parameter: Carrying Capacity
Carrying_capacity = 300

# Parameter: TE Progeny
TE_progeny = ProbabilityTable( 0.15, 0, 0.55, 1, 0.30, 2 )

# Parameter: TE Death Rate
TE_death_rate = 0.005

# Parameter: Insertion Bias
TE_Insertion_Distribution = Triangle( pmax=0, pzero=3.0/3.0 )
Gene_Insertion_Distribution = Triangle( pzero=1.0/3.0, pmax=1 )

# Parameter: Kidnapping Frequency
Kidnapping_frequency = lambda live_aut, live_naut : 1 - 1/(1 + 0.02 * live_naut)

# Parameter: Initial non-autonomous TEs
Initial_NAut_TEs = 0

saved = None

# This configuration file was programmatically generated.
# Used permutation '111100102', which corresponds to 'HHHHLLHLZ'. Reference the configuration_mappings in the configuration file to determine what is 'high' and what is 'low'