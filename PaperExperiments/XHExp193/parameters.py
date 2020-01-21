
# parameters.py
"""
Exp 193 - {'Initial_genes': '500', 'Host_mutation_rate': '0.03', 'TE_progeny': '0.15, 0, 0.55, 1, 0.30, 2', 'TE_Insertion_Distribution': 'Triangle( pmax=0, pzero=3.0/3.0 )', 'Carrying_capacity': '300', 'TE_excision_rate': '0.1', 'Junk_BP': '1.4', 'Gene_Insertion_Distribution': 'Triangle( pzero=1.0/3.0, pmax=1 )', 'mutation_effect': '0.01', 'TE_death_rate': '0.0005'}
"""
from TEUtil import *;

# note that "#" indicates a comment

# set the following to True if you want messages printed to the screen
# while the program runs - search for these keywords in TESim.py to see
# what each one prints out

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

TE_Insertion_Distribution = Triangle( pmax=0, pzero=3.0/3.0 );
Gene_Insertion_Distribution = Triangle( pzero=1.0/3.0, pmax=1 );
# Triangle( pmax, pzero ) generates values between pmax and pzero with 
#   a triangular probability distribution, where pmax is the point of highest
#   probability, and pzero is the point of lowest probability
#   - you can change the orientation of the triangle by reversing the values
#   of pmax and pzero
# Flat() generates values between 0 and 1 with uniform probability

Gene_length = 1000;		        # use 1000?
TE_length = 1000;			# use 1000?

TE_death_rate = 0.0005;
TE_excision_rate = 0.1;	# set this to zero for retro transposons

# for retro transposons this is the probability of the given number of progeny
# for dna transposons this is the probability of the given number of progeny
#   ___PLUS___ the original re-inserting
TE_progeny = ProbabilityTable( 0.15, 0, 0.55, 1, 0.30, 2 );


						 
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

MILLION = 1000000;

Junk_BP = 1.4 * MILLION;

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
		
Carrying_capacity = 300;
Host_reproduction_rate = 1;  # how many offspring each host has

Host_survival_rate = lambda propfit: min( Carrying_capacity * propfit, 0.95 );
    # propfit = proportion of fitness owned by this individual

Maximum_generations = 1500;
Terminate_no_TEs = True;	# end simulation if there are no TEs left

# seed = 0;
seed = None;   # if seed = None, the random number generator's initial state is
               # set "randomly"

save_frequency = 50;    # Frequency with with which to save state of experiment

saved = None;   # if saved = None then we start a new simulation from scratch
                # if saves = string, then we open that file and resume a simulation
