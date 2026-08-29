"""Generated parameters for condition LLHLLLH-HH.

Code order: TE_progeny, TE_death_rate, Insertion_bias,
Corrected_mutation_rate, NC_BP, Mutation_effect, Carrying_capacity;
suffix: Kidnapping_frequency/Initial_NAut_TEs, or Z for none.
"""

from TEUtil import *

seed = None
saved = None

output = {
    "SPLAT": False,
    "SPLAT FITNESS": False,
    "INITIALIZATION": False,
    "GENERATION": True,
    "HOST EXTINCTION": True,
    "TE EXTINCTION": True,
    "TRIAL NO": True,
    "GENE INIT": False,
    "TE INIT": False,
    "BULK SIM": True,
    "CHECKPOINT": True,
}

Gene_length = 1000
TE_length = lambda autonomous: 6000 if autonomous else 300
Append_gene = True
Initial_Aut_TEs = 1
Initial_NAut_TEs = 3
TE_excision_rate = 0.0
Host_start_fitness = 1.0
Host_reproduction_rate = 1
Maximum_generations = 1500
Terminate_no_TEs = True
save_frequency = 50

TE_progeny = ProbabilityTable(0.15, 0, 0.55, 1, 0.30, 2)
TE_death_rate = 0.005
TE_Insertion_Distribution = Triangle(pmax=0, pzero=3.0/3.0)
Gene_Insertion_Distribution = Triangle(pzero=1.0/3.0, pmax=1)
Initial_genes = 500
Host_mutation_rate = 0.03
Junk_BP = 1400000
Host_mutation = ProbabilityTable(
    0.40, lambda fit: 0.0,
    0.30, lambda fit: fit - random.random() * 0.01,
    0.15, lambda fit: fit,
    0.15, lambda fit: fit + random.random() * 0.01,
)
Insertion_effect = ProbabilityTable(
    0.30, lambda fit: 0.0,
    0.20, lambda fit: fit - random.random() * 0.01,
    0.30, lambda fit: fit,
    0.20, lambda fit: fit + random.random() * 0.01,
)
Carrying_capacity = 300
Host_survival_rate = lambda propfit: min(Carrying_capacity * propfit, 0.95)
Kidnapping_frequency = lambda live_aut, live_naut: (
    1 - 1 / (1 + 0.07 * live_naut)
)
