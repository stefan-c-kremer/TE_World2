"""
Optimized parameter handling for simulation of TEs in the genome, focusing on ecological perspectives.
Includes original parameters (https://github.com/stefan-c-kremer/TE_World2) plus new performance optimization settings, and new parameter settings for ecological perspectives.
"""
import yaml
import os
import argparse
import sys
from typing import List
# Import specific functions to avoid circular imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from TEUtil_ABM2 import Triangle, Flat, ProbabilityTable, set_random_seed

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='TE Simulation with configurable parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python TESim_ABM2.py -c example_config.yaml
  python TESim_ABM2.py --config my_simulation_config.yaml
  python TESim_ABM2.py --help
        """
    )
    
    parser.add_argument(
        '-c', '--config',
        default='example_config.yaml',
        help='Configuration file path (default: example_config.yaml)'
    )
    
    # Parse arguments if this module is being run directly
    if __name__ == "__main__":
        return parser.parse_args()
    else:
        # If imported as a module, parse from sys.argv
        return parser.parse_args(sys.argv[1:])

def load_config(config_file="example_config.yaml"):
    """Load configuration from YAML file with fallback to defaults"""
    try:
        with open(config_file, 'r') as fh:
            config = yaml.safe_load(fh)
        print(f"Loaded configuration from {config_file}")
        return config
    except FileNotFoundError:
        print(f"Warning: Configuration file {config_file} not found, using defaults")
        return {}
    except yaml.YAMLError as e:
        print(f"Warning: Error parsing {config_file}: {e}, using defaults")
        return {}

# Parse command line arguments
args = parse_arguments()

# Load configuration using command line argument
config = load_config(args.config)

# Random seed will be set after all imports are complete
seed = None  # None for random seed, or integer for fixed (reproducible)seed

# Output control - fully config-driven from YAML file
# All logging control is now dictated by the config.yaml file
output_config = config.get('output', {})
logging_config = config.get('logging', {})

# Output control - determines what gets printed during simulation
# If not specified in config, defaults to minimal logging for performance
output = {
    "SPLAT": output_config.get('splat', False),   # detailed fitness logging
    "SPLAT FITNESS": output_config.get('splat_fitness', False),  # fitness value output
    "INITIALIZATION": output_config.get('initialization', False),  # initialization messages
    "GENERATION": output_config.get('generation', False),  # generation progress
    "HOST EXTINCTION": output_config.get('host_extinction', True),  # host extinction messages
    "TE EXTINCTION": output_config.get('te_extinction', True),  # TE extinction messages
    "TRIAL NO": output_config.get('trial_no', True),  # trial number
    "GENE INIT": output_config.get('gene_init', False),  # gene initialization messages
    "TE INIT": output_config.get('te_init', False),  # TE initialization messages
    "PERFORMANCE": output_config.get('performance', False),  # Performance monitoring output
}

# Logging level control from config
logging_level = logging_config.get('level', 'INFO').upper()
logging_to_file = logging_config.get('to_file', False)
logging_file_path = logging_config.get('file_path', './logs/simulation.log')

# Performance optimization settings - load from config with defaults
performance_config = config.get('performance', {})
performance = {
    "USE_PARALLEL": performance_config.get('use_parallel', True),
    "MAX_WORKERS": performance_config.get('max_workers', 8),
    "BATCH_SIZE": performance_config.get('batch_size', 50),
    "CACHE_STATISTICS": performance_config.get('cache_statistics', True),
    "USE_INTERVALTREE": performance_config.get('use_intervaltree', True),
    "USE_BIT_ARRAYS": performance_config.get('use_bit_arrays', True),
    "MEMORY_OPTIMIZE_FREQ": performance_config.get('memory_optimize_freq', 50),
    "TRACE_FREQUENCY": performance_config.get('trace_frequency', 1),
}

# Memory management settings - load from config with defaults
memory = {
    "ELEMENT_POOL_SIZE": performance_config.get('element_pool_size', 2000),
    "GENE_POOL_SIZE": performance_config.get('gene_pool_size', 500),
    "RANDOM_CACHE_SIZE": performance_config.get('random_cache_size', 10000),
    "STATS_CACHE_SIZE": performance_config.get('stats_cache_size', 100),
    "ENABLE_COPY_ON_WRITE": performance_config.get('enable_copy_on_write', True),
}

# Distribution parameters using optimized classes
TE_Insertion_Distribution = Triangle(pmax=0, pzero=2.0/3.0)
Gene_Insertion_Distribution = Triangle(pzero=1.0/3.0, pmax=1)

# Element size parameters - load from config with defaults
simulation_config = config.get('simulation', {})

# Gene subtype parameters - load from config with defaults
gene_config = config.get('genes', {})

# Gene subtype definitions
GENE_SUBTYPES = gene_config.get('subtypes', {
    'ORF': {
        'length': 1500,
        'frequency': 0.6,  # 60% of genes are ORFs
        'fitness_contribution': 1.0,
        'mutation_sensitivity': 0.8,
        'te_insertion_effect': 'high'  # high, medium, low
    },
    'promoter': {
        'length': 200,
        'frequency': 0.25,  # 25% of genes are promoters
        'fitness_contribution': 0.3,
        'mutation_sensitivity': 0.9,
        'te_insertion_effect': 'high'
    },
    'enhancer': {
        'length': 500,
        'frequency': 0.15,  # 15% of genes are enhancers
        'fitness_contribution': 0.5,
        'mutation_sensitivity': 0.7,
        'te_insertion_effect': 'medium'
    }
})

# Validate gene subtype frequencies sum to 1.0
subtype_frequencies = [subtype['frequency'] for subtype in GENE_SUBTYPES.values()]
if abs(sum(subtype_frequencies) - 1.0) > 1e-6:
    print(f"WARNING: Gene subtype frequencies sum to {sum(subtype_frequencies)}, not 1.0")
    # Normalize frequencies
    total_freq = sum(subtype_frequencies)
    for subtype in GENE_SUBTYPES.values():
        subtype['frequency'] /= total_freq

# Create probability table for gene subtype selection
gene_subtype_args = []
for i, (subtype_name, subtype_config) in enumerate(GENE_SUBTYPES.items()):
    gene_subtype_args.extend([subtype_config['frequency'], i])
GENE_SUBTYPE_SELECTION = ProbabilityTable(*gene_subtype_args)

# Function to get gene length based on subtype
def get_gene_length(subtype: str = None) -> int:
    """Get gene length for a specific subtype or average length if no subtype specified."""
    if subtype and subtype in GENE_SUBTYPES:
        return GENE_SUBTYPES[subtype]['length']
    elif GENE_SUBTYPES:
        # Return weighted average length
        total_length = sum(
            GENE_SUBTYPES[st]['length'] * GENE_SUBTYPES[st]['frequency'] 
            for st in GENE_SUBTYPES
        )
        return int(total_length)
    else:
        # Fallback to legacy single gene length
        return simulation_config.get('gene_length', 1000)

# Function to get gene subtype based on probability
def get_random_gene_subtype() -> str:
    """Get a random gene subtype based on configured frequencies."""
    if not GENE_SUBTYPES:
        return 'ORF'  # Default subtype
    
    subtype_names = list(GENE_SUBTYPES.keys())
    selected_idx = GENE_SUBTYPE_SELECTION.generate()
    return subtype_names[selected_idx]

# Function to get fitness contribution for a gene subtype
def get_gene_fitness_contribution(subtype: str) -> float:
    """Get fitness contribution multiplier for a gene subtype."""
    if subtype in GENE_SUBTYPES:
        return GENE_SUBTYPES[subtype]['fitness_contribution']
    return 1.0  # Default contribution

# Function to get mutation sensitivity for a gene subtype
def get_gene_mutation_sensitivity(subtype: str) -> float:
    """Get mutation sensitivity for a gene subtype."""
    if subtype in GENE_SUBTYPES:
        return GENE_SUBTYPES[subtype]['mutation_sensitivity']
    return 0.8  # Default sensitivity

# Function to get TE insertion effect level for a gene subtype
def get_gene_te_insertion_effect(subtype: str) -> str:
    """Get TE insertion effect level for a gene subtype."""
    if subtype in GENE_SUBTYPES:
        return GENE_SUBTYPES[subtype]['te_insertion_effect']
    return 'medium'  # Default effect level

# TE type configuration - load from config with defaults
te_config = config.get('tes', {})

# TE type definitions with autonomous/non-autonomous support
TE_TYPES = te_config.get('types', {
    'SINE': {
        'length': 300,
        'frequency': 0.4,  # 40% of TEs are SINEs
        'autonomous': False,  # SINEs are typically non-autonomous
        'parasitizes': ['LINE'],  # SINEs parasitize LINEs specifically
        'death_rate': 0.3,
        'excision_rate': 0.0,  # Retrotransposons don't excise
        'progeny_distribution': [0.15, 0, 0.55, 1, 0.30, 2]
    },
    'LINE': {
        'length': 6000,
        'frequency': 0.3,  # 30% of TEs are LINEs
        'autonomous': True,  # LINEs are typically autonomous
        'parasitizes': [],  # Autonomous TEs don't parasitize others
        'death_rate': 0.4,
        'excision_rate': 0.0,  # Retrotransposons don't excise
        'progeny_distribution': [0.20, 0, 0.50, 1, 0.30, 2]
    },
    'LTR': {
        'length': 8000,
        'frequency': 0.2,  # 20% of TEs are LTRs
        'autonomous': True,  # LTRs are typically autonomous
        'parasitizes': [],  # Autonomous TEs don't parasitize others
        'death_rate': 0.5,
        'excision_rate': 0.0,  # Retrotransposons don't excise
        'progeny_distribution': [0.25, 0, 0.45, 1, 0.30, 2]
    },
    'DNA_TRANSPOSON': {
        'length': 2000,
        'frequency': 0.1,  # 10% of TEs are DNA transposons
        'autonomous': True,  # DNA transposons are typically autonomous
        'parasitizes': [],  # Autonomous TEs don't parasitize others
        'death_rate': 0.6,
        'excision_rate': 0.1,  # DNA transposons can excise
        'progeny_distribution': [0.30, 0, 0.40, 1, 0.30, 2]
    }
})

# Validate TE type frequencies sum to 1.0
te_frequencies = [te_type['frequency'] for te_type in TE_TYPES.values()]
if abs(sum(te_frequencies) - 1.0) > 1e-6:
    print(f"WARNING: TE type frequencies sum to {sum(te_frequencies)}, not 1.0")
    # Normalize frequencies
    total_freq = sum(te_frequencies)
    for te_type in TE_TYPES.values():
        te_type['frequency'] /= total_freq

# Create probability table for TE type selection
te_type_args = []
for i, (te_type_name, te_type_config) in enumerate(TE_TYPES.items()):
    te_type_args.extend([te_type_config['frequency'], i])
TE_TYPE_SELECTION = ProbabilityTable(*te_type_args)

# Function to get TE length based on type
def get_te_length(te_type: str = None) -> int:
    """Get TE length for a specific type or average length if no type specified."""
    if te_type and te_type in TE_TYPES:
        return TE_TYPES[te_type]['length']
    elif TE_TYPES:
        # Return weighted average length
        total_length = sum(
            TE_TYPES[tt]['length'] * TE_TYPES[tt]['frequency'] 
            for tt in TE_TYPES
        )
        return int(total_length)
    else:
        # Fallback to legacy single TE length
        return simulation_config.get('te_length', 1000)

# Function to get TE type based on probability
def get_random_te_type() -> str:
    """Get a random TE type based on configured frequencies."""
    if not TE_TYPES:
        return 'SINE'  # Default type
    
    te_type_names = list(TE_TYPES.keys())
    selected_idx = TE_TYPE_SELECTION.generate()
    return te_type_names[selected_idx]

# Function to get autonomous status for a TE type
def get_te_autonomous_status(te_type: str = None) -> bool:
    """Get autonomous status for a TE type or random type if none specified."""
    if te_type is None:
        te_type = get_random_te_type()
    
    if te_type in TE_TYPES:
        return TE_TYPES[te_type]['autonomous']
    return False  # Default to non-autonomous

# Function to get TE death rate for a specific type
def get_te_death_rate(te_type: str = None) -> float:
    """Get death rate for a TE type."""
    if te_type and te_type in TE_TYPES:
        return TE_TYPES[te_type]['death_rate']
    return TE_death_rate  # Default death rate

# Function to get TE excision rate for a specific type
def get_te_excision_rate(te_type: str = None) -> float:
    """Get excision rate for a TE type."""
    if te_type and te_type in TE_TYPES:
        return TE_TYPES[te_type]['excision_rate']
    return TE_excision_rate  # Default excision rate

# Function to get progeny distribution for a specific TE type
def get_te_progeny_distribution(te_type: str = None) -> ProbabilityTable:
    """Get progeny distribution for a TE type."""
    if te_type and te_type in TE_TYPES:
        dist_values = TE_TYPES[te_type]['progeny_distribution']
        return ProbabilityTable(*dist_values)
    return TE_progeny  # Default progeny distribution

# Function to get parasitism information for a TE type
def get_te_parasitism_targets(te_type: str = None) -> List[str]:
    """Get the list of TE types that this TE type parasitizes."""
    if te_type and te_type in TE_TYPES:
        return TE_TYPES[te_type].get('parasitizes', [])
    return []  # Default to no parasitism

# Store individual TE lengths for reference
def get_te_lengths():
    """Get a dictionary of TE lengths for each TE type present in the configuration"""
    te_lengths = {}
    
    if TE_TYPES:
        for te_type, config in TE_TYPES.items():
            te_lengths[te_type] = config['length']
    
    return te_lengths

def get_active_te_types():
    """Get a list of TE types that are defined in the configuration"""
    if TE_TYPES:
        return list(TE_TYPES.keys())
    return []

# Legacy gene length for backward compatibility
Gene_length = get_gene_length()

# TE length determination using new TE type system
TE_length = get_te_length()

# Store TE-specific information
TE_LENGTHS = get_te_lengths()
ACTIVE_TE_TYPES = get_active_te_types()

# Print TE configuration summary
if ACTIVE_TE_TYPES:
    print(f"Active TE types: {ACTIVE_TE_TYPES}")
    print(f"TE lengths: {TE_LENGTHS}")
    print(f"Autonomous TEs: {[te_type for te_type, config in TE_TYPES.items() if config['autonomous']]}")
    print(f"Non-autonomous TEs: {[te_type for te_type, config in TE_TYPES.items() if not config['autonomous']]}")
else:
    print("No specific TE types defined - using generic TE simulation")

# TE behavior parameters - load from config with defaults
TE_death_rate = simulation_config.get('te_death_rate', 0.5)
TE_excision_rate = simulation_config.get('te_excision_rate', 0.1)

# TE progeny distribution
# For retrotransposons: probability of given number of progeny
# For DNA transposons: probability of given number of progeny PLUS original re-inserting
TE_progeny = ProbabilityTable(0.15, 0, 0.55, 1, 0.30, 2)

# Population parameters - load from config with defaults
Initial_genes = simulation_config.get('initial_genes', 500)
Initial_TEs = simulation_config.get('initial_tes', 1)

# Gene placement behavior - load from config with defaults
Append_gene = simulation_config.get('append_gene', True)

# Genome size - load from config with defaults
MILLION = 1000000
Junk_BP = simulation_config.get('junk_bp', 14 * MILLION)

# Host fitness and mutation parameters - load from config with defaults
Host_start_fitness = simulation_config.get('host_start_fitness', 1.0)
Host_mutation_rate = simulation_config.get('host_mutation_rate', 0.03)

# Host mutation effects (using optimized lambda functions)
# Relative fitness changes based on current fitness
# Note: vrng will be initialized later, so we use a placeholder for now
Host_mutation = None  # Will be initialized after vrng is available

# Fitness effects of TE insertions into genes
# Note: vrng will be initialized later, so we use a placeholder for now
Insertion_effect = None  # Will be initialized after vrng is available

# Population genetics parameters - load from config with defaults
Carrying_capacity = simulation_config.get('carrying_capacity', 300)
Host_reproduction_rate = simulation_config.get('host_reproduction_rate', 1)

# Survival rate function (vectorized for performance)
def Host_survival_rate(propfit):
    """
    Optimized survival rate calculation.
    propfit = proportion of fitness owned by this individual
    """
    return np.minimum(Carrying_capacity * propfit, 0.95)

# Simulation control parameters - load from config with defaults
Maximum_generations = simulation_config.get('maximum_generations', 50)
Terminate_no_TEs = simulation_config.get('terminate_no_tes', True)

# Random seed control
seed = None  # None for random seed, integer for reproducible results

# Save/load parameters - load from config with defaults
save_frequency = simulation_config.get('save_frequency', 50)
saved = None          # Filename to load saved state (None to start fresh)

# New optimization parameters - load from config with defaults
trace_frequency = simulation_config.get('trace_frequency', 1)

# Benchmarking and profiling - load from config with defaults
enable_profiling = performance_config.get('debug_performance', False)
profile_output = "profile.txt"  # Output file for profiling results

# Advanced memory optimization - load from config with defaults
enable_compression = performance_config.get('enable_compression', False)
compression_level = performance_config.get('compression_level', 6)

# Parallel processing configuration - load from config with defaults
parallel_threshold = performance_config.get('parallel_threshold', 100)
chunk_size_factor = performance_config.get('chunk_size_factor', 0.1)

# Validation and debugging - load from config with defaults
validate_data_structures = performance_config.get('validate_data_structures', False)
debug_memory_usage = performance_config.get('debug_memory_usage', False)
debug_performance = performance_config.get('debug_performance', False)

# Export settings for external analysis
export_detailed_stats = False   # Export detailed statistics for analysis
export_format = "csv"          # Format for exported data (csv, json, hdf5)
export_frequency = 10          # Export detailed data every N generations

# Function to validate parameters
def validate_parameters():
    """Validate parameter settings and print warnings for potential issues."""
    
    if Initial_genes > Junk_BP // Gene_length:
        print("WARNING: Too many initial genes for genome size")
    
    if TE_death_rate > 1.0 or TE_death_rate < 0.0:
        raise ValueError("TE_death_rate must be between 0 and 1")
    
    if TE_excision_rate > 1.0 or TE_excision_rate < 0.0:
        raise ValueError("TE_excision_rate must be between 0 and 1")
    
    if Host_mutation_rate > 1.0 or Host_mutation_rate < 0.0:
        raise ValueError("Host_mutation_rate must be between 0 and 1")
    
    if Carrying_capacity <= 0:
        raise ValueError("Carrying_capacity must be positive")
    
    if performance["MAX_WORKERS"] > 16:
        print("WARNING: More than 16 workers may cause memory issues")
    
    if performance["TRACE_FREQUENCY"] > 10:
        print("WARNING: Infrequent tracing may miss important dynamics")
    
    # Memory usage estimation
    estimated_memory_mb = (
        Carrying_capacity * Initial_genes * 0.001 +  # Genes
        Carrying_capacity * Initial_TEs * 10 * 0.001 +  # TEs (assuming growth)
        Junk_BP * 0.000001  # Genome representation
    )
    
    if estimated_memory_mb > 1000:
        print(f"WARNING: Estimated memory usage: {estimated_memory_mb:.1f} MB")

# Function to optimize parameters for large simulations
def optimize_for_large_simulation():
    """Optimize parameters for large-scale simulations."""
    global performance, memory
    
    performance["USE_PARALLEL"] = True
    performance["BATCH_SIZE"] = 100
    performance["TRACE_FREQUENCY"] = 5  # Reduce tracing frequency
    performance["MEMORY_OPTIMIZE_FREQ"] = 25  # More frequent optimization
    
    memory["ELEMENT_POOL_SIZE"] = 5000
    memory["GENE_POOL_SIZE"] = 1000
    memory["RANDOM_CACHE_SIZE"] = 50000
    memory["ENABLE_COPY_ON_WRITE"] = True
    
    print("Parameters optimized for large-scale simulation")

# Function to optimize parameters for detailed analysis
def optimize_for_detailed_analysis():
    """Optimize parameters for detailed data collection and analysis."""
    global performance, export_detailed_stats, export_frequency
    
    performance["TRACE_FREQUENCY"] = 1  # Collect every generation
    export_detailed_stats = True
    export_frequency = 1
    
    print("Parameters optimized for detailed analysis")

# Function to get current memory usage estimate
def estimate_memory_usage():
    """Estimate memory usage based on current parameters."""
    
    # Element storage
    elements_mb = (Initial_genes + Initial_TEs * 10) * Carrying_capacity * 0.001
    
    # Genome representation
    if performance["USE_BIT_ARRAYS"] and Junk_BP > 1000000:
        genome_mb = Junk_BP * Carrying_capacity * 0.000001 / 8  # Bit array
    else:
        genome_mb = Junk_BP * Carrying_capacity * 0.000001  # Full representation
    
    # Object pools and caches
    cache_mb = (memory["ELEMENT_POOL_SIZE"] + memory["GENE_POOL_SIZE"]) * 0.001
    cache_mb += memory["RANDOM_CACHE_SIZE"] * 0.00001  # Random number cache
    
    total_mb = elements_mb + genome_mb + cache_mb
    
    print(f"Estimated memory usage:")
    print(f"  Elements: {elements_mb:.1f} MB")
    print(f"  Genomes: {genome_mb:.1f} MB") 
    print(f"  Caches: {cache_mb:.1f} MB")
    print(f"  Total: {total_mb:.1f} MB")
    
    return total_mb

# Validate parameters on import
if __name__ != "__main__":
    validate_parameters()

# Benchmark mode for performance testing
def enable_benchmark_mode():
    """Enable settings optimized for benchmarking performance."""
    global output, performance, Maximum_generations
    
    # Disable most output
    for key in output:
        output[key] = False
    
    # Optimize for speed
    performance["TRACE_FREQUENCY"] = 10
    performance["USE_PARALLEL"] = True
    performance["BATCH_SIZE"] = 100
    
    # Shorter simulation for benchmarking
    Maximum_generations = 20
    
    print("Benchmark mode enabled - optimized for performance testing")

# Development mode with extra validation
def enable_development_mode():
    """Enable settings for development and debugging."""
    global validate_data_structures, debug_memory_usage, debug_performance
    
    validate_data_structures = True
    debug_memory_usage = True
    debug_performance = True
    
    print("Development mode enabled - extra validation and debugging active")

# Quick setup functions for common simulation types
def setup_small_test():
    """Setup for small test simulations."""
    global Carrying_capacity, Maximum_generations, Initial_genes, Initial_TEs
    
    Carrying_capacity = 50
    Maximum_generations = 20
    Initial_genes = 100
    Initial_TEs = 1
    
    print("Small test simulation setup complete")

def setup_large_population():
    """Setup for large population simulations."""
    global Carrying_capacity, performance
    
    Carrying_capacity = 1000
    optimize_for_large_simulation()
    
    print("Large population simulation setup complete")

def setup_long_evolution():
    """Setup for long evolutionary simulations."""
    global Maximum_generations, save_frequency, performance
    
    Maximum_generations = 500
    save_frequency = 25
    performance["TRACE_FREQUENCY"] = 5
    
    print("Long evolution simulation setup complete")

# Print configuration summary
def print_config_summary():
    """Print a summary of current configuration."""
    print("\n=== TE Simulation Configuration ===")
    print(f"Population size: {Carrying_capacity}")
    print(f"Generations: {Maximum_generations}")
    print(f"Initial genes: {Initial_genes}")
    print(f"Initial TEs: {Initial_TEs}")
    print(f"Genome size: {Junk_BP/MILLION:.1f} Mbp")
    print(f"Parallel processing: {performance['USE_PARALLEL']}")
    print(f"Max workers: {performance['MAX_WORKERS']}")
    print(f"Batch size: {performance['BATCH_SIZE']}")
    print(f"Trace frequency: {performance['TRACE_FREQUENCY']}")
    print("===================================\n")

# Auto-optimize based on genome size and population
def auto_optimize():
    """Automatically optimize parameters based on simulation size."""
    total_elements = Carrying_capacity * (Initial_genes + Initial_TEs * 10)
    
    if total_elements > 100000:  # Large simulation
        optimize_for_large_simulation()
    elif Junk_BP > 50 * MILLION:  # Large genome
        performance["USE_BIT_ARRAYS"] = True
        performance["BATCH_SIZE"] = 100
    elif Carrying_capacity < 100:  # Small population
        performance["USE_PARALLEL"] = False
    
    print("Auto-optimization complete")

# Call auto-optimize by default
if __name__ != "__main__":
    auto_optimize()

# Function to initialize probability tables that depend on vrng
def initialize_probability_tables():
    """Initialize probability tables that depend on the global vrng."""
    global Host_mutation, Insertion_effect
    
    # Import vrng here to avoid circular imports
    from TEUtil_ABM2 import vrng
    
    # Host mutation effects (simplified for now)
    # Use simple multipliers instead of lambda functions
    Host_mutation = ProbabilityTable(
        0.40, 1.0,    # No change (40%)
        0.30, 0.9,    # Deleterious (30%) - 10% reduction
        0.15, 1.0,    # Neutral (15%)
        0.15, 1.1     # Beneficial (15%) - 10% increase
    )

    # Fitness effects of TE insertions into genes (simplified)
    Insertion_effect = ProbabilityTable(
        0.30, 0.0,    # Lethal (30%)
        0.20, 0.9,    # Deleterious (20%) - 10% reduction
        0.30, 1.0,    # Neutral (30%)
        0.20, 1.1     # Beneficial (20%) - 10% increase
    )
    
    print("Probability tables initialized successfully")

# Initialize probability tables after vrng is available
if __name__ != "__main__":
    # Small delay to ensure vrng is initialized
    import time
    time.sleep(0.1)
    initialize_probability_tables()
    
    # Set random seed after all initialization is complete
    set_random_seed(seed) 