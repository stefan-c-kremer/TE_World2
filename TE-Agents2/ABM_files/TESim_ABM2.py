"""
Optimized TE simulation with high-performance data structures and algorithms.
Implements IntervalTree lookups, vectorized operations, parallel processing, 
and memory-efficient representations for large-scale genome simulations.
"""

import os
import sys
import math
import time
import gzip
import bisect
import subprocess
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import cpu_count, shared_memory
from typing import List, Dict, Tuple, Optional, Union, Any
import warnings

import numpy as np
from numba import jit, njit
from intervaltree import IntervalTree, Interval
import psutil

# Import optimized utilities
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from TEUtil_ABM2 import (
    VectorizedRandom, Triangle, Flat, ProbabilityTable, ObjectPool, BitArray,
    triangle_pool, flat_pool, probability_table_pool, 
    batch_random_bool, optimize_memory_usage, set_random_seed
)

# Initialize vrng after parameters are loaded
from TEUtil_ABM2 import vrng

# Ensure vrng is properly initialized
if vrng is None:
    set_random_seed(None)

# Import parameters after command line arguments are parsed
# This ensures the config file is loaded correctly
import parameters_ABM2 as parameters

################################################################################
# Config-driven logging system
################################################################################

import logging
import os

# Initialize logging system based on config
def setup_logging():
    """Setup logging system based on configuration."""
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(parameters.logging_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure logging
    log_level = getattr(logging, parameters.logging_level, logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if enabled)
    if parameters.logging_to_file:
        file_handler = logging.FileHandler(parameters.logging_file_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Initialize logging
logger = setup_logging()

# Debug function to show logging configuration
def debug_logging_config():
    """Debug function to show current logging configuration."""
    print(f"Logging configuration:")
    print(f"  Level: {parameters.logging_level}")
    print(f"  To file: {parameters.logging_to_file}")
    print(f"  File path: {parameters.logging_file_path}")
    print(f"  Output settings:")
    for key, value in parameters.output.items():
        print(f"    {key}: {value}")
    
    # Check if logs directory exists
    log_dir = os.path.dirname(parameters.logging_file_path)
    if log_dir:
        print(f"  Log directory exists: {os.path.exists(log_dir)}")
        if not os.path.exists(log_dir):
            print(f"  Creating log directory: {log_dir}")
            os.makedirs(log_dir, exist_ok=True)
    
    # Test file writing
    if parameters.logging_to_file:
        try:
            with open(parameters.logging_file_path, 'a') as f:
                f.write(f"# Log file test - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            print(f"  Log file writable: True")
        except Exception as e:
            print(f"  Log file writable: False - {e}")

# Debug logging configuration on startup
debug_logging_config()

def output(keyword: str, message: str):
    """
    Config-driven output function that respects logging settings.
    Supports both console and file output based on configuration.
    """
    # Check if this output type is enabled in config
    if (hasattr(parameters, 'output') and 
        keyword in parameters.output and 
        parameters.output[keyword]):
        
        # Map output keywords to log levels
        log_level = logging.INFO  # Default level
        
        # Critical events
        if keyword in ['HOST EXTINCTION', 'TE EXTINCTION']:
            log_level = logging.WARNING
        
        # Detailed debugging
        elif keyword in ['SPLAT', 'SPLAT FITNESS', 'GENE INIT', 'TE INIT']:
            log_level = logging.DEBUG
        
        # Performance monitoring
        elif keyword in ['PERFORMANCE']:
            log_level = logging.INFO
        
        # General information
        elif keyword in ['INITIALIZATION', 'GENERATION', 'TRIAL NO']:
            log_level = logging.INFO
        
        # Log the message with appropriate level
        logger.log(log_level, f"[{keyword}]: {message}")
    
    # Also log to console for critical events regardless of config
    elif keyword in ['HOST EXTINCTION', 'TE EXTINCTION']:
        logger.warning(f"[{keyword}]: {message}")

################################################################################
# Performance monitoring and memory management
################################################################################

class PerformanceMonitor:
    """Monitor and optimize performance metrics during simulation."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()
        self.generation_times = []
        self.memory_usage = []
        self.cpu_usage = []
    
    def log_generation_stats(self, generation: int):
        """Log performance statistics for a generation."""
        current_time = time.time()
        generation_time = current_time - (self.start_time + sum(self.generation_times))
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        cpu_percent = self.process.cpu_percent()
        
        self.generation_times.append(generation_time)
        self.memory_usage.append(memory_mb)
        self.cpu_usage.append(cpu_percent)
        
        if generation % 10 == 0:  # Log every 10 generations
            avg_time = np.mean(self.generation_times[-10:])
            avg_memory = np.mean(self.memory_usage[-10:])
            # Use config-driven logging instead of direct print
            output("PERFORMANCE", f"Gen {generation}: {avg_time:.3f}s/gen, {avg_memory:.1f}MB, {cpu_percent:.1f}% CPU")

# Global performance monitor
perf_monitor = PerformanceMonitor()

################################################################################
# Constants and optimized data structures
################################################################################

# Use constants instead of string comparisons for better performance
JUNK_TYPE = 0
GENE_TYPE = 1
TE_TYPE = 2

# Result codes for collision detection (replaces exceptions, significantly faster)
class InsertResult:
    SUCCESS = 0
    COLLISION_TE = 1
    COLLISION_GENE = 2
    OUT_OF_BOUNDS = 3

# Pre-compiled numba functions for hot paths
@njit
def calculate_fitness_effects(fitness_values, effects):
    """Vectorized fitness effect calculation."""
    return np.maximum(0.0, fitness_values + effects)

@njit
def batch_survival_probability(fitnesses, total_fitness, carrying_capacity):
    """Calculate survival probabilities for a batch of individuals."""
    if total_fitness <= 0:
        return np.zeros_like(fitnesses)
    proportional_fitness = fitnesses / total_fitness
    return np.minimum(carrying_capacity * proportional_fitness, 0.95)

@njit
def fast_element_overlap_check(start1, end1, start2, end2):
    """Fast overlap check between two genomic elements."""
    return not (end1 <= start2 or end2 <= start1)

################################################################################
# Memory-efficient element representation
################################################################################

class ElementData:
    """Memory-efficient element storage using numpy arrays and bit packing."""
    
    def __init__(self, initial_capacity=1000):
        self.capacity = initial_capacity
        self.size = 0
        
        # Packed arrays for element data
        self.starts = np.zeros(initial_capacity, dtype=np.int64)
        self.ends = np.zeros(initial_capacity, dtype=np.int64)
        self.types = np.zeros(initial_capacity, dtype=np.uint8)  # 0=junk, 1=gene, 2=TE
        self.dead_flags = np.zeros(initial_capacity, dtype=bool)  # For TEs
        self.lengths = np.zeros(initial_capacity, dtype=np.int32)
        
        # Gene subtype storage (for genes only)
        self.gene_subtypes = np.full(initial_capacity, '', dtype=object)  # String array for subtypes
        
        # TE autonomous flag storage (for TEs only)
        self.te_autonomous = np.zeros(initial_capacity, dtype=bool)  # For TEs
        
        # Free list for recycling indices
        self.free_indices = []
    
    def allocate_element(self) -> int:
        """Allocate a new element index, reusing freed indices when possible."""
        if self.free_indices:
            return self.free_indices.pop()
        
        if self.size >= self.capacity:
            self._resize()
        
        idx = self.size
        self.size += 1
        return idx
    
    def free_element(self, idx: int):
        """Mark an element index as free for reuse."""
        self.free_indices.append(idx)
    
    def _resize(self):
        """Resize arrays when capacity is exceeded."""
        new_capacity = self.capacity * 2
        
        self.starts = np.resize(self.starts, new_capacity)
        self.ends = np.resize(self.ends, new_capacity)
        self.types = np.resize(self.types, new_capacity)
        self.dead_flags = np.resize(self.dead_flags, new_capacity)
        self.lengths = np.resize(self.lengths, new_capacity)
        
        # Resize gene subtypes array - handle object arrays properly
        new_gene_subtypes = np.full(new_capacity, '', dtype=object)
        new_gene_subtypes[:len(self.gene_subtypes)] = self.gene_subtypes
        self.gene_subtypes = new_gene_subtypes
        
        # Resize TE autonomous array
        self.te_autonomous = np.resize(self.te_autonomous, new_capacity)
        
        self.capacity = new_capacity

# Global element storage
element_data = ElementData()

################################################################################
# Optimized Element classes with object pooling
################################################################################

class Element:
    """
    Lightweight element with minimal memory footprint.
    Uses global element_data for storage and implements copy-on-write.
    """
    
    def __init__(self, length: int, start: int, element_type: int, subtype: str = ''):
        self.idx = element_data.allocate_element()
        self._chromosome = None
        
        # Store data in global arrays
        element_data.starts[self.idx] = start
        element_data.ends[self.idx] = start + length
        element_data.types[self.idx] = element_type
        element_data.lengths[self.idx] = length
        element_data.dead_flags[self.idx] = False
        element_data.gene_subtypes[self.idx] = subtype
        
        # Debug: Print element type assignment (commented out to reduce noise)
        # print(f"DEBUG: Created element at idx {self.idx}, type {element_type} ({'GENE' if element_type == GENE_TYPE else 'TE' if element_type == TE_TYPE else 'JUNK'}), class {type(self).__name__}")
    
    @property
    def start(self) -> int:
        return element_data.starts[self.idx]
    
    @start.setter
    def start(self, value: int):
        # Ensure start position is non-negative
        value = max(0, value)
        element_data.starts[self.idx] = value
        
        # Ensure length is valid
        length = element_data.lengths[self.idx]
        if length <= 0:
            length = 1
            element_data.lengths[self.idx] = length
        
        element_data.ends[self.idx] = value + length
    
    @property
    def end(self) -> int:
        return element_data.ends[self.idx]
    
    @property
    def length(self) -> int:
        return element_data.lengths[self.idx]
    
    @property
    def subtype(self) -> str:
        return element_data.gene_subtypes[self.idx]
    
    @property
    def chromosome(self):
        return self._chromosome
    
    @chromosome.setter 
    def chromosome(self, value):
        self._chromosome = value
    
    def __lt__(self, other):
        return self.start < other.start
    
    def __le__(self, other):
        return self.start <= other.start
    
    def __eq__(self, other):
        return self.start == other.start
    
    def __ne__(self, other):
        if isinstance(other, Element):
            return self.start != other.start
        return True
    
    def __gt__(self, other):
        return self.start > other.start
    
    def __ge__(self, other):
        return self.start >= other.start
    
    def reset(self, *args, **kwargs):
        """Reset element data for object pool reuse."""
        # This is a base method - subclasses should override
        pass
    
    def __del__(self):
        """Clean up element data when object is destroyed."""
        if hasattr(self, 'idx'):
            try:
                # Access element_data as global variable
                global element_data
                if element_data is not None:
                    element_data.free_element(self.idx)
            except (AttributeError, TypeError, NameError):
                # Ignore cleanup errors during shutdown
                pass

class EukGene(Element):
    """Memory-efficient eukaryotic gene with introns, exons, and regulatory regions."""
    
    def __init__(self, start: int, length: int = None, subtype: str = None):
        if length is None:
            if subtype and subtype in parameters.GENE_SUBTYPES:
                length = parameters.GENE_SUBTYPES[subtype]['length']
            else:
                # Get random subtype if none specified
                subtype = parameters.get_random_gene_subtype()
                length = parameters.GENE_SUBTYPES[subtype]['length']
        elif subtype is None:
            # If length is specified but no subtype, determine subtype from length
            subtype = self._determine_subtype_from_length(length)
        
        super().__init__(length, start, GENE_TYPE, subtype)
        
        # Initialize eukaryotic-specific properties
        self._initialize_eukaryotic_structure()
    
    def _determine_subtype_from_length(self, length: int) -> str:
        """Determine gene subtype based on length (fallback method)."""
        for subtype, config in parameters.GENE_SUBTYPES.items():
            if abs(length - config['length']) < 100:  # Within 100bp tolerance
                return subtype
        return 'ORF'  # Default fallback
    
    def _initialize_eukaryotic_structure(self):
        """Initialize eukaryotic gene structure based on subtype."""
        if self.subtype == 'ORF':
            # Protein-coding gene with introns and exons
            self.exon_count = max(1, int(self.length / 1000))  # Rough estimate
            self.intron_count = max(0, self.exon_count - 1)
            self.has_polyA_site = True
            self.has_splice_sites = self.intron_count > 0
            self.regulatory_complexity = 'high'
            self.tf_binding_sites = max(1, int(self.length / 200))  # TF sites every ~200bp
            
        elif self.subtype == 'promoter':
            # Promoter region with transcription factor binding sites
            self.exon_count = 0
            self.intron_count = 0
            self.has_polyA_site = False
            self.has_splice_sites = False
            self.regulatory_complexity = 'very_high'
            self.tf_binding_sites = max(1, int(self.length / 50))  # TF sites every ~50bp
            
        elif self.subtype == 'enhancer':
            # Enhancer region with multiple regulatory elements
            self.exon_count = 0
            self.intron_count = 0
            self.has_polyA_site = False
            self.has_splice_sites = False
            self.regulatory_complexity = 'medium'
            self.tf_binding_sites = max(2, int(self.length / 100))  # TF sites every ~100bp
            
        elif self.subtype == 'intron':
            # Intronic region - spliced out during transcription
            self.exon_count = 0
            self.intron_count = 1
            self.has_polyA_site = False
            self.has_splice_sites = True
            self.regulatory_complexity = 'low'
            self.tf_binding_sites = max(1, int(self.length / 300))  # Fewer TF sites in introns
            
        elif self.subtype == 'silencer':
            # Silencer region - represses transcription
            self.exon_count = 0
            self.intron_count = 0
            self.has_polyA_site = False
            self.has_splice_sites = False
            self.regulatory_complexity = 'high'
            self.tf_binding_sites = max(1, int(self.length / 80))  # TF sites every ~80bp
            
        elif self.subtype == 'insulator':
            # Insulator region - blocks enhancer-promoter interactions
            self.exon_count = 0
            self.intron_count = 0
            self.has_polyA_site = False
            self.has_splice_sites = False
            self.regulatory_complexity = 'medium'
            self.tf_binding_sites = max(1, int(self.length / 150))  # TF sites every ~150bp
            
        else:
            # Default for unknown subtypes
            self.exon_count = 1
            self.intron_count = 0
            self.has_polyA_site = False
            self.has_splice_sites = False
            self.regulatory_complexity = 'low'
            self.tf_binding_sites = 0
    
    @property
    def is_protein_coding(self) -> bool:
        """Check if this gene codes for protein."""
        return self.subtype == 'ORF'
    
    @property
    def is_regulatory(self) -> bool:
        """Check if this is a regulatory element."""
        return self.subtype in ['promoter', 'enhancer']
    
    @property
    def splicing_complexity(self) -> str:
        """Get splicing complexity level."""
        if self.intron_count == 0:
            return 'none'
        elif self.intron_count <= 2:
            return 'simple'
        elif self.intron_count <= 5:
            return 'moderate'
        else:
            return 'complex'
    
    def get_te_insertion_impact(self, te_type: str) -> float:
        """Calculate TE insertion impact based on eukaryotic gene structure."""
        te_effect_level = parameters.get_gene_te_insertion_effect(self.subtype)
        base_effect = 1.0 if te_effect_level == 'medium' else (2.0 if te_effect_level == 'high' else 0.5)
        
        # Adjust based on eukaryotic structure and gene subtype
        if self.subtype == 'ORF':
            if self.has_splice_sites:
                # Insertion in intron of protein-coding gene - often less harmful
                if te_type in ['SINE', 'LINE']:  # Often insert in introns
                    return base_effect * 0.6  # 40% reduction in impact
                else:
                    return base_effect * 1.1  # 10% increase for other TEs
            else:
                # Single exon gene - any insertion is harmful
                return base_effect * 1.5
                
        elif self.subtype == 'promoter':
            # Promoter regions are very sensitive to TE insertions
            return base_effect * 2.0  # Double impact in promoters
                
        elif self.subtype == 'enhancer':
            # Enhancer regions are moderately sensitive
            return base_effect * 1.3
            
        elif self.subtype == 'intron':
            # Intronic regions are least sensitive - TEs often insert here
            if te_type in ['SINE', 'LINE']:
                return base_effect * 0.3  # 70% reduction - very low impact
            else:
                return base_effect * 0.8  # 20% reduction for other TEs
                
        elif self.subtype == 'silencer':
            # Silencer regions are sensitive - can disrupt repression
            return base_effect * 1.8
            
        elif self.subtype == 'insulator':
            # Insulator regions are moderately sensitive
            return base_effect * 1.2
                
        return base_effect
    
    def reset(self, start: int, length: int, subtype: str):
        """Reset gene data for object pool reuse."""
        # Reuse existing index or allocate new one
        if not hasattr(self, 'idx'):
            self.idx = element_data.allocate_element()
        
        # Update data in global arrays
        element_data.starts[self.idx] = start
        element_data.ends[self.idx] = start + length
        element_data.types[self.idx] = GENE_TYPE
        element_data.lengths[self.idx] = length
        element_data.dead_flags[self.idx] = False
        element_data.gene_subtypes[self.idx] = subtype
        element_data.te_autonomous[self.idx] = False
        
        # Reset chromosome reference
        self._chromosome = None
        
        # Reinitialize eukaryotic structure
        self._initialize_eukaryotic_structure()
    
    def copy(self):
        """Create a shallow copy using object pooling."""
        gene = gene_pool.get(self.start, element_data.lengths[self.idx], self.subtype)
        gene.chromosome = self.chromosome
        return gene

class SelectiveInsertTE(Element):
    """
    Memory-efficient transposable element with optimized jumping behavior.
    Uses vectorized operations for batch jumping and collision detection.
    Supports autonomous and non-autonomous TE behavior.
    """
    
    def __init__(self, start: int, dead: bool = False, length: int = None, autonomous: bool = False, te_type: str = None):
        if length is None:
            if te_type:
                length = parameters.get_te_length(te_type)
            else:
                length = parameters.TE_length
        super().__init__(length, start, TE_TYPE)
        element_data.dead_flags[self.idx] = dead
        element_data.te_autonomous[self.idx] = autonomous
        self._te_type = te_type or self._determine_te_type()
    
    @property
    def dead(self) -> bool:
        return element_data.dead_flags[self.idx]
    
    @dead.setter
    def dead(self, value: bool):
        element_data.dead_flags[self.idx] = value
    
    @property
    def autonomous(self) -> bool:
        return element_data.te_autonomous[self.idx]
    
    @autonomous.setter
    def autonomous(self, value: bool):
        element_data.te_autonomous[self.idx] = value
    
    @property
    def te_type(self) -> str:
        return self._te_type
    
    def can_transpose(self) -> bool:
        """
        Check if this TE can transpose based on autonomous status and parasitism rules.
        Non-autonomous TEs require specific autonomous TEs to be present and active.
        """
        if self.dead:
            return False
        
        if self.autonomous:
            return True
        
        # Non-autonomous TEs need specific autonomous TEs to be present
        if self.chromosome and self.chromosome.host:
            # Get the TE types this TE parasitizes
            parasitism_targets = parameters.get_te_parasitism_targets(self.te_type)
            
            if not parasitism_targets:
                # No parasitism targets defined - fall back to any autonomous TE
                autonomous_tes = [te for te in self.chromosome.TEs(live=True, dead=False) 
                                if te.autonomous and not te.dead]
                return len(autonomous_tes) > 0
            
            # Check for specific autonomous TE types that this TE parasitizes
            for te in self.chromosome.TEs(live=True, dead=False):
                if (te.autonomous and not te.dead and 
                    te.te_type in parasitism_targets):
                    return True
            
            # No suitable autonomous TEs found
            return False
        
        return False
    
    def jump(self) -> Dict[str, int]:
        """
        Optimized jumping with vectorized operations and efficient collision detection.
        Returns jump effects without using exception-based control flow.
        Includes autonomous/non-autonomous TE behavior and type-specific parameters.
        """
        jump_effects = {
            'TEDEATH': 0, 'COLLISIO': 0, 'TOTAL_JU': 0, 'LETHAL_J': 0,
            'DELETE_J': 0, 'NEUTRA_J': 0, 'BENEFI_J': 0
        }
        
        # Check if TE is still alive and in chromosome
        if self not in self.chromosome.elements or self.dead:
            return jump_effects
        
        # Check if TE can transpose (autonomous or has autonomous helpers)
        if not self.can_transpose():
            output("SPLAT", f"TE at position {self.start} cannot transpose (autonomous: {self.autonomous}, dead: {self.dead})")
            return jump_effects
        
        # Import vrng locally to avoid circular import issues
        from TEUtil_ABM2 import vrng
        
        # Get type-specific death rate
        death_rate = parameters.get_te_death_rate(self.te_type)
        
        # Death check with cached random number
        if vrng.uniform() < death_rate:
            self.dead = True
            jump_effects['TEDEATH'] += 1
            output("SPLAT", f"TE at position {self.start} died during jump (type: {self.te_type}, death_rate: {death_rate})")
            return jump_effects
        
        # Get type-specific excision rate and progeny distribution
        excision_rate = parameters.get_te_excision_rate(self.te_type)
        progeny_dist = parameters.get_te_progeny_distribution(self.te_type)
        
        # Determine progeny count based on TE type
        if excision_rate == 0.0:  # Retrotransposon
            progeny = progeny_dist.generate()
        elif vrng.uniform() < excision_rate:  # DNA transposon excision
            self.chromosome.excise(self)
            progeny = progeny_dist.generate()
        else:  # DNA transposon no excision
            progeny = 0
        
        # Batch creation of progeny TEs
        if progeny > 0:
            output("SPLAT", f"TE at position {self.start} creating {progeny} progeny (type: {self.te_type}, autonomous: {self.autonomous})")
            new_tes = self._create_progeny_batch(progeny)
            for te in new_tes:
                result = self.chromosome.insert_optimized(te)
                jump_effects['TOTAL_JU'] += 1
                
                if result.collision_type == InsertResult.COLLISION_GENE:
                    output("SPLAT", f"Progeny TE at position {te.start} collided with gene (type: {self.te_type})")
                    # Handle gene collision with vectorized fitness calculation
                    self._handle_gene_collision(result.collided_element, jump_effects)
                elif result.collision_type == InsertResult.COLLISION_TE:
                    output("SPLAT", f"Progeny TE at position {te.start} collided with existing TE (type: {self.te_type})")
                    jump_effects['COLLISIO'] += 1
                else:
                    output("SPLAT", f"Progeny TE successfully inserted at position {te.start} (type: {self.te_type})")
        
        return jump_effects
    
    def _create_progeny_batch(self, count: int) -> List['SelectiveInsertTE']:
        """Create multiple progeny TEs efficiently using object pooling."""
        # Batch sample insertion positions
        positions = parameters.TE_Insertion_Distribution.sample(size=count)
        positions = (positions * self.chromosome.length).astype(int)
        
        # Ensure positions is always iterable
        if not hasattr(positions, '__iter__'):
            positions = [positions]
        
        progeny = []
        for pos in positions:
            # Inherit autonomous status and TE type from parent TE
            te = te_pool.get(pos, False, element_data.lengths[self.idx], self.autonomous, self.te_type)
            
            # Validate that we got the correct type
            if not isinstance(te, SelectiveInsertTE):
                logger.error(f"Wrong object type returned from te_pool: {type(te)} instead of SelectiveInsertTE")
                # Create a new TE directly instead of using the pool
                te = SelectiveInsertTE(pos, False, element_data.lengths[self.idx], self.autonomous, self.te_type)
            
            te.chromosome = self.chromosome
            progeny.append(te)
        
        return progeny
    
    def _handle_gene_collision(self, gene, jump_effects: Dict[str, int]):
        """Handle collision with gene using eukaryotic gene structure and TE type-specific effects."""
        ind = self.chromosome.host
        if ind is None or ind.fitness <= 0.0:
            return
        
        # Determine TE type for impact calculation
        te_type = self._determine_te_type()
        
        # Use eukaryotic gene's TE insertion impact calculation
        if hasattr(gene, 'get_te_insertion_impact'):
            # New eukaryotic gene with sophisticated impact calculation
            impact_multiplier = gene.get_te_insertion_impact(te_type)
        else:
            # Fallback for legacy genes
            gene_subtype = gene.subtype
            te_effect_level = parameters.get_gene_te_insertion_effect(gene_subtype)
            impact_multiplier = 1.0 if te_effect_level == 'medium' else (2.0 if te_effect_level == 'high' else 0.5)
        
        output("SPLAT FITNESS", f"Gene collision - TE type: {te_type}, Gene subtype: {gene.subtype}, Impact multiplier: {impact_multiplier:.3f}")
        
        # Calculate fitness effect based on eukaryotic gene structure
        if gene.subtype == 'ORF':
            if gene.has_splice_sites and te_type in ['SINE', 'LINE']:
                # TE insertion in intron of protein-coding gene - often less harmful
                fitness_effect = -0.1 * impact_multiplier
                jump_effects['NEUTRA_J'] += 1
            else:
                # TE insertion in exon - more harmful
                fitness_effect = -0.3 * impact_multiplier
                jump_effects['DELETE_J'] += 1
                
        elif gene.subtype == 'promoter':
            # Promoter regions are very sensitive to TE insertions
            fitness_effect = -0.5 * impact_multiplier
            jump_effects['DELETE_J'] += 1
            
        elif gene.subtype == 'enhancer':
            # Enhancer regions are moderately sensitive
            fitness_effect = -0.2 * impact_multiplier
            jump_effects['NEUTRA_J'] += 1
            
        elif gene.subtype == 'intron':
            # Intronic regions are least sensitive - TEs often insert here
            if te_type in ['SINE', 'LINE']:
                fitness_effect = -0.05 * impact_multiplier  # Very low impact
                jump_effects['NEUTRA_J'] += 1
            else:
                fitness_effect = -0.1 * impact_multiplier
                jump_effects['NEUTRA_J'] += 1
                
        elif gene.subtype == 'silencer':
            # Silencer regions are sensitive - can disrupt repression
            fitness_effect = -0.4 * impact_multiplier
            jump_effects['DELETE_J'] += 1
            
        elif gene.subtype == 'insulator':
            # Insulator regions are moderately sensitive
            fitness_effect = -0.15 * impact_multiplier
            jump_effects['NEUTRA_J'] += 1
            
        else:
            # Unknown gene type - moderate effect
            fitness_effect = -0.2 * impact_multiplier
            jump_effects['NEUTRA_J'] += 1
        
        # Apply fitness effect
        old_fitness = ind.fitness
        ind.fitness += fitness_effect
        if ind.fitness <= 0.0:
            ind.fitness = 0.0
            jump_effects['LETHAL_J'] += 1
        
        output("SPLAT FITNESS", f"Fitness change: {old_fitness:.4f} -> {ind.fitness:.4f} (effect: {fitness_effect:.4f})")
    
    def _determine_te_type(self) -> str:
        """Determine TE type based on length and autonomous status.
           Uses the configured TE types to determine the most likely type."""
        te_length = element_data.lengths[self.idx]
        is_autonomous = element_data.te_autonomous[self.idx]
        
        # If we have configured TE types, use them to determine type
        if parameters.TE_TYPES:
            # Find the TE type that best matches this TE's characteristics
            best_match = None
            best_score = float('inf')
            
            for te_type, config in parameters.TE_TYPES.items():
                # Calculate similarity score based on length and autonomous status
                length_diff = abs(te_length - config['length'])
                autonomous_match = (is_autonomous == config['autonomous'])
                
                # Weight length difference more heavily than autonomous status
                score = length_diff + (0 if autonomous_match else 1000)
                
                if score < best_score:
                    best_score = score
                    best_match = te_type
            
            if best_match:
                return best_match
        
        # Fallback to legacy determination based on length
        if te_length <= 1000:
            return 'SINE'
        elif te_length <= 6000:
            return 'LINE'
        elif te_length <= 10000:
            return 'LTR'
        else:
            return 'DNA_TRANSPOSON'
    
    def birth(self):
        """Create optimized copy using object pooling."""
        start = int(parameters.TE_Insertion_Distribution.sample() * self.chromosome.length)
        baby = te_pool.get(start, False, element_data.lengths[self.idx], self.autonomous, self.te_type)
        baby.chromosome = self.chromosome
        return baby
    
    def reset(self, start: int, dead: bool, length: int, autonomous: bool, te_type: str):
        """Reset TE data for object pool reuse."""
        # Reuse existing index or allocate new one
        if not hasattr(self, 'idx'):
            self.idx = element_data.allocate_element()
        
        # Update data in global arrays
        element_data.starts[self.idx] = start
        element_data.ends[self.idx] = start + length
        element_data.types[self.idx] = TE_TYPE
        element_data.lengths[self.idx] = length
        element_data.dead_flags[self.idx] = dead
        element_data.gene_subtypes[self.idx] = ''
        element_data.te_autonomous[self.idx] = autonomous
        
        # Update TE-specific attributes
        self._te_type = te_type or self._determine_te_type()
        
        # Reset chromosome reference
        self._chromosome = None
    
    def copy(self):
        """Efficient copying with object pool."""
        te_copy = te_pool.get(self.start, self.dead, element_data.lengths[self.idx], self.autonomous, self.te_type)
        te_copy.chromosome = self.chromosome
        return te_copy

################################################################################
# Object pools for memory efficiency
################################################################################

def create_gene(start=0, length=None, subtype=None):
    if length is None:
        if subtype and subtype in parameters.GENE_SUBTYPES:
            length = parameters.GENE_SUBTYPES[subtype]['length']
        else:
            subtype = parameters.get_random_gene_subtype()
            length = parameters.GENE_SUBTYPES[subtype]['length']
    return EukGene(start, length, subtype)

def create_te(start=0, dead=False, length=None, autonomous=False, te_type=None):
    if length is None:
        if te_type:
            length = parameters.get_te_length(te_type)
        else:
            length = parameters.TE_length
    return SelectiveInsertTE(start, dead, length, autonomous, te_type)

gene_pool = ObjectPool(create_gene, max_size=500)
te_pool = ObjectPool(create_te, max_size=2000)

################################################################################
# High-performance Chromosome with IntervalTree
################################################################################

class CollisionResult:
    """Result of element insertion attempt."""
    
    def __init__(self, collision_type=InsertResult.SUCCESS, collided_element=None):
        self.collision_type = collision_type
        self.collided_element = collided_element

class OptimizedChromosome:
    """
    High-performance chromosome using IntervalTree for O(log n) lookups
    and optimized batch operations for large-scale simulations.
    """
    
    def __init__(self, length: int = None, elements: List[Element] = None):
        self.length = length if length is not None else parameters.Junk_BP
        self.host = None
        
        # IntervalTree for fast spatial queries
        self.interval_tree = IntervalTree()
        
        # Sorted list of elements for linear operations  
        self.elements = []
        
        # Bit array for fast occupancy checking in large genomes
        if self.length > 1000000:  # Use bit array for large genomes
            self.occupancy_bits = BitArray(self.length)
        else:
            self.occupancy_bits = None
        
        # Statistics cache
        self._stats_cache = {}
        self._stats_dirty = True
        
        if elements:
            for element in elements:
                self._add_element_to_structures(element)
    
    def _add_element_to_structures(self, element: Element):
        """Add element to all internal data structures efficiently."""
        element.chromosome = self
        
        # Validate interval before adding to tree
        if element.start >= element.end:
            logger.warning(f"Invalid interval detected: start={element.start}, end={element.end}, length={element.length}")
            # Fix the interval by ensuring end > start
            if element.length > 0:
                element_data.ends[element.idx] = element.start + element.length
            else:
                # If length is also invalid, set a minimum length
                element_data.lengths[element.idx] = 1
                element_data.ends[element.idx] = element.start + 1
            logger.info(f"Fixed interval: start={element.start}, end={element.end}, length={element.length}")
        
        # Add to interval tree
        self.interval_tree.addi(element.start, element.end, element)
        
        # Add to sorted list using binary search
        bisect.insort_left(self.elements, element)
        
        # Update bit array if using
        if self.occupancy_bits:
            for pos in range(element.start, element.end):
                if pos < self.length:
                    self.occupancy_bits.set_bit(pos)
        
        self._stats_dirty = True
    
    def _remove_element_from_structures(self, element: Element):
        """Remove element from all internal data structures."""
        # Remove from interval tree - check if it exists first
        interval_to_remove = Interval(element.start, element.end, element)
        if interval_to_remove in self.interval_tree:
            self.interval_tree.remove(interval_to_remove)
        
        # Remove from sorted list
        if element in self.elements:
            self.elements.remove(element)
        
        # Update bit array if using
        if self.occupancy_bits:
            for pos in range(element.start, element.end):
                if pos < self.length:
                    self.occupancy_bits.clear_bit(pos)
        
        self._stats_dirty = True
    
    def __getitem__(self, index: int):
        """
        Fast element lookup using IntervalTree; O(log n) instead of O(n).
        Returns the element at the given index, or JUNK_TYPE if no element exists.
        """
        # Use bit array for very fast checking in large genomes
        if self.occupancy_bits and not self.occupancy_bits.get_bit(index):
            return JUNK_TYPE
        
        # IntervalTree lookup
        overlapping = self.interval_tree.at(index)
        if overlapping:
            return next(iter(overlapping)).data  # Return first overlapping element
        return JUNK_TYPE
    
    def insert_optimized(self, element: Element) -> CollisionResult:
        """
        Optimized insertion with efficient collision detection.
        Returns collision result instead of raising exceptions.
        """
        # Check what's at the insertion point
        existing = self[element.start]
        
        collision_result = CollisionResult()
        
        if existing != JUNK_TYPE:
            if element_data.types[existing.idx] == TE_TYPE:
                # Remove existing TE
                self._remove_element_from_structures(existing)
                te_pool.put(existing)  # Return to object pool
                collision_result.collision_type = InsertResult.COLLISION_TE
            else:  # Gene collision
                collision_result.collision_type = InsertResult.COLLISION_GENE
                collision_result.collided_element = existing
                # Don't remove gene, insert anyway and let caller handle fitness effects
        
        # Perform insertion with position adjustment
        self._insert_with_adjustment(element)
        return collision_result
    
    def _insert_with_adjustment(self, element: Element):
        """Insert element and adjust positions of downstream elements."""
        # Adjust positions of all elements that come after the insertion point
        affected_elements = []
        for existing in self.elements:
            if existing.start >= element.start:
                affected_elements.append(existing)
        
        # Remove affected elements temporarily
        for existing in affected_elements:
            self._remove_element_from_structures(existing)
        
        # Add new element
        self._add_element_to_structures(element)
        
        # Re-add affected elements with adjusted positions
        for existing in affected_elements:
            # Ensure the adjustment doesn't create negative positions
            new_start = max(0, existing.start + element.length)
            existing.start = new_start
            
            # Validate the element before re-adding
            if existing.start >= existing.end:
                logger.warning(f"Invalid element after adjustment: start={existing.start}, end={existing.end}, length={existing.length}")
                # Fix the element
                if existing.length > 0:
                    element_data.ends[existing.idx] = existing.start + existing.length
                else:
                    element_data.lengths[existing.idx] = 1
                    element_data.ends[existing.idx] = existing.start + 1
            
            self._add_element_to_structures(existing)
        
        # Update chromosome length
        self.length += element.length
    
    def excise(self, element: Element):
        """Remove element and shift following elements back."""
        if element not in self.elements:
            return
        
        # Find elements that need position adjustment
        affected_elements = []
        for existing in self.elements:
            if existing.start > element.start:
                affected_elements.append(existing)
        
        # Remove affected elements temporarily
        for existing in affected_elements:
            self._remove_element_from_structures(existing)
        
        # Remove the target element
        self._remove_element_from_structures(element)
        
        # Re-add affected elements with adjusted positions
        for existing in affected_elements:
            existing.start -= element.length
            self._add_element_to_structures(existing)
        
        # Update chromosome length
        self.length -= element.length
        
        # Return element to pool
        if element_data.types[element.idx] == TE_TYPE:
            te_pool.put(element)
        else:
            gene_pool.put(element)
    
    def genes(self, subtype: str = None) -> List[EukGene]:
        """Get genes with optional subtype filtering and caching for repeated calls."""
        cache_key = f'genes_{subtype}' if subtype else 'genes'
        
        if self._stats_dirty or cache_key not in self._stats_cache:
            genes = [e for e in self.elements if element_data.types[e.idx] == GENE_TYPE]
            
            if subtype:
                genes = [g for g in genes if g.subtype == subtype]
            
            self._stats_cache[cache_key] = genes
        
        return self._stats_cache[cache_key]
    
    def TEs(self, live: bool = True, dead: bool = True) -> List[SelectiveInsertTE]:
        """Get TEs with specified life status, using caching."""
        cache_key = f'TEs_{live}_{dead}'
        if self._stats_dirty or cache_key not in self._stats_cache:
            tes = []
            for e in self.elements:
                # Check both the type array and the actual class type
                if (element_data.types[e.idx] == TE_TYPE and 
                    isinstance(e, SelectiveInsertTE)):
                    is_dead = element_data.dead_flags[e.idx]
                    if (live and not is_dead) or (dead and is_dead):
                        tes.append(e)
            self._stats_cache[cache_key] = tes
        return self._stats_cache[cache_key]
    
    def junk(self) -> int:
        """Calculate junk BP with caching."""
        if self._stats_dirty or 'junk' not in self._stats_cache:
            total_element_length = sum(element_data.lengths[e.idx] for e in self.elements)
            self._stats_cache['junk'] = self.length - total_element_length
        return self._stats_cache['junk']
    
    def jump_batch(self) -> Dict[str, int]:
        """
        Optimized batch jumping for all live TEs using vectorized operations.
        Significantly faster than individual TE jumping.
        """
        live_tes = self.TEs(live=True, dead=False)
        if not live_tes:
            return {'TEDEATH': 0, 'COLLISIO': 0, 'TOTAL_JU': 0, 
                   'LETHAL_J': 0, 'DELETE_J': 0, 'NEUTRA_J': 0, 'BENEFI_J': 0}
        
        # Import vrng locally to avoid circular import issues
        from TEUtil_ABM2 import vrng
        
        # Vectorized death probability calculation
        death_probs = vrng.uniform(size=len(live_tes))
        death_mask = death_probs < parameters.TE_death_rate
        
        # Ensure death_mask is always an array
        if not hasattr(death_mask, '__iter__'):
            death_mask = [death_mask]
        
        # Apply death to TEs
        total_jump_effects = {'TEDEATH': 0, 'COLLISIO': 0, 'TOTAL_JU': 0,
                             'LETHAL_J': 0, 'DELETE_J': 0, 'NEUTRA_J': 0, 'BENEFI_J': 0}
        
        for i, te in enumerate(live_tes):
            if death_mask[i]:
                te.dead = True
                total_jump_effects['TEDEATH'] += 1
            else:
                # Individual jump for live TEs (could be further optimized)
                te_effects = te.jump()
                for key, value in te_effects.items():
                    total_jump_effects[key] += value
        
        self._stats_dirty = True
        return total_jump_effects
    
    def copy(self, host):
        """Efficient chromosome copying with copy-on-write semantics."""
        new_chromosome = OptimizedChromosome(length=self.length)
        new_chromosome.host = host
        
        # Copy elements using object pools
        for element in self.elements:
            new_element = element.copy()
            new_element.chromosome = new_chromosome
            new_chromosome._add_element_to_structures(new_element)
        
        return new_chromosome

################################################################################
# Optimized Chromosome subclass with batch element addition
################################################################################

class TestChromosome2(OptimizedChromosome):
    """
    Optimized chromosome with efficient batch element initialization.
    Uses vectorized operations for placing initial genes and TEs.
    """
    
    def __init__(self, length: int = None):
        super().__init__(length or parameters.Junk_BP)
    
    def add_elements(self, genes: int = None, TEs: int = None):
        """
        Batch addition of elements using vectorized position generation
        and efficient collision handling.
        """
        genes = genes or parameters.Initial_genes
        TEs = TEs or parameters.Initial_TEs
        
        # Batch generate gene positions
        self._add_genes_batch(genes)
        
        # Batch generate TE positions  
        self._add_tes_batch(TEs)
        
        self._stats_dirty = True
    
    def _add_genes_batch(self, count: int):
        """Add genes in batches with vectorized position generation and subtype distribution."""
        attempts = 0
        max_attempts = count * 10  # Prevent infinite loops
        
        output("GENE INIT", f"Starting to add {count} genes")
        
        while len(self.genes()) < count and attempts < max_attempts:
            # Generate batch of positions
            batch_size = min(50, count - len(self.genes()))
            positions = parameters.Gene_Insertion_Distribution.sample(size=batch_size)
            positions = (positions * self.length).astype(int)
            
            # Ensure positions is always iterable
            if not hasattr(positions, '__iter__'):
                positions = [positions]
            
            for pos in positions:
                if len(self.genes()) >= count:
                    break
                
                try:
                    # Get random gene subtype for this gene
                    subtype = parameters.get_random_gene_subtype()
                    gene = gene_pool.get(pos, None, subtype)
                    result = self.insert_optimized(gene)
                    
                    if result.collision_type == InsertResult.COLLISION_GENE:
                        output("GENE INIT", f"Gene collision at position {pos}, subtype {subtype}")
                        if hasattr(parameters, "Append_gene") and parameters.Append_gene:
                            # Move gene to end of collided gene
                            gene.start = result.collided_element.end
                            self._add_element_to_structures(gene)
                            output("GENE INIT", f"Appended gene to end of collided gene at position {gene.start}")
                        else:
                            gene_pool.put(gene)  # Return to pool
                            continue
                    
                    attempts += 1
                    
                except Exception as e:
                    output("GENE INIT", f"Exception during gene creation: {e}")
                    attempts += 1
                    continue
        
        output("GENE INIT", f"Added {len(self.genes())} genes after {attempts} attempts")
        
        # Log gene subtype distribution
        if hasattr(parameters, 'GENE_SUBTYPES'):
            for subtype in parameters.GENE_SUBTYPES.keys():
                subtype_count = len(self.genes(subtype=subtype))
                output("GENE INIT", f"Gene subtype {subtype}: {subtype_count} genes")
    
    def _add_tes_batch(self, count: int):
        """Add TEs in batches with efficient collision handling."""
        attempts = 0
        max_attempts = count * 10
        
        output("TE INIT", f"Starting to add {count} TEs")
        
        while len(self.TEs()) < count and attempts < max_attempts:
            batch_size = min(20, count - len(self.TEs()))
            positions = parameters.TE_Insertion_Distribution.sample(size=batch_size)
            positions = (positions * self.length).astype(int)
            
            # Ensure positions is always iterable
            if not hasattr(positions, '__iter__'):
                positions = [positions]
            
            for pos in positions:
                if len(self.TEs()) >= count:
                    break
                
                # Randomly select TE type and get its autonomous status
                te_type = parameters.get_random_te_type()
                autonomous = parameters.get_te_autonomous_status(te_type)
                # Get TE length for the specified type
                te_length = parameters.get_te_length(te_type)
                te = te_pool.get(pos, False, te_length, autonomous, te_type)
                result = self.insert_optimized(te)
                
                if result.collision_type == InsertResult.COLLISION_GENE:
                    output("TE INIT", f"TE-gene collision at position {pos}, TE type {te_type}, autonomous {autonomous}")
                elif result.collision_type == InsertResult.COLLISION_TE:
                    output("TE INIT", f"TE-TE collision at position {pos}, TE type {te_type}")
                
                attempts += 1
        
        output("TE INIT", f"Added {len(self.TEs())} TEs after {attempts} attempts")
        
        # Log TE type distribution
        if hasattr(parameters, 'TE_TYPES'):
            for te_type in parameters.TE_TYPES.keys():
                te_type_count = len([te for te in self.TEs() if te.te_type == te_type])
                output("TE INIT", f"TE type {te_type}: {te_type_count} TEs")
        
        # Log autonomous vs non-autonomous distribution
        autonomous_count = len([te for te in self.TEs() if te.autonomous])
        non_autonomous_count = len([te for te in self.TEs() if not te.autonomous])
        output("TE INIT", f"Autonomous TEs: {autonomous_count}, Non-autonomous TEs: {non_autonomous_count}")

################################################################################
# Optimized Host with vectorized operations
################################################################################

class Host:
    """
    Optimized host with efficient chromosome management and
    vectorized mutation operations.
    """
    
    def __init__(self, species, chromosome=None, fitness: float = None):
        self.species = species
        self.fitness = fitness or parameters.Host_start_fitness
        
        if chromosome is None:
            self.chromosome = [cls() for cls in self.species.chromosomes]
        else:
            self.chromosome = chromosome
        
        # Set host references
        for chrom in self.chromosome:
            chrom.host = self
    
    def jump_and_mutate(self) -> Dict[str, int]:
        """
        Optimized jumping and mutation using vectorized operations
        and efficient batch processing.
        """
        total_jump_effects = {'TEDEATH': 0, 'COLLISIO': 0, 'TOTAL_JU': 0,
                             'LETHAL_J': 0, 'DELETE_J': 0, 'NEUTRA_J': 0, 'BENEFI_J': 0}
        
        # Batch jump operations across all chromosomes
        for chromosome in self.chromosome:
            chrom_effects = chromosome.jump_batch()
            for key, value in chrom_effects.items():
                total_jump_effects[key] += value
        
        # Import vrng locally to avoid circular import issues
        from TEUtil_ABM2 import vrng
        
        # Host mutation with cached random number
        if vrng.uniform() < parameters.Host_mutation_rate and self.fitness > 0.0:
            mutation_multiplier = parameters.Host_mutation.generate()
            new_fitness = self.fitness * mutation_multiplier
            self.fitness = max(0.0, new_fitness)
        
        return total_jump_effects
    
    def clone(self):
        """Efficient cloning with copy-on-write semantics."""
        new_host = Host(self.species, [], self.fitness)
        
        # Copy chromosomes using optimized copying
        new_chromosomes = [chrom.copy(new_host) for chrom in self.chromosome]
        new_host.chromosome = new_chromosomes
        
        return new_host

################################################################################
# Species class (unchanged but with type hints)
################################################################################

class Species:
    """Represents species with optimized chromosome types."""
    
    def __init__(self, cells: int, chromosomes: List[type]):
        self.cells = cells
        self.chromosomes = chromosomes

################################################################################
# Optimized Population with parallel processing
################################################################################

class Population:
    """
    High-performance population with parallel processing support
    and vectorized operations for large-scale simulations.
    """
    
    def __init__(self, capacity: int, species: Species, individual=None, generation_no: int = 0):
        self.capacity = capacity
        self.species = species
        self.generation_no = generation_no
        self.use_parallel = capacity > 100  # Use parallel processing for large populations
        self.n_workers = min(cpu_count(), 8)  # Limit workers to prevent memory issues
        
        if individual is None:
            # Create initial population
            host = Host(species)
            host.chromosome[0].add_elements()  
            self.individual = [host.clone() for _ in range(capacity)]
        else:
            self.individual = individual
    
    def replication(self):
        """
        Optimized replication using parallel processing for large populations.
        """
        if self.use_parallel and len(self.individual) > 50:
            # Parallel replication for large populations
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                for dup in range(parameters.Host_reproduction_rate):
                    # Submit cloning tasks in batches
                    batch_size = max(10, len(self.individual) // self.n_workers)
                    futures = []
                    
                    for i in range(0, len(self.individual), batch_size):
                        batch = self.individual[i:i+batch_size]
                        future = executor.submit(self._clone_batch, batch)
                        futures.append(future)
                    
                    # Collect results
                    new_individuals = []
                    for future in futures:
                        new_individuals.extend(future.result())
                    
                    self.individual.extend(new_individuals)
        else:
            # Sequential replication for small populations
            for dup in range(parameters.Host_reproduction_rate):
                self.individual.extend([ind.clone() for ind in self.individual])
    
    def _clone_batch(self, individuals: List[Host]) -> List[Host]:
        """Clone a batch of individuals efficiently."""
        return [ind.clone() for ind in individuals]
    
    def jump_and_mutate(self) -> Dict[str, int]:
        """
        Parallel jumping and mutation with result aggregation.
        Uses vectorized operations and efficient batch processing.
        """
        total_effects = {'TEDEATH': 0, 'COLLISIO': 0, 'TOTAL_JU': 0,
                        'LETHAL_J': 0, 'DELETE_J': 0, 'NEUTRA_J': 0, 'BENEFI_J': 0}
        
        # Temporarily disable parallel processing to avoid hanging
        # if self.use_parallel and len(self.individual) > 50:
        #     # Parallel processing for large populations
        #     with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
        #         batch_size = max(10, len(self.individual) // self.n_workers)
        #         futures = []
        #         
        #         for i in range(0, len(self.individual), batch_size):
        #             batch = self.individual[i:i+batch_size]
        #             future = executor.submit(self._jump_and_mutate_batch, batch)
        #             futures.append(future)
        #         
        #         # Aggregate results
        #         for future in futures:
        #             batch_effects = future.result()
        #             for key, value in batch_effects.items():
        #             total_effects[key] += value
        # else:
        # Sequential processing for small populations
        for individual in self.individual:
            ind_effects = individual.jump_and_mutate()
            for key, value in ind_effects.items():
                total_effects[key] += value
        
        return total_effects
    
    def _jump_and_mutate_batch(self, individuals: List[Host]) -> Dict[str, int]:
        """Process a batch of individuals for jumping and mutation."""
        batch_effects = {'TEDEATH': 0, 'COLLISIO': 0, 'TOTAL_JU': 0,
                        'LETHAL_J': 0, 'DELETE_J': 0, 'NEUTRA_J': 0, 'BENEFI_J': 0}
        
        for individual in individuals:
            ind_effects = individual.jump_and_mutate()
            for key, value in ind_effects.items():
                batch_effects[key] += value
        
        return batch_effects
    
    def selection_and_drift(self):
        """
        Optimized selection using vectorized fitness calculations
        and efficient probability-based survival.
        """
        if not self.individual:
            return
        
        # Vectorized fitness calculation
        fitnesses = np.array([ind.fitness for ind in self.individual])
        total_fitness = np.sum(fitnesses)
        
        if total_fitness > 0.0:
            # Vectorized survival probability calculation
            survival_probs = batch_survival_probability(
                fitnesses, total_fitness, parameters.Carrying_capacity
            )
            
            # Import vrng locally to avoid circular import issues
            from TEUtil_ABM2 import vrng
            
            # Vectorized random selection
            random_vals = vrng.uniform(size=len(self.individual))
            survivors_mask = random_vals < survival_probs
            
            # Filter survivors
            self.individual = [ind for i, ind in enumerate(self.individual) 
                             if survivors_mask[i]]
        else:
            self.individual = []
    
    def generation(self) -> Dict[str, int]:
        """
        Optimized generation processing with performance monitoring
        and memory management.
        """
        # Monitor performance
        perf_monitor.log_generation_stats(self.generation_no)
        
        # Log generation start
        output("GENERATION", f"Starting generation {self.generation_no} with {len(self.individual)} individuals")
        
        # Run generation steps
        initial_pop = len(self.individual)
        self.replication()
        output("GENERATION", f"After replication: {len(self.individual)} individuals (was {initial_pop})")
        
        jump_effects = self.jump_and_mutate()
        output("GENERATION", f"Jump effects: {jump_effects}")
        
        # Log detailed jump effects
        if jump_effects['TOTAL_JU'] > 0:
            output("GENERATION", f"Total jumps: {jump_effects['TOTAL_JU']}")
            output("GENERATION", f"TE deaths: {jump_effects['TEDEATH']}")
            output("GENERATION", f"Collisions: {jump_effects['COLLISIO']}")
            output("GENERATION", f"Lethal jumps: {jump_effects['LETHAL_J']}")
            output("GENERATION", f"Deleterious jumps: {jump_effects['DELETE_J']}")
            output("GENERATION", f"Neutral jumps: {jump_effects['NEUTRA_J']}")
            output("GENERATION", f"Beneficial jumps: {jump_effects['BENEFI_J']}")
        
        before_selection = len(self.individual)
        self.selection_and_drift()
        after_selection = len(self.individual)
        output("GENERATION", f"Selection: {before_selection} -> {after_selection} individuals")
        
        # Log fitness statistics
        if self.individual:
            fitnesses = [ind.fitness for ind in self.individual]
            avg_fitness = sum(fitnesses) / len(fitnesses)
            min_fitness = min(fitnesses)
            max_fitness = max(fitnesses)
            output("GENERATION", f"Fitness stats - Avg: {avg_fitness:.4f}, Min: {min_fitness:.4f}, Max: {max_fitness:.4f}")
        
        self.generation_no += 1
        
        # Periodic memory optimization
        if self.generation_no % 50 == 0:
            optimize_memory_usage()
            output("GENERATION", "Memory optimization completed")
        
        return jump_effects

################################################################################
# Optimized data collection with reduced overhead
################################################################################

class OptimizedTracefile:
    """
    High-performance trace file writer with configurable collection frequency
    and efficient batch writing to reduce I/O overhead.
    """
    
    def __init__(self, collection_frequency: int = 1):
        self.collection_frequency = collection_frequency
        self.buffer = []
        self.buffer_size = 100  # Write in batches
        
        # Same format as original but with better performance and gene subtype support
        self.values = [
            ("time", "8.1f"), ("gen", "8d"), ("pop_size", "8d"),
            ("LTETOTAL", "8d"), ("LTE000pe", "8d"), ("LTE025pe", "8d"),
            ("LTE050pe", "8d"), ("LTE075pe", "8d"), ("LTE100pe", "8d"),
            ("DTETOTAL", "8d"), ("DTE000pe", "8d"), ("DTE025pe", "8d"),
            ("DTE050pe", "8d"), ("DTE075pe", "8d"), ("DTE100pe", "8d"),
            ("AUTONOMOUS_TOTAL", "8d"), ("AUTONOMOUS000pe", "8d"), ("AUTONOMOUS025pe", "8d"),
            ("AUTONOMOUS050pe", "8d"), ("AUTONOMOUS075pe", "8d"), ("AUTONOMOUS100pe", "8d"),
            ("NONAUTONOMOUS_TOTAL", "8d"), ("NONAUTONOMOUS000pe", "8d"), ("NONAUTONOMOUS025pe", "8d"),
            ("NONAUTONOMOUS050pe", "8d"), ("NONAUTONOMOUS075pe", "8d"), ("NONAUTONOMOUS100pe", "8d"),
            ("FIT000pe", "8.6f"), ("FIT025pe", "8.6f"), ("FIT050pe", "8.6f"),
            ("FIT075pe", "8.6f"), ("FIT100pe", "8.6f"), ("TEDEATH", "8d"),
            ("COLLISIO", "8d"), ("TOTAL_JU", "8d"), ("LETHAL_J", "8d"),
            ("DELETE_J", "8d"), ("NEUTRA_J", "8d"), ("BENEFI_J", "8d"),
            ("GSIZE000", "8d"), ("GSIZE025", "8d"), ("GSIZE050", "8d"),
            ("GSIZE075", "8d"), ("GSIZE100", "8d"), ("TELOC000", "8d"),
            ("TELOC025", "8d"), ("TELOC050", "8d"), ("TELOC075", "8d"),
            ("TELOC100", "8d"), ("GELOC000", "8d"), ("GELOC025", "8d"),
            ("GELOC050", "8d"), ("GELOC075", "8d"), ("GELOC100", "8d")
        ]
        
        # Add TE type columns dynamically
        if hasattr(parameters, 'TE_TYPES') and parameters.TE_TYPES:
            for te_type in parameters.TE_TYPES.keys():
                self.values.extend([
                    (f"{te_type.upper()}_TOTAL", "8d"),
                    (f"{te_type.upper()}000pe", "8d"),
                    (f"{te_type.upper()}025pe", "8d"),
                    (f"{te_type.upper()}050pe", "8d"),
                    (f"{te_type.upper()}075pe", "8d"),
                    (f"{te_type.upper()}100pe", "8d")
                ])
        
        # Add gene subtype columns dynamically
        if hasattr(parameters, 'GENE_SUBTYPES') and parameters.GENE_SUBTYPES:
            for subtype in parameters.GENE_SUBTYPES.keys():
                self.values.extend([
                    (f"{subtype.upper()}_TOTAL", "8d"),
                    (f"{subtype.upper()}000pe", "8d"),
                    (f"{subtype.upper()}025pe", "8d"),
                    (f"{subtype.upper()}050pe", "8d"),
                    (f"{subtype.upper()}075pe", "8d"),
                    (f"{subtype.upper()}100pe", "8d")
                ])
        
        self.headerstr = ", ".join([f"{item[0]:>8s}" for item in self.values]) + '\n'
        self.formatstr = ", ".join([f"%%(%s)%s" % item for item in self.values]) + '\n'
        
        # Initialize file
        if os.path.exists("trace.csv"):
            self.fp = open("trace.csv", "a", 1)
        else:
            self.fp = open("trace.csv", "w", 1)
            self.fp.write(self.headerstr)
    
    def trace(self, valdict: Dict[str, Any], generation: int):
        """
        Trace data with configurable frequency and batch writing.
        """
        if generation % self.collection_frequency != 0:
            return
        
        self.buffer.append(self.formatstr % valdict)
        
        if len(self.buffer) >= self.buffer_size:
            self._flush_buffer()
    
    def _flush_buffer(self):
        """Write buffered data to file."""
        if self.buffer:
            self.fp.writelines(self.buffer)
            self.fp.flush()
            self.buffer.clear()
    
    def close(self):
        """Close file and flush any remaining data."""
        self._flush_buffer()
        self.fp.close()

################################################################################
# High-performance Experiment class
################################################################################

class Experiment:
    """
    Optimized experiment with performance monitoring,  
    configurable data collection, and memory management.
    """
    
    def __init__(self, statefile: Optional[str] = None, 
                 trace_frequency: int = 1, use_parallel: bool = True):
        self.trace_frequency = trace_frequency
        self.use_parallel = use_parallel
        
        if statefile:
            self.load(statefile)
            output("LOADING", f"Loaded {statefile}")
        else:
            test_species1 = Species(1, [TestChromosome2])
            self.pop = Population(parameters.Carrying_capacity, test_species1)
            
            # Enable parallel processing if requested in config; see UTILs and README for more detail
            self.pop.use_parallel = use_parallel
        
        # Performance monitoring
        global perf_monitor
        perf_monitor = PerformanceMonitor()
        
        # Log initial statistics
        c0 = self.pop.individual[0].chromosome[0]
        output("INITIALIZATION", 
               f"Experiment.__init__: pop {len(self.pop.individual)} "
               f"TEs {len(c0.TEs())} genes {len(c0.genes())}")
        
        # Log detailed initial statistics
        live_tes = len(c0.TEs(live=True, dead=False))
        dead_tes = len(c0.TEs(live=False, dead=True))
        output("INITIALIZATION", f"TE breakdown: {live_tes} live, {dead_tes} dead")
        
        # Log gene subtypes if available
        if hasattr(parameters, 'GENE_SUBTYPES'):
            for subtype in parameters.GENE_SUBTYPES.keys():
                subtype_count = len(c0.genes(subtype=subtype))
                output("INITIALIZATION", f"Gene subtype {subtype}: {subtype_count}")
        
        # Log TE types if available
        if hasattr(parameters, 'TE_TYPES'):
            for te_type in parameters.TE_TYPES.keys():
                te_type_count = len([te for te in c0.TEs() if te.te_type == te_type])
                output("INITIALIZATION", f"TE type {te_type}: {te_type_count}")
    
    def save(self):
        """Save experiment state with compression."""
        filename = f"state-{self.pop.generation_no:07d}.gz"
        with gzip.open(filename, "w") as fp:
            fp.write(f"import random; random.setstate({repr(random.getstate())});\n".encode())
            fp.write(f"self.pop = {repr(self.pop)};\n".encode())
    
    def load(self, statefile: str):
        """Load experiment state."""
        with gzip.open(statefile, "r") as fp:
            exec(fp.read().decode())
    
    def sim_generations(self):
        """
        Run optimized simulation with configurable data collection,
        performance monitoring, and memory management.
        """
        tf = OptimizedTracefile(self.trace_frequency)
        
        try:
            # Save initial state
            self.save()
            
            # Initial trace
            tracedict = self.get_tracedict()
            tf.trace(tracedict, 0)
            
            # Main simulation loop
            while self.pop.generation_no < parameters.Maximum_generations:
                generation_start = time.time()
                
                output("GENERATION", f"Generation: {self.pop.generation_no}")
                
                # Run generation with optimized operations
                te_effects = self.pop.generation()
                
                # Collect and trace statistics
                tracedict = self.get_tracedict()
                tracedict.update(te_effects)
                tf.trace(tracedict, self.pop.generation_no)
                
                # Check termination conditions
                if not self.pop.individual:
                    output("HOST EXTINCTION", 
                           f"Host extinction after {self.pop.generation_no} generations")
                    break
                
                if not tracedict['LTETOTAL'] > 0:
                    output("TE EXTINCTION", 
                           f"TE extinction after {self.pop.generation_no} generations")
                    if hasattr(parameters, "Terminate_no_TEs") and parameters.Terminate_no_TEs:
                        break
                
                # Periodic saving
                if self.pop.generation_no % parameters.save_frequency == 0:
                    self.save()
                
                # Performance logging
                generation_time = time.time() - generation_start
                if self.pop.generation_no % 10 == 0:
                    print(f"Generation {self.pop.generation_no} completed in {generation_time:.3f}s")
                    
        finally:
            tf.close()
    
    def get_tracedict(self) -> Dict[str, Any]:
        """
        Optimized statistics collection with caching and vectorized calculations.
        Includes gene subtype statistics and TE type/autonomous statistics.
        """
        # Use vectorized operations for statistical calculations
        live_tes = []
        dead_tes = []
        fitnesses = []
        genome_sizes = []
        te_locs = []
        gene_locs = []
        
        # TE type and autonomous statistics
        te_type_counts = {}
        autonomous_te_counts = []
        non_autonomous_te_counts = []
        
        if parameters.TE_TYPES:
            for te_type in parameters.TE_TYPES.keys():
                te_type_counts[te_type] = []
        
        # Gene subtype statistics
        gene_subtype_counts = {}
        if parameters.GENE_SUBTYPES:
            for subtype in parameters.GENE_SUBTYPES.keys():
                gene_subtype_counts[subtype] = []
        
        for individual in self.pop.individual:
            chrom = individual.chromosome[0]
            
            # Use cached statistics when possible
            live_te_count = len(chrom.TEs(live=True, dead=False))
            dead_te_count = len(chrom.TEs(live=False, dead=True))
            
            live_tes.append(live_te_count)
            dead_tes.append(dead_te_count)
            fitnesses.append(individual.fitness)
            genome_sizes.append(chrom.length)
            
            # Collect TE type and autonomous statistics
            autonomous_count = 0
            non_autonomous_count = 0
            
            for te in chrom.TEs(live=True, dead=False):
                te_locs.append(te.start)
                if te.autonomous:
                    autonomous_count += 1
                else:
                    non_autonomous_count += 1
                
                # Count by TE type
                if parameters.TE_TYPES and te.te_type in te_type_counts:
                    te_type_counts[te.te_type].append(1)
                else:
                    # For TEs without a specific type, count as 'OTHER'
                    if 'OTHER' not in te_type_counts:
                        te_type_counts['OTHER'] = []
                    te_type_counts['OTHER'].append(1)
            
            autonomous_te_counts.append(autonomous_count)
            non_autonomous_te_counts.append(non_autonomous_count)
            
            for gene in chrom.genes():
                gene_locs.append(gene.start)
            
            # Collect gene subtype counts
            if parameters.GENE_SUBTYPES:
                for subtype in parameters.GENE_SUBTYPES.keys():
                    subtype_count = len(chrom.genes(subtype=subtype))
                    gene_subtype_counts[subtype].append(subtype_count)
        
        # Vectorized percentile calculations
        live_tes = np.array(live_tes)
        dead_tes = np.array(dead_tes) 
        fitnesses = np.array(fitnesses)
        genome_sizes = np.array(genome_sizes)
        te_locs = np.array(te_locs) if te_locs else np.array([0])
        gene_locs = np.array(gene_locs) if gene_locs else np.array([0])
        autonomous_te_counts = np.array(autonomous_te_counts)
        non_autonomous_te_counts = np.array(non_autonomous_te_counts)
        
        # Calculate percentiles efficiently
        def fast_percentiles(arr, prefix):
            if len(arr) == 0:
                return {f'{prefix}{p:03d}pe': 0 for p in [0, 25, 50, 75, 100]}
            
            percentiles = np.percentile(arr, [0, 25, 50, 75, 100])
            return {f'{prefix}{p:03d}pe': int(val) for p, val in zip([0, 25, 50, 75, 100], percentiles)}
        
        def fast_percentiles_no_pe(arr, prefix):
            if len(arr) == 0:
                return {f'{prefix}{p:03d}': 0 for p in [0, 25, 50, 75, 100]}
            
            percentiles = np.percentile(arr, [0, 25, 50, 75, 100])
            return {f'{prefix}{p:03d}': int(val) for p, val in zip([0, 25, 50, 75, 100], percentiles)}
        
        tracedict = {
            'time': time.time(),
            'gen': self.pop.generation_no,
            'pop_size': len(self.pop.individual),
            'LTETOTAL': np.sum(live_tes),
            'DTETOTAL': np.sum(dead_tes),
            'AUTONOMOUS_TOTAL': np.sum(autonomous_te_counts),
            'NONAUTONOMOUS_TOTAL': np.sum(non_autonomous_te_counts),
            'TEDEATH': 0,  # Will be updated by generation effects
            'COLLISIO': 0,
            'TOTAL_JU': 0,
            'LETHAL_J': 0,
            'DELETE_J': 0,
            'NEUTRA_J': 0,
            'BENEFI_J': 0,
        }
        
        # Add TE type statistics
        if parameters.TE_TYPES:
            for te_type in parameters.TE_TYPES.keys():
                if te_type in te_type_counts and te_type_counts[te_type]:
                    type_counts = np.array(te_type_counts[te_type])
                    tracedict[f'{te_type.upper()}_TOTAL'] = np.sum(type_counts)
                    tracedict.update(fast_percentiles(type_counts, f'{te_type.upper()}'))
                else:
                    tracedict[f'{te_type.upper()}_TOTAL'] = 0
                    tracedict.update(fast_percentiles(np.array([0]), f'{te_type.upper()}'))
        
        # Add autonomous/non-autonomous TE percentiles
        tracedict.update(fast_percentiles(autonomous_te_counts, 'AUTONOMOUS'))
        tracedict.update(fast_percentiles(non_autonomous_te_counts, 'NONAUTONOMOUS'))
        
        # Add gene subtype statistics
        if parameters.GENE_SUBTYPES:
            for subtype in parameters.GENE_SUBTYPES.keys():
                subtype_counts = np.array(gene_subtype_counts[subtype])
                tracedict[f'{subtype.upper()}_TOTAL'] = np.sum(subtype_counts)
                tracedict.update(fast_percentiles(subtype_counts, f'{subtype.upper()}'))
        
        # Add percentile data
        tracedict.update(fast_percentiles(live_tes, 'LTE'))
        tracedict.update(fast_percentiles(dead_tes, 'DTE'))
        tracedict.update({f'FIT{p:03d}pe': val for p, val in 
                         zip([0, 25, 50, 75, 100], np.percentile(fitnesses, [0, 25, 50, 75, 100]))})
        tracedict.update(fast_percentiles_no_pe(genome_sizes, 'GSIZE'))
        tracedict.update(fast_percentiles_no_pe(te_locs, 'TELOC'))
        tracedict.update(fast_percentiles_no_pe(gene_locs, 'GELOC'))
        
        return tracedict

################################################################################
# Optimized output function
################################################################################

def main():
    """Optimized main execution with performance monitoring."""
    # Set random seed for reproducibility
    if hasattr(parameters, 'seed') and parameters.seed is not None:
        set_random_seed(parameters.seed)
    
    # Create and run optimized experiment
    experiment = Experiment(
        statefile=getattr(parameters, 'saved', None),
        trace_frequency=getattr(parameters, 'trace_frequency', 1),
        use_parallel=True
    )
    
    experiment.sim_generations()

if __name__ == "__main__":
    main() 