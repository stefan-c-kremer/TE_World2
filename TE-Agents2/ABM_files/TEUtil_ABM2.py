"""
Optimized utilities for TE simulation with vectorized operations and memory efficiency.
Implements batch operations, object pooling, and numpy-based calculations for performance.
Supports GPU acceleration via CuPy when available.
"""
import math
import numpy as np
from numba import jit, njit
from typing import Union, Tuple, Optional, List
import warnings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GPU detection and imports
try:
    import cupy as cp
    GPU_AVAILABLE = True
    # Test if CUDA is actually working
    try:
        test_array = cp.array([1, 2, 3])
        del test_array
        GPU_FRAMEWORK = "cupy_cuda"
    except:
        GPU_AVAILABLE = False
        GPU_FRAMEWORK = None
except ImportError:
    # Try PyTorch as alternative GPU framework
    try:
        import torch
        if torch.cuda.is_available():
            GPU_AVAILABLE = True
            GPU_FRAMEWORK = "pytorch_cuda"
        elif hasattr(torch, 'hip') and torch.hip.is_available():  # AMD ROCm support
            GPU_AVAILABLE = True
            GPU_FRAMEWORK = "pytorch_rocm"
        else:
            GPU_AVAILABLE = False
            GPU_FRAMEWORK = None
    except ImportError:
        GPU_AVAILABLE = False
        GPU_FRAMEWORK = None

# Suppress numba warnings for cleaner output
# TODO: add levels of logging verbosity. suppress some warnings by default.
# Filter numpy warnings (compatible with different numpy versions)
try:
    warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
except AttributeError:
    # In newer numpy versions, this warning may be in a different location or removed...
    pass

# Object pools for memory efficiency
_triangle_pool = []
_flat_pool = []
_probability_table_pool = []

class ObjectPool:
    """Generic object pool for memory-efficient object reuse."""
    
    def __init__(self, factory_func, max_size=1000):
        self.factory_func = factory_func
        self.max_size = max_size
        self.pool = []
    
    def get(self, *args, **kwargs):
        """Get an object from the pool or create a new one."""
        if self.pool:
            obj = self.pool.pop()
            if hasattr(obj, 'reset'):
                try:
                    obj.reset(*args, **kwargs)
                except TypeError as e:
                    # If reset fails, create a new object instead
                    return self.factory_func(*args, **kwargs)
            return obj
        return self.factory_func(*args, **kwargs)
    
    def put(self, obj):
        """Return an object to the pool."""
        if len(self.pool) < self.max_size:
            self.pool.append(obj)

class VectorizedRandom:
    """Random number generation with intelligent caching and batch operations."""
    
    def __init__(self, seed=None):
        """Initialize the vectorized random generator."""
        import numpy.random as npr
        
        # Initialize the underlying numpy random generator
        self.rng = npr.default_rng(seed)
        
        # Cache settings
        self._batch_threshold = 1000
        self._cache_size = 10000
        
        # Initialize caches
        self._cached_uniform = np.array([])
        self._cached_normal = np.array([])
        self._uniform_idx = 0
        self._normal_idx = 0
        
        # Refill caches initially
        self._refill_uniform_cache()
        self._refill_normal_cache()
    
    def _refill_uniform_cache(self):
        """Refill the uniform random number cache."""
        self._cached_uniform = self.rng.uniform(0, 1, self._cache_size)
        self._uniform_idx = 0
    
    def _refill_normal_cache(self):
        """Refill the normal random number cache."""
        self._cached_normal = self.rng.normal(0, 1, self._cache_size)
        self._normal_idx = 0
    
    def uniform(self, low=0.0, high=1.0, size=None):
        """Get uniform random number(s) with intelligent caching."""
        if size is None:
            # Single value - use cache
            if self._uniform_idx >= len(self._cached_uniform):
                self._refill_uniform_cache()
            val = self._cached_uniform[self._uniform_idx] * (high - low) + low
            self._uniform_idx += 1
            return val
        elif size <= self._batch_threshold:
            # Small batch - use cache if possible
            if self._uniform_idx + size > len(self._cached_uniform):
                self._refill_uniform_cache()
            
            if size == 1:
                val = self._cached_uniform[self._uniform_idx] * (high - low) + low
                self._uniform_idx += 1
                return val
            else:
                # Extract batch from cache
                result = self._cached_uniform[self._uniform_idx:self._uniform_idx + size] * (high - low) + low
                self._uniform_idx += size
                return result
        else:
            # Large batch - direct generation
            return self.rng.uniform(low, high, size)
    
    def normal(self, loc=0, scale=1, size=None):
        """Get normal random number(s) with intelligent caching."""
        if size is None:
            # Single value - use cache
            if self._normal_idx >= len(self._cached_normal):
                self._refill_normal_cache()
            val = self._cached_normal[self._normal_idx] * scale + loc
            self._normal_idx += 1
            return val
        elif size <= self._batch_threshold:
            # Small batch - use cache if possible
            if self._normal_idx + size > len(self._cached_normal):
                self._refill_normal_cache()
            
            if size == 1:
                val = self._cached_normal[self._normal_idx] * scale + loc
                self._normal_idx += 1
                return val
            else:
                # Extract batch from cache
                result = self._cached_normal[self._normal_idx:self._normal_idx + size] * scale + loc
                self._normal_idx += size
                return result
        else:
            # Large batch - direct generation
            return self.rng.normal(loc, scale, size)
    
    def get_cache_stats(self):
        """Get cache utilization statistics."""
        return {
            'uniform_cache_used': self._uniform_idx,
            'uniform_cache_size': len(self._cached_uniform),
            'normal_cache_used': self._normal_idx,
            'normal_cache_size': len(self._cached_normal),
            'batch_threshold': self._batch_threshold
        }
    
    def get_generator(self):
        """Get the underlying numpy.random.Generator for advanced usage."""
        return self.rng
    
    def integers(self, low, high=None, size=None, dtype=np.int64):
        """Generate random integers with caching for small batches."""
        if size is None:
            return self.rng.integers(low, high, dtype=dtype)
        elif size <= self._batch_threshold:
            # For small batches, use direct generation (no caching for integers)
            return self.rng.integers(low, high, size=size, dtype=dtype)
        else:
            # Large batch - direct generation
            return self.rng.integers(low, high, size=size, dtype=dtype)
    
    def choice(self, a, size=None, replace=True, p=None):
        """Generate random choices with caching for small batches."""
        if size is None:
            return self.rng.choice(a, replace=replace, p=p)
        elif size <= self._batch_threshold:
            # For small batches, use direct generation (no caching for choices)
            return self.rng.choice(a, size=size, replace=replace, p=p)
        else:
            # Large batch - direct generation
            return self.rng.choice(a, size=size, replace=replace, p=p)

# Global vectorized random generator - will be initialized by set_random_seed
vrng = None

# using numba for just-in-time compilation of the following functions. When these functions are called, they're compiled in machine code specific to the input types, and subsequent calls are faster.

@njit 
def _triangle_sample_single(pzero, pmax, random_val):
    """Numba-optimized single triangle sampling."""
    return (pmax - pzero) * math.sqrt(random_val) + pzero

@njit
def _triangle_sample_batch(pzero_arr, pmax_arr, random_vals):
    """Numba-optimized batch triangle sampling."""
    return (pmax_arr - pzero_arr) * np.sqrt(random_vals) + pzero_arr

@njit
def _triangle_sample_batch_scalar(pzero, pmax, random_vals):
    """Numba-optimized batch triangle sampling with scalar parameters."""
    return (pmax - pzero) * np.sqrt(random_vals) + pzero

class Triangle:
    """Triangle distribution with batch operations and object pooling."""
    
    def __init__(self, pzero: float, pmax: float):
        self.pzero = pzero
        self.pmax = pmax
        self._validate_params()
    
    def _validate_params(self):
        """Validate distribution parameters."""
        if self.pzero < 0 or self.pmax < 0:
            raise ValueError("Triangle parameters must be non-negative")
    
    def reset(self, pzero: float, pmax: float):
        """Reset parameters for object pool reuse."""
        self.pzero = pzero
        self.pmax = pmax
        self._validate_params()
    
    def sample(self, size: Optional[int] = None) -> Union[float, np.ndarray]:
        """
        Sample from triangle distribution with optional batch generation.
        
        Args:
            size: If None, return single value. If int, return array of that size.
            
        Returns:
            Single float or numpy array of samples.
        """
        if size is None:
            # Single sample with cached random number
            return _triangle_sample_single(self.pzero, self.pmax, vrng.uniform())
        else:
            # CPU implementation only - GPU doesn't show speedup for this operation
            random_vals = vrng.uniform(size=size)
            return _triangle_sample_batch_scalar(self.pzero, self.pmax, random_vals)
    
    def __repr__(self):
        return f"Triangle(pzero={self.pzero}, pmax={self.pmax})"

class Flat:
    """Flat (uniform) distribution with batch operations."""
    
    def __init__(self):
        self.low = 0.0
        self.high = 1.0
    
    def reset(self, low: float = 0.0, high: float = 1.0):
        """Reset parameters for object pool reuse."""
        self.low = low
        self.high = high
    
    def sample(self, size: Optional[int] = None) -> Union[float, np.ndarray]:
        """
        Sample from uniform distribution.
        
        Args:
            size: If None, return single value. If int, return array of that size.
            
        Returns:
            Single float or numpy array of samples.
        """
        if size is None:
            return vrng.uniform() * (self.high - self.low) + self.low
        else:
            # CPU implementation only - GPU doesn't show speedup for this operation
            return vrng.uniform(size=size) * (self.high - self.low) + self.low
    
    def __repr__(self):
        return f"Flat(low={self.low}, high={self.high})"

class ProbabilitiesDontAddTo100(ValueError):
    """Exception for invalid probability tables."""
    pass

class ProbabilityTable:
    """
    Optimized probability table with batch operations and vectorized sampling.
    Supports both callable and non-callable values for efficiency.
    Includes GPU acceleration via CuPy when available.
    """
    
    def __init__(self, *args):
        self.table = []
        self.cumulative_probs = []
        self.values = []
        self._gpu_cumulative_probs = None
        self._gpu_values = None
        self._use_gpu = GPU_AVAILABLE
        self._setup_table(args)
    
    def _setup_table(self, args):
        """Setup internal probability table structure."""
        args = list(args)
        cumulative_prob = 0.0
        
        while len(args) > 0:
            prob = args.pop(0)
            value = args.pop(0)
            cumulative_prob += prob
            
            self.table.append((cumulative_prob, value))
            self.cumulative_probs.append(cumulative_prob)
            self.values.append(value)
        
        if len(args) != 0:
            raise ValueError("Invalid number of arguments")
        
        # TODO: should I make this epsilon instead of 1e-10?
        if abs(cumulative_prob - 1.0) > 1e-10:
            raise ProbabilitiesDontAddTo100(f"Probabilities sum to {cumulative_prob}, not 1.0")
        
        # Convert to numpy arrays for faster lookup
        self.cumulative_probs = np.array(self.cumulative_probs)
        
        # Setup GPU arrays if available
        if self._use_gpu:
            try:
                if GPU_FRAMEWORK == "cupy_cuda":
                    # CuPy implementation
                    self._gpu_cumulative_probs = cp.asarray(self.cumulative_probs)
                    # Only move numeric values to GPU (callable values stay on CPU)
                    numeric_values = []
                    for value in self.values:
                        if not callable(value) and isinstance(value, (int, float, np.number)):
                            numeric_values.append(value)
                        else:
                            numeric_values.append(None)  # Placeholder for non-numeric values
                    
                    if any(v is not None for v in numeric_values):
                        self._gpu_values = cp.array(numeric_values)
                    else:
                        self._gpu_values = None
                        
                elif GPU_FRAMEWORK in ["pytorch_cuda", "pytorch_rocm"]:
                    # PyTorch implementation
                    import torch
                    self._gpu_cumulative_probs = torch.tensor(self.cumulative_probs, device='cuda' if GPU_FRAMEWORK == "pytorch_cuda" else 'cuda', dtype=torch.float32)
                    # Only move numeric values to GPU (callable values stay on CPU)
                    numeric_values = []
                    for value in self.values:
                        if not callable(value) and isinstance(value, (int, float, np.number)):
                            numeric_values.append(value)
                        else:
                            numeric_values.append(None)  # Placeholder for non-numeric values
                    
                    if any(v is not None for v in numeric_values):
                        # Use the same data type as the original values
                        if all(isinstance(v, int) for v in numeric_values if v is not None):
                            dtype = torch.int32
                        else:
                            dtype = torch.float32
                        self._gpu_values = torch.tensor(numeric_values, device='cuda' if GPU_FRAMEWORK == "pytorch_cuda" else 'cuda', dtype=dtype)
                    else:
                        self._gpu_values = None
                else:
                    self._gpu_cumulative_probs = None
                    self._gpu_values = None
                    
            except Exception as e:
                # Fallback to CPU if GPU setup fails
                self._use_gpu = False
                self._gpu_cumulative_probs = None
                self._gpu_values = None
    
    def reset(self, *args):
        """Reset table for object pool reuse."""
        self.table.clear()
        self.cumulative_probs = []
        self.values = []
        self._gpu_cumulative_probs = None
        self._gpu_values = None
        self._setup_table(args)
    
    def _generate_gpu_batch(self, size: int, keep_on_gpu: bool = False):
        """
        GPU-accelerated batch generation for numeric values.
        
        Args:
            size: Number of values to generate
            keep_on_gpu: If True, return GPU tensor/array instead of CPU numpy array
        """
        if not self._use_gpu or self._gpu_values is None:
            return None
        
        if GPU_FRAMEWORK == "cupy_cuda":
            # CuPy implementation - use our vrng for consistency
            random_vals = vrng.uniform(size=size)
            rnds = cp.asarray(random_vals)
            
            # Use GPU searchsorted for faster lookup
            indices = cp.searchsorted(self._gpu_cumulative_probs, rnds, side='right')
            indices = cp.clip(indices, 0, len(self._gpu_values) - 1)
            
            # Vectorized gather operation
            results = cp.take(self._gpu_values, indices)
            
            # Only transfer to CPU if explicitly requested
            if keep_on_gpu:
                return results
            else:
                return cp.asnumpy(results)
            
        elif GPU_FRAMEWORK in ["pytorch_cuda", "pytorch_rocm"]:
            # PyTorch implementation - use our vrng for consistency
            import torch
            
            # Use our vrng system for consistency
            random_vals = vrng.uniform(size=size)
            rnds = torch.tensor(random_vals, device=self._gpu_values.device, dtype=torch.float32)
            
            # Use GPU searchsorted for faster lookup
            indices = torch.searchsorted(self._gpu_cumulative_probs, rnds, right=True)
            indices = torch.clamp(indices, 0, len(self._gpu_values) - 1)
            
            # Vectorized gather operation - indices need to be properly shaped
            # For 1D tensors, we need to unsqueeze indices to match the gather operation
            indices = indices.unsqueeze(0)  # Shape: (1, size)
            results = torch.gather(self._gpu_values.unsqueeze(0), 1, indices)  # Shape: (1, size)
            results = results.squeeze(0)  # Shape: (size,)
            
            # Only transfer to CPU if explicitly requested
            if keep_on_gpu:
                return results
            else:
                return results.cpu().numpy()
        
        return None
    
    def generate(self, size: Optional[int] = None, keep_on_gpu: bool = False):
        """
        Generate value(s) from probability table.
        
        Args:
            size: If None, return single value. If int, return list of values.
            keep_on_gpu: If True and using GPU, return GPU tensor/array instead of CPU list
            
        Returns:
            Single value or list of values based on probability distribution.
        """
        if size is None:
            # Single generation
            rnd = vrng.uniform()
            idx = np.searchsorted(self.cumulative_probs, rnd, side='right')
            if idx >= len(self.values):
                idx = len(self.values) - 1
            
            value = self.values[idx]
            return value() if callable(value) else value
        else:
            # Check if we can use GPU acceleration for numeric values
            if self._use_gpu and self._gpu_values is not None:
                # Check if all values are numeric (no callables)
                all_numeric = all(not callable(v) and isinstance(v, (int, float, np.number)) 
                                for v in self.values)
                
                if all_numeric:
                    # Use GPU acceleration
                    gpu_results = self._generate_gpu_batch(size, keep_on_gpu=keep_on_gpu)
                    if keep_on_gpu:
                        return gpu_results
                    else:
                        return gpu_results.tolist()
            
            # Fallback to CPU implementation
            rnds = vrng.uniform(size=size)
            indices = np.searchsorted(self.cumulative_probs, rnds, side='right')
            indices = np.clip(indices, 0, len(self.values) - 1)
            
            # Vectorized gather operation using numpy.take
            # First, check if all values are numeric (no callables)
            all_numeric = all(not callable(v) and isinstance(v, (int, float, np.number)) 
                            for v in self.values)
            
            if all_numeric:
                # Convert values to numpy array for vectorized operation
                values_array = np.array(self.values)
                results = np.take(values_array, indices)
                return results.tolist()
            else:
                # For callable values, we still need to process individually
                # but we can vectorize the index lookup part
                results = []
                for idx in indices:
                    value = self.values[idx]
                    results.append(value() if callable(value) else value)
                return results

    def generate_gpu_batch(self, size: int):
        """
        Generate a batch of values and keep them on GPU for further processing.
        This avoids the host-device transfer overhead.
        
        Args:
            size: Number of values to generate
            
        Returns:
            GPU tensor/array (CuPy array or PyTorch tensor)
        """
        return self.generate(size=size, keep_on_gpu=True)
    
    def __repr__(self):
        result = "ProbabilityTable("
        for prob, value in self.table:
            value_repr = 'lambda?' if callable(value) else repr(value)
            result += f"{prob},{value_repr},"
        result = result[:-1] + ")"
        return result

class GPUBatchProcessor:
    """
    Utility class for accumulating GPU operations to minimize host-device transfers.
    
    Example usage:
        # Create a probability table
        pt = ProbabilityTable(0.3, 1, 0.7, 2)
        
        # Create batch processor
        processor = GPUBatchProcessor(pt)
        
        # Add multiple batches (stays on GPU)
        processor.add_batch(1000)
        processor.add_batch(2000)
        processor.add_batch(1500)
        
        # Get all results at once (single transfer)
        results = processor.get_all_results()
        
        # Or keep on GPU for further processing
        gpu_results = processor.get_all_results(transfer_to_cpu=False)
    """
    
    def __init__(self, probability_table):
        self.prob_table = probability_table
        self.gpu_results = []
        self.batch_sizes = []
        
    def add_batch(self, size: int):
        """Add a batch generation to the queue (stays on GPU)."""
        if not self.prob_table._use_gpu:
            raise ValueError("Probability table is not using GPU")
        
        gpu_result = self.prob_table.generate_gpu_batch(size)
        self.gpu_results.append(gpu_result)
        self.batch_sizes.append(size)
        
    def get_all_results(self, transfer_to_cpu: bool = True):
        """
        Get all accumulated results, optionally transferring to CPU.
        
        Args:
            transfer_to_cpu: If True, transfer all results to CPU and return as lists
            
        Returns:
            List of results (GPU tensors/arrays or CPU lists)
        """
        if transfer_to_cpu:
            cpu_results = []
            for gpu_result in self.gpu_results:
                if GPU_FRAMEWORK == "cupy_cuda":
                    cpu_results.append(cp.asnumpy(gpu_result).tolist())
                elif GPU_FRAMEWORK in ["pytorch_cuda", "pytorch_rocm"]:
                    cpu_results.append(gpu_result.cpu().numpy().tolist())
            return cpu_results
        else:
            return self.gpu_results.copy()
    
    def clear(self):
        """Clear accumulated results."""
        self.gpu_results.clear()
        self.batch_sizes.clear()

# Factory functions for object pools
def create_triangle():
    return Triangle(0, 1)

def create_flat():
    return Flat()

def create_probability_table():
    return ProbabilityTable(1.0, 0)

# Global object pools
triangle_pool = ObjectPool(create_triangle, max_size=100)
flat_pool = ObjectPool(create_flat, max_size=100)
probability_table_pool = ObjectPool(create_probability_table, max_size=100)

# Utility functions for optimized operations
@njit
def fast_searchsorted(a, v):
    """Numba-optimized binary search for single value."""
    return np.searchsorted(a, v)

@njit
def batch_fitness_calculation(fitness_array, mutation_effects):
    """Vectorized fitness calculation."""
    return np.maximum(0.0, fitness_array + mutation_effects)

def batch_random_bool(size: int, probability: float) -> np.ndarray:
    """Generate batch of boolean values based on probability."""
    # CPU implementation only - GPU doesn't show speedup for this operation
    return vrng.uniform(size=size) < probability

def optimize_memory_usage():
    """Optimize memory usage by clearing caches and pools."""
    global vrng
    vrng._cached_uniform = np.array([])
    vrng._cached_normal = np.array([])
    vrng._uniform_idx = 0
    vrng._normal_idx = 0
    
    # Clear object pools
    triangle_pool.pool.clear()
    flat_pool.pool.clear()
    probability_table_pool.pool.clear()

def set_random_seed(seed: Optional[int] = None):
    """Set global random seed for reproducibility."""
    global vrng
    
    # If no seed provided, generate a random one for consistency
    if seed is None:
        import time
        seed = int(time.time() * 1000) % (2**32)  # Use timestamp-based seed
    
    # Log the seed being used
    logger.info(f"Setting random seed: {seed}")
    
    # Set our vectorized random generator (uses modern default_rng internally)
    vrng = VectorizedRandom(seed)
    
    # Set GPU seed if available
    if GPU_AVAILABLE:
        try:
            if GPU_FRAMEWORK == "cupy_cuda":
                cp.random.seed(seed)
                logger.info(f"Set CuPy GPU seed: {seed}")
            elif GPU_FRAMEWORK in ["pytorch_cuda", "pytorch_rocm"]:
                import torch
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    logger.info(f"Set PyTorch GPU seed: {seed}")
                else:
                    logger.info(f"Set PyTorch CPU seed: {seed}")
        except Exception as e:
            logger.warning(f"Failed to set GPU seed: {e}")
    else:
        logger.info("GPU not available - using CPU-only seeding")
    
    logger.info(f"Random seed {seed} set successfully for all components")

def is_gpu_available():
    """Check if GPU acceleration is available."""
    return GPU_AVAILABLE

def get_gpu_framework():
    """Get the detected GPU framework."""
    return GPU_FRAMEWORK

def get_gpu_info():
    """Get detailed GPU information."""
    info = {
        "available": GPU_AVAILABLE,
        "framework": GPU_FRAMEWORK
    }
    
    if GPU_AVAILABLE:
        try:
            if GPU_FRAMEWORK == "cupy_cuda":
                info["device_count"] = cp.cuda.runtime.getDeviceCount()
                info["device_name"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
            elif GPU_FRAMEWORK in ["pytorch_cuda", "pytorch_rocm"]:
                import torch
                info["device_count"] = torch.cuda.device_count()
                info["device_name"] = torch.cuda.get_device_name(0)
        except:
            pass
    
    return info

# In case users also want to benchmark for themselves, either on their personal system or on a server.
def benchmark_sampling(distribution, n_samples=100000):
    """Benchmark sampling performance."""
    import time
    
    # Single sampling
    start = time.time()
    for _ in range(1000):
        distribution.sample()
    single_time = time.time() - start
    
    # Batch sampling
    start = time.time()
    distribution.sample(size=n_samples)
    batch_time = time.time() - start
    
    print(f"Single sampling (1000 calls): {single_time:.4f}s")
    print(f"Batch sampling ({n_samples} samples): {batch_time:.4f}s")
    print(f"Speedup factor: {(single_time * n_samples / 1000) / batch_time:.2f}x")
    
    # GPU info if available
    if GPU_AVAILABLE:
        print(f"GPU acceleration: Available (CuPy)")
    else:
        print(f"GPU acceleration: Not available")

def benchmark_gpu_vs_cpu():
    """Benchmark GPU vs CPU performance for probability table."""
    if not GPU_AVAILABLE:
        print("GPU not available for benchmarking")
        return
    
    import time
    
    # Create a probability table with numeric values
    pt = ProbabilityTable(0.3, 1, 0.4, 2, 0.3, 3)
    
    sizes = [1000, 10000, 100000, 1000000]
    
    print("GPU vs CPU Benchmark for ProbabilityTable.generate:")
    print("=" * 50)
    
    for size in sizes:
        # CPU timing
        start = time.time()
        cpu_results = pt.generate(size=size)
        cpu_time = time.time() - start
        
        # GPU timing (force GPU by temporarily disabling CPU fallback)
        original_use_gpu = pt._use_gpu
        pt._use_gpu = True
        start = time.time()
        gpu_results = pt.generate(size=size)
        gpu_time = time.time() - start
        pt._use_gpu = original_use_gpu
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        
        print(f"Size {size:8d}: CPU={cpu_time:.4f}s, GPU={gpu_time:.4f}s, Speedup={speedup:.2f}x")

def benchmark_all_gpu_operations():
    """Comprehensive benchmark of GPU-accelerated operations."""
    if not GPU_AVAILABLE:
        print("GPU not available for comprehensive benchmarking")
        return
    
    import time
    
    print("GPU vs CPU Benchmark (ProbabilityTable Only)")
    print("=" * 50)
    print(f"GPU Framework: {get_gpu_framework()}")
    print()
    
    # Test sizes
    sizes = [1000, 10000, 100000, 500000]
    
    # Test probability table (only operation with significant GPU speedup)
    pt = ProbabilityTable(0.2, 1, 0.3, 2, 0.3, 3, 0.2, 4)
    
    print("Probability Table Generation:")
    print("-" * 30)
    for size in sizes:
        # CPU timing
        pt._use_gpu = False
        start = time.time()
        cpu_results = pt.generate(size=size)
        cpu_time = time.time() - start
        
        # GPU timing
        pt._use_gpu = True
        start = time.time()
        gpu_results = pt.generate(size=size)
        gpu_time = time.time() - start
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        
        print(f"Size {size:8d}: CPU={cpu_time:.4f}s, GPU={gpu_time:.4f}s, Speedup={speedup:.2f}x")

# Memory-efficient bit operations for large genomes

@njit
def _popcount_64bit_simple(word: np.uint64) -> np.uint32:
    """Simple and efficient 64-bit popcount using bit manipulation."""
    count = 0
    while word:
        count += word & 1
        word >>= 1
    return count

@njit
def _count_set_bits_array(data: np.ndarray) -> np.uint32:
    """Count set bits in entire array using simple method."""
    total = 0
    for word in data:
        total += _popcount_64bit_simple(word)
    return total

class BitArray:
    """Memory-efficient bit array for marking genome positions."""
    
    def __init__(self, size: int):
        self.size = size
        self.data = np.zeros((size + 63) // 64, dtype=np.uint64)
    
    def set_bit(self, pos: int):
        """Set bit at position."""
        if pos < 0 or pos >= self.size:
            return  # Out of bounds, ignore
        word_idx = pos // 64
        bit_idx = pos % 64
        if word_idx < len(self.data):
            self.data[word_idx] |= np.uint64(1 << bit_idx)
    
    def clear_bit(self, pos: int):
        """Clear bit at position."""
        if pos < 0 or pos >= self.size:
            return  # Out of bounds, ignore
        word_idx = pos // 64
        bit_idx = pos % 64
        if word_idx < len(self.data):
            self.data[word_idx] &= ~np.uint64(1 << bit_idx)
    
    def get_bit(self, pos: int) -> bool:
        """Get bit at position."""
        if pos < 0 or pos >= self.size:
            return False  # Out of bounds, return False
        word_idx = pos // 64
        bit_idx = pos % 64
        if word_idx < len(self.data):
            return bool(self.data[word_idx] & np.uint64(1 << bit_idx))
        return False
    
    def count_set_bits(self) -> int:
        """Count total number of set bits using optimized algorithm."""
        return int(_count_set_bits_array(self.data))
    
    def count_set_bits_old(self) -> int:
        """Original slow implementation for comparison."""
        return int(np.sum([bin(word).count('1') for word in self.data]))

def estimate_gpu_transfer_cost(size: int, dtype=np.float32) -> float:
    """
    Estimate the time cost of GPU→CPU transfer for a given array size.
    
    Args:
        size: Number of elements in the array
        dtype: Data type (affects memory size)
        
    Returns:
        Estimated transfer time in milliseconds
    """
    # Typical PCIe bandwidth: ~12-16 GB/s for PCIe 3.0, ~32 GB/s for PCIe 4.0
    # We'll use a conservative estimate of 10 GB/s
    bandwidth_gbps = 10.0
    
    # Calculate memory size in bytes
    if dtype == np.float32:
        bytes_per_element = 4
    elif dtype == np.float64:
        bytes_per_element = 8
    elif dtype == np.int32:
        bytes_per_element = 4
    elif dtype == np.int64:
        bytes_per_element = 8
    else:
        bytes_per_element = 4  # Default assumption
    
    total_bytes = size * bytes_per_element
    transfer_time_ms = (total_bytes / (bandwidth_gbps * 1e9)) * 1000
    
    return transfer_time_ms

def should_use_gpu_for_batch(size: int, num_operations: int = 1) -> bool:
    """
    Determine if GPU acceleration is worth it for a given batch size.
    
    Args:
        size: Number of elements to generate
        num_operations: Number of operations to perform (for batch processing)
        
    Returns:
        True if GPU is recommended, False if CPU is better
    """
    if not is_gpu_available():
        return False
    
    # Estimate transfer cost
    transfer_cost_ms = estimate_gpu_transfer_cost(size) * num_operations
    
    # Rough estimate of GPU computation time (very approximate)
    # GPU computation is typically much faster than transfer for large batches
    gpu_compute_time_ms = size / 1e6  # Rough estimate: 1M elements per ms
    
    # CPU computation time estimate
    cpu_compute_time_ms = size / 1e5  # Rough estimate: 100K elements per ms
    
    # Total GPU time = compute + transfer
    total_gpu_time = gpu_compute_time_ms + transfer_cost_ms
    
    # Use GPU if it's faster
    return total_gpu_time < cpu_compute_time_ms

# Export commonly used functions
__all__ = [
    'Triangle', 'Flat', 'ProbabilityTable', 'VectorizedRandom',
    'ObjectPool', 'BitArray', 'vrng', 'triangle_pool', 'flat_pool', 
    'probability_table_pool', 'batch_random_bool', 'optimize_memory_usage',
    'set_random_seed', 'benchmark_sampling', 'benchmark_gpu_vs_cpu', 'benchmark_all_gpu_operations',
    'is_gpu_available', 'GPU_AVAILABLE', 'get_gpu_framework', 'get_gpu_info'
] 