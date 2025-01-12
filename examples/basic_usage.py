"""
Basic Usage Example for KV Cache Calculator

This example demonstrates how to:
1. Load model configurations
2. Calculate KV cache sizes
3. Print results
"""

from kv_cache_calculator.calculator import calculate_kv_cache_size
from kv_cache_calculator.models import ModelConfig

# Define a basic model configuration
basic_config = ModelConfig(
    num_layers=32,
    num_heads=32,
    hidden_size=4096,
    seq_length=2048,
    dtype_bytes=2  # FP16
)

# Calculate KV cache size
cache_size = calculate_kv_cache_size(basic_config)

# Print results
print(f"KV Cache Size: {cache_size / (1024 ** 2):.2f} MB")
