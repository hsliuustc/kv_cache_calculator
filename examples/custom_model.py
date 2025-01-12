"""
Custom Model Configuration Example

This example demonstrates how to:
1. Create custom model configurations
2. Calculate KV cache sizes for custom models
3. Compare different configurations
"""

from kv_cache_calculator.calculator import calculate_kv_cache_size
from kv_cache_calculator.models import ModelConfig

# Create custom configurations
custom_config_1 = ModelConfig(
    num_layers=48,
    num_heads=24,
    hidden_size=6144,
    seq_length=4096,
    dtype_bytes=2  # FP16
)

custom_config_2 = ModelConfig(
    num_layers=64,
    num_heads=32,
    hidden_size=8192,
    seq_length=8192,
    dtype_bytes=4  # FP32
)

# Calculate and compare cache sizes
cache_size_1 = calculate_kv_cache_size(custom_config_1)
cache_size_2 = calculate_kv_cache_size(custom_config_2)

print(f"Custom Model 1 KV Cache Size: {cache_size_1 / (1024 ** 3):.2f} GB")
print(f"Custom Model 2 KV Cache Size: {cache_size_2 / (1024 ** 3):.2f} GB")
print(f"Size Difference: {(cache_size_2 - cache_size_1) / (1024 ** 3):.2f} GB")
