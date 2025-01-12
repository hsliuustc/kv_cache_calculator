from typing import Union
from .models import DenseModel, MoEModel, ModelVariant

def calculate_kv_cache_size(
    model_variant: ModelVariant,
    sequence_length: int = 2048,
    batch_size: int = 1,
    precision_bits: int = 16
) -> float:
    """Calculate the KV cache size in GB for a transformer model.
    
    Args:
        model_variant: Model variant configuration
        sequence_length: Maximum sequence length
        batch_size: Number of sequences processed in parallel
        precision_bits: Number of bits used for precision (16 or 32)
        
    Returns:
        KV cache size in gigabytes
        
    Raises:
        ValueError: If precision_bits is invalid or parameters are invalid
    """
    if precision_bits not in (16, 32):
        raise ValueError("precision_bits must be 16 or 32")
    if sequence_length <= 0 or batch_size <= 0:
        raise ValueError("sequence_length and batch_size must be positive integers")
    
    # Calculate size per layer
    bytes_per_element = precision_bits / 8
    size_per_layer = (
        model_variant.hidden_size * sequence_length * 2 *  # 2 for K and V
        model_variant.num_attention_heads * bytes_per_element
    )
    
    # For MoE models, multiply by number of active experts (default 2)
    if hasattr(model_variant, 'num_experts') and model_variant.num_experts:
        active_experts = getattr(model_variant, 'active_experts', 2)
        size_per_layer *= active_experts
    
    # Total size across all layers
    total_size = size_per_layer * model_variant.num_layers * batch_size
    
    # Convert to GB
    return total_size / (1024 ** 3)

def calculate_model_kv_cache(
    model: Union[DenseModel, MoEModel],
    variant_name: str,
    sequence_length: int = 2048,
    batch_size: int = 1,
    precision_bits: int = 16
) -> float:
    """Calculate KV cache size for a specific model variant.
    
    Args:
        model: Model instance (DenseModel or MoEModel)
        variant_name: Name of the model variant
        sequence_length: Maximum sequence length
        batch_size: Number of sequences processed in parallel
        precision_bits: Number of bits used for precision (16 or 32)
        
    Returns:
        KV cache size in gigabytes
        
    Raises:
        ValueError: If variant not found or parameters are invalid
    """
    variant = model.get_variant(variant_name)
    if not variant:
        raise ValueError(f"Variant {variant_name} not found in model {model.name}")
    
    return calculate_kv_cache_size(
        variant,
        sequence_length=sequence_length,
        batch_size=batch_size,
        precision_bits=precision_bits
    )
