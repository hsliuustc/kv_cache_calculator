import pytest
from kv_cache_calculator.calculator import calculate_kv_cache_size, calculate_model_kv_cache
from kv_cache_calculator.models import DenseModel, MoEModel, ModelVariant

@pytest.fixture
def dense_variant():
    return ModelVariant(
        name="test",
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32
    )

@pytest.fixture
def moe_variant():
    return ModelVariant(
        name="test-moe",
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_experts=8,
        expert_capacity=32,
        active_experts=2
    )

def test_dense_model_calculation(dense_variant):
    # Test basic calculation
    size = calculate_kv_cache_size(dense_variant, sequence_length=2048)
    assert pytest.approx(size, 0.01) == 32.0  # Expected value for these parameters
    
    # Test batch size
    size_batch = calculate_kv_cache_size(dense_variant, sequence_length=2048, batch_size=2)
    assert size_batch == size * 2
    
    # Test precision
    size_fp32 = calculate_kv_cache_size(dense_variant, sequence_length=2048, precision_bits=32)
    assert size_fp32 == size * 2

def test_moe_model_calculation(moe_variant):
    # Test basic MoE calculation
    size = calculate_kv_cache_size(moe_variant, sequence_length=2048)
    assert pytest.approx(size, 0.01) == 64.0  # 2x larger than dense model (2 active experts)
    
    # Test batch size
    size_batch = calculate_kv_cache_size(moe_variant, sequence_length=2048, batch_size=2)
    assert size_batch == size * 2
    
    # Test precision
    size_fp32 = calculate_kv_cache_size(moe_variant, sequence_length=2048, precision_bits=32)
    assert size_fp32 == size * 2

def test_model_based_calculation():
    # Create test models
    dense_model = DenseModel(
        name="test-dense",
        description="Test dense model",
        variants=[ModelVariant(
            name="test",
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32
        )]
    )
    
    moe_model = MoEModel(
        name="test-moe",
        description="Test MoE model",
        variants=[ModelVariant(
            name="test-moe",
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_experts=8,
            expert_capacity=32,
            active_experts=2
        )]
    )
    
    # Test dense model calculation
    dense_size = calculate_model_kv_cache(dense_model, "test", sequence_length=2048)
    assert pytest.approx(dense_size, 0.01) == 32.0
    
    # Test MoE model calculation
    moe_size = calculate_model_kv_cache(moe_model, "test-moe", sequence_length=2048)
    assert pytest.approx(moe_size, 0.01) == 64.0  # 2x larger than dense model (2 active experts)
    
    # Test invalid variant
    with pytest.raises(ValueError):
        calculate_model_kv_cache(dense_model, "invalid", sequence_length=2048)

def test_invalid_parameters():
    variant = ModelVariant(
        name="test",
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32
    )
    
    # Test invalid precision
    with pytest.raises(ValueError):
        calculate_kv_cache_size(variant, precision_bits=64)
        
    # Test invalid sequence length
    with pytest.raises(ValueError):
        calculate_kv_cache_size(variant, sequence_length=-1)
        
    # Test invalid batch size
    with pytest.raises(ValueError):
        calculate_kv_cache_size(variant, batch_size=0)
