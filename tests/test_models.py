import pytest
from kv_cache_calculator.models import get_model_config, list_available_models

def test_get_model_config():
    # Test valid model variant
    gpt3_config = get_model_config("GPT-3")
    assert gpt3_config is not None
    assert len(gpt3_config.variants) > 0
    
    # Test 175B variant
    variant_175b = next(v for v in gpt3_config.variants if v.name == "175B")
    assert variant_175b.num_layers == 96
    assert variant_175b.hidden_size == 12288

    # Test invalid model
    assert get_model_config("InvalidModel") is None

def test_list_available_models():
    models = list_available_models()
    assert len(models) > 0
    
    # Check GPT-3 details
    gpt3 = next(m for m in models if m["name"] == "GPT-3")
    assert gpt3["description"] == "OpenAI's GPT-3 model"
    assert len(gpt3["variants"]) == 2
    assert gpt3["variants"][0]["name"] == "175B"
    assert gpt3["variants"][1]["name"] == "6.7B"

    # Check Qwen details
    qwen = next(m for m in models if m["name"] == "Qwen")
    assert qwen["description"] == "Qwen models by Alibaba Cloud"
    assert len(qwen["variants"]) == 3
    assert set(v["name"] for v in qwen["variants"]) == {"7B", "14B", "72B"}

    # Check Mixtral details
    mixtral = next(m for m in models if m["name"] == "Mixtral")
    assert mixtral["description"] == "Mixture of Experts model by Mistral AI"
    assert len(mixtral["variants"]) == 2
    assert set(v["name"] for v in mixtral["variants"]) == {"8x7B", "8x22B"}

    # Check LLaMA details
    llama = next(m for m in models if m["name"] == "LLaMA")
    assert llama["description"] == "Meta's LLaMA models"
    assert len(llama["variants"]) == 3
    assert set(v["name"] for v in llama["variants"]) == {"7B", "13B", "70B"}

def test_model_variant_parameters():
    # Test Qwen 72B parameters
    qwen_config = get_model_config("Qwen")
    variant_72b = next(v for v in qwen_config.variants if v.name == "72B")
    assert variant_72b.num_layers == 80
    assert variant_72b.hidden_size == 8192

    # Test Mixtral 8x7B parameters
    mixtral_config = get_model_config("Mixtral")
    variant_8x7b = next(v for v in mixtral_config.variants if v.name == "8x7B")
    assert variant_8x7b.num_layers == 32
    assert variant_8x7b.hidden_size == 4096

    # Test LLaMA 70B parameters
    llama_config = get_model_config("LLaMA")
    variant_70b = next(v for v in llama_config.variants if v.name == "70B")
    assert variant_70b.num_layers == 80
    assert variant_70b.hidden_size == 8192
