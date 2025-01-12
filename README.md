# KV Cache Calculator

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/yourusername/kv_cache_calculator/actions/workflows/tests.yml/badge.svg)](https://github.com/yourusername/kv_cache_calculator/actions)

A Python package for calculating Key-Value (KV) cache sizes for transformer-based language models, including both dense and mixture-of-experts (MoE) architectures.

## Features

- 🧮 Calculate KV cache size for dense transformer models
- 🎛️ Support for mixture-of-experts (MoE) architectures
- ⚙️ Configurable parameters:
  - Number of layers
  - Hidden size
  - Number of attention heads
  - Number of experts (for MoE)
  - Active experts per token (for MoE)
  - Sequence length
  - Batch size
  - Precision (16-bit or 32-bit)
- 📊 Built-in support for popular model architectures
- ✅ Comprehensive test coverage

## Installation

```bash
pip install kv-cache-calculator
```

## Usage

### Basic Usage

```python
from kv_cache_calculator import calculate_kv_cache_size
from kv_cache_calculator.models import ModelVariant

# Define model parameters
model_variant = ModelVariant(
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    num_experts=8,  # For MoE models
    active_experts=2  # For MoE models
)

# Calculate KV cache size
cache_size = calculate_kv_cache_size(
    model_variant,
    sequence_length=2048,
    batch_size=1,
    precision_bits=16
)

print(f"KV Cache Size: {cache_size} GB")
```

### CLI Usage

```bash
kv-cache-calc --num-layers 32 --hidden-size 4096 --num-heads 32 --seq-len 2048
```

## Supported Models

The calculator supports calculations for:

- Dense Transformer Models
- Mixture-of-Experts (MoE) Models
- Sparse Transformer Models

## Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Roadmap

- [ ] Add support for more model architectures
- [ ] Create visualization tools
- [ ] Add benchmarking capabilities
- [ ] Develop web interface
