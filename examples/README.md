# KV Cache Calculator Examples

This directory contains example configurations and usage scenarios for the KV Cache Calculator.

## Example Files

1. **basic_usage.py** - Demonstrates basic usage of the calculator
2. **llama2_config.yaml** - Example configuration for LLaMA 2 models
3. **gpt3_config.yaml** - Example configuration for GPT-3 models
4. **custom_model.py** - Shows how to create custom model configurations

## Running Examples

To run an example:

```bash
python examples/basic_usage.py
```

Or use the CLI with example configs:

```bash
python -m kv_cache_calculator.cli --config examples/llama2_config.yaml
python -m kv_cache_calculator.cli --config examples/gpt3_config.yaml
```

## Contributing Examples

To contribute new examples:
1. Add your example file to this directory
2. Include clear documentation in the file
3. Update this README with a brief description
4. Ensure the example follows the project's coding standards
