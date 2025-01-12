import argparse
from . import calculator
from . import models

def main():
    parser = argparse.ArgumentParser(description="KV Cache Size Estimator")
    parser.add_argument("sequence_length", type=int, 
                       help="Input sequence length")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size (default: 1)")
    parser.add_argument("--precision", type=int, default=2, choices=[2, 4],
                       help="Precision size: 2 for float16, 4 for float32 (default: 2)")
    
    args = parser.parse_args()

    print("KV Cache Size Estimator")
    print("=======================")
    print(f"Sequence Length: {args.sequence_length}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Precision: {'float16' if args.precision == 2 else 'float32'}\n")
    
    for model_name in models.MODELS:
        config = models.get_model_config(model_name)
        size = calculator.calculate_kv_cache_size(
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            num_layers=config["num_layers"],
            hidden_size=config["hidden_size"],
            precision_size=args.precision
        )
        size_gb = size / (1024 ** 3)
        print(f"{model_name}: {size_gb:.2f} GB")
