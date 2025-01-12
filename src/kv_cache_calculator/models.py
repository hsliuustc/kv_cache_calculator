from dataclasses import dataclass
from typing import List, Dict, Optional
import yaml
from pathlib import Path

@dataclass
class ModelVariant:
    name: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_experts: Optional[int] = None  # Only for MoE models
    expert_capacity: Optional[int] = None  # Only for MoE models
    active_experts: Optional[int] = 2  # Number of active experts per token, default 2

class BaseModel:
    def __init__(self, name: str, description: str, variants: List[ModelVariant]):
        self.name = name
        self.description = description
        self.variants = variants

    def get_variant(self, variant_name: str) -> Optional[ModelVariant]:
        return next((v for v in self.variants if v.name == variant_name), None)

class DenseModel(BaseModel):
    """Standard dense transformer model"""
    pass

class MoEModel(BaseModel):
    """Mixture of Experts model"""
    def __init__(self, name: str, description: str, variants: List[ModelVariant]):
        super().__init__(name, description, variants)
        # Validate MoE-specific parameters
        for variant in self.variants:
            if variant.num_experts is None:
                raise ValueError(f"MoE model {name} variant {variant.name} missing num_experts")
            if variant.expert_capacity is None:
                raise ValueError(f"MoE model {name} variant {variant.name} is missing the 'expert_capacity' parameter")

# Initialize with empty models
MODELS: Dict[str, BaseModel] = {}

def create_model_from_config(model_config: dict) -> BaseModel:
    """Create appropriate model instance from config"""
    variants = [
        ModelVariant(
            name=v["name"],
            num_layers=v["num_layers"],
            hidden_size=v["hidden_size"],
            num_attention_heads=v["num_attention_heads"],
            num_experts=v.get("num_experts"),
            expert_capacity=v.get("expert_capacity")
        )
        for v in model_config["variants"]
    ]
    
    if any(v.get("num_experts") for v in model_config["variants"]):
        return MoEModel(
            name=model_config["name"],
            description=model_config["description"],
            variants=variants
        )
    return DenseModel(
        name=model_config["name"],
        description=model_config["description"],
        variants=variants
    )

# Load configuration on import
config_path = Path(__file__).parent.parent.parent / "config" / "models.yaml"
if config_path.exists():
    with open(config_path, 'r') as f:
        MODELS = {
            model["name"]: create_model_from_config(model)
            for model in yaml.safe_load(f)["models"]
        }

def get_model_config(model_name: str) -> Optional[BaseModel]:
    """Get configuration for a specific model"""
    return MODELS.get(model_name)

def list_available_models() -> List[dict]:
    """List all available models with their variants"""
    return [
        {
            "name": model.name,
            "description": model.description,
            "variants": [
                {
                    "name": v.name,
                    "num_layers": v.num_layers,
                    "hidden_size": v.hidden_size,
                    "num_attention_heads": v.num_attention_heads,
                    "num_experts": v.num_experts,
                    "expert_capacity": v.expert_capacity
                }
                for v in model.variants
            ]
        }
        for model in MODELS.values()
    ]

def load_models_from_config(config_path: Path):
    """Load models from YAML configuration file"""
    global MODELS
    if config_path.exists():
        with open(config_path, 'r') as f:
            MODELS.update({
                model["name"]: create_model_from_config(model)
                for model in yaml.safe_load(f)["models"]
            })
