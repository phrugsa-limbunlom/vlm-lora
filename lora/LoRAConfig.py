from typing import Optional, Dict, Any


class LoRAConfig:
    """Configuration class for LoRA parameters."""

    def __init__(
            self,
            r: int = 4,
            lora_alpha: float = 32.0,
            lora_dropout: float = 0.1,
            target_modules: Optional[list] = None,
            merge_weights: bool = False,
    ):
        """
        Initialize LoRA configuration.

        Args:
            r: Rank of LoRA decomposition
            lora_alpha: LoRA scaling factor
            lora_dropout: Dropout probability
            target_modules: List of module names to apply LoRA to
            merge_weights: Whether to merge weights
        """
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"]
        self.merge_weights = merge_weights

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "merge_weights": self.merge_weights,
        }