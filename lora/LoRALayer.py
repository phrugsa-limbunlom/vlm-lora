import math

from torch import nn


class LoRALayer(nn.Module):
    """
    Base LoRA layer that implements low-rank adaptation.

    LoRA decomposes the weight update ΔW into two low-rank matrices:
    ΔW = B @ A, where A ∈ R^(r×d_in) and B ∈ R^(d_out×r)

    The forward pass becomes: h = W₀x + ΔWx = W₀x + BAx
    """

    def __init__(
            self,
            r: int,
            lora_alpha: float = 1.0,
            lora_dropout: float = 0.0,
            merge_weights: bool = False,
    ):
        """
        Initialize LoRA layer.

        Args:
            r: Rank of the low-rank decomposition
            lora_alpha: Scaling factor for LoRA weights
            lora_dropout: Dropout probability for LoRA layers
            merge_weights: Whether to merge LoRA weights with original weights
        """
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.merge_weights = merge_weights
        self.merged = False

        # Dropout layer for LoRA
        if lora_dropout > 0.0:
            self.lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout_layer = lambda x: x

    def reset_parameters(self):
        """Reset LoRA parameters using appropriate initialization."""
        # Initialize A with random values, B with zeros
        # This ensures ΔW = BA starts at zero
        if hasattr(self, 'lora_A') and hasattr(self.lora_A, 'weight'):
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        if hasattr(self, 'lora_B') and hasattr(self.lora_B, 'weight'):
            nn.init.zeros_(self.lora_B.weight)
