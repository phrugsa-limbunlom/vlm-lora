import torch
from torch import nn

from lora.LoRALayer import LoRALayer


class LoRALinear(LoRALayer):
    """
    LoRA adaptation for linear layers.

    This replaces a standard nn.Linear layer with LoRA adaptation,
    allowing efficient fine-tuning by only updating the low-rank components.
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 4,
            lora_alpha: float = 1.0,
            lora_dropout: float = 0.0,
            bias: bool = True,
            merge_weights: bool = False,
    ):
        """
        Initialize LoRA linear layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            r: Rank of LoRA decomposition
            lora_alpha: LoRA scaling factor
            lora_dropout: Dropout probability
            bias: Whether to include bias
            merge_weights: Whether to merge weights during inference
        """
        super().__init__(r, lora_alpha, lora_dropout, merge_weights)

        self.in_features = in_features
        self.out_features = out_features

        # Original linear layer (frozen during LoRA training)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # LoRA components
        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            self.scaling = self.lora_alpha / self.r

            # Initialize parameters
            self.reset_parameters()

        # Freeze original weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LoRA linear layer.

        Args:
            x: Input tensor

        Returns:
            Output tensor with LoRA adaptation applied
        """
        # Original linear transformation
        result = self.linear(x)

        # Add LoRA adaptation if rank > 0
        if self.r > 0 and not self.merged:
            # LoRA forward: x -> A -> dropout -> B -> scale
            lora_result = self.lora_A(x)
            lora_result = self.lora_dropout_layer(lora_result)
            lora_result = self.lora_B(lora_result)
            result = result + lora_result * self.scaling

        return result

    def merge_weights(self):
        """Merge LoRA weights with original weights for inference efficiency."""
        if self.r > 0 and not self.merged:
            # Compute ΔW = B @ A
            delta_weight = (self.lora_B.weight @ self.lora_A.weight) * self.scaling

            # Merge with original weights
            self.linear.weight.data += delta_weight
            self.merged = True

    def unmerge_weights(self):
        """Unmerge LoRA weights from original weights."""
        if self.r > 0 and self.merged:
            # Compute ΔW = B @ A
            delta_weight = (self.lora_B.weight @ self.lora_A.weight) * self.scaling

            # Subtract from original weights
            self.linear.weight.data -= delta_weight
            self.merged = False