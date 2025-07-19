from typing import Optional, List

from lora.LoRAConfig import LoRAConfig


class VLMLoRAConfig(LoRAConfig):
    """
    Extended LoRA configuration specifically for VLM models.

    This includes common target modules for both vision and language components.
    """

    def __init__(
            self,
            r: int = 4,
            lora_alpha: float = 32.0,
            lora_dropout: float = 0.1,
            target_modules: Optional[List[str]] = None,
            merge_weights: bool = False,
            adapt_vision: bool = True,
            adapt_language: bool = True,
    ):
        """
        Initialize VLM LoRA configuration.

        Args:
            r: Rank of LoRA decomposition
            lora_alpha: LoRA scaling factor
            lora_dropout: Dropout probability
            target_modules: Custom target modules (overrides defaults)
            merge_weights: Whether to merge weights
            adapt_vision: Whether to adapt vision encoder
            adapt_language: Whether to adapt language model
        """
        if target_modules is None:
            target_modules = self._get_default_target_modules(adapt_vision, adapt_language)

        super().__init__(r, lora_alpha, lora_dropout, target_modules, merge_weights)
        self.adapt_vision = adapt_vision
        self.adapt_language = adapt_language

    def _get_default_target_modules(self, adapt_vision: bool, adapt_language: bool) -> List[str]:
        """Get default target modules for VLM adaptation."""
        modules = []

        if adapt_vision:
            # Common vision transformer modules
            modules.extend([
                "vision_encoder.transformer.layers",
                "vision_encoder.blocks",
                "visual.transformer.resblocks",
                "vision_model.encoder.layers",
                "q_proj", "k_proj", "v_proj", "out_proj",  # Attention projections
                "fc1", "fc2",  # MLP layers
            ])

        if adapt_language:
            # Common language model modules
            modules.extend([
                "language_model.model.layers",
                "text_model.encoder.layers",
                "transformer.h",
                "lm_head",
                "embed_tokens",
                "gate_proj", "up_proj", "down_proj",  # LLaMA-style MLP
            ])

        return modules
