from typing import List, Dict, Any

import torch
import torch.nn as nn

from lora.lora_config import LoRAConfig
from lora.lora_linear import LoRALinear


class VLMLoRAAdapter:
    """
    Adapter class for applying LoRA to Vision Language Models.

    This class handles the automatic replacement of linear layers with
    LoRA-adapted versions, supporting both vision and language components.
    """

    def __init__(self, config: LoRAConfig):
        """
        Initialize VLM LoRA adapter.

        Args:
            config: LoRA configuration specifying parameters and target modules
        """
        self.config = config
        self.adapted_modules = {}
        self.original_modules = {}
        
        print(f"LoRA Adapter initialized with target modules: {self.config.target_modules}")
        print(f"LoRA rank: {self.config.r}")
        print(f"LoRA alpha: {self.config.lora_alpha}")

    def apply_lora(self, model: nn.Module, module_name: str = "") -> nn.Module:
        """
        Apply LoRA adaptation to a model.

        Args:
            model: The model to adapt
            module_name: Current module name (used for recursion)

        Returns:
            Model with LoRA adaptations applied
        """
        for name, module in model.named_children():
            full_name = f"{module_name}.{name}" if module_name else name

            if isinstance(module, nn.Linear):
                print(f"Found linear module: {full_name}")

            # Check if this module should be adapted
            if self._should_adapt_module(full_name, module):

                # Replace with LoRA version
                lora_module = self._create_lora_module(module)
                setattr(model, name, lora_module)

                # Store references
                self.adapted_modules[full_name] = lora_module
                self.original_modules[full_name] = module

                print(f"Applied LoRA to: {full_name}")
            else:
                # Recursively apply to children
                self.apply_lora(module, full_name)

        return model

    def _should_adapt_module(self, module_name: str, module: nn.Module) -> bool:
        """
        Determine if a module should be adapted with LoRA.

        Args:
            module_name: Name of the module
            module: The module instance

        Returns:
            True if module should be adapted
        """
        # Check if it's a linear layer
        if not isinstance(module, nn.Linear):
            return False

        # Check if module name matches target patterns
        for target in self.config.target_modules:
            if target in module_name:
                print(f"Matched module '{module_name}' with target pattern '{target}'")
                return True

        return False

    def _create_lora_module(self, original_module: nn.Linear) -> LoRALinear:
        """
        Create a LoRA version of a linear module.

        Args:
            original_module: Original linear module

        Returns:
            LoRA-adapted linear module
        """
        # Get the dtype and device from the original module
        original_dtype = original_module.weight.dtype
        original_device = original_module.weight.device
        
        print(f"Creating LoRA module for {original_module.weight.shape} with dtype {original_dtype} on device {original_device}")
        
        # Create LoRA module with same dimensions
        lora_module = LoRALinear(
            in_features=original_module.in_features,
            out_features=original_module.out_features,
            r=self.config.r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias=original_module.bias is not None,
            merge_weights=self.config.merge_weights
        )

        # Forward pass one time
        print("========Test forward pass========")
        x = torch.randn(3, lora_module.in_features)
        print("- input shape: ", x.shape)
        lora_module.forward(x)
        
        # Move LoRA module to the same device and dtype as original
        lora_module = lora_module.to(original_device).to(original_dtype)

        # Copy original weights with proper type handling
        try:
            lora_module.linear.weight.data = original_module.weight.data.clone()
            if original_module.bias is not None:
                lora_module.linear.bias.data = original_module.bias.data.clone()
        except Exception as e:
            print(f"Error copying weights: {e}")
            print(f"Original weight dtype: {original_module.weight.dtype}, device: {original_module.weight.device}")
            print(f"LoRA weight dtype: {lora_module.linear.weight.dtype}, device: {lora_module.linear.weight.device}")
            raise

        print(f"- Original weight: {original_module.weight.shape}")
        print(f"- LoRA weight: {lora_module.linear.weight.shape}")

        return lora_module

    def merge_and_save(self, model: nn.Module, save_path: str):
        """
        Merge LoRA weights and save the model.

        Args:
            model: Model with LoRA adaptations
            save_path: Path to save the merged model
        """
        # Merge all LoRA weights
        for module_name, lora_module in self.adapted_modules.items():
            if hasattr(lora_module, 'merge_weights'):
                lora_module.merge_weights()
                print(f"Merged LoRA weights for: {module_name}")

        # Save the model
        torch.save(model.state_dict(), save_path)
        print(f"Saved merged model to: {save_path}")

    def save_lora_weights(self, model: nn.Module, save_path: str):
        """
        Save only the LoRA weights (not the full model).

        Args:
            model: Model with LoRA adaptations
            save_path: Path to save LoRA weights
        """
        lora_state_dict = {}

        for name, param in model.named_parameters():
            if 'lora_' in name and param.requires_grad:
                lora_state_dict[name] = param.data.clone()

        torch.save({
            'lora_state_dict': lora_state_dict,
            'config': self.config.to_dict()
        }, save_path)

        print(f"Saved LoRA weights to: {save_path}")
        print(f"Number of LoRA parameters: {len(lora_state_dict)}")

    def load_lora_weights(self, model: nn.Module, load_path: str):
        """
        Load LoRA weights into a model.

        Args:
            model: Model to load LoRA weights into
            load_path: Path to load LoRA weights from
        """
        checkpoint = torch.load(load_path, map_location='cpu')
        lora_state_dict = checkpoint['lora_state_dict']

        # Load LoRA weights
        missing_keys, unexpected_keys = model.load_state_dict(lora_state_dict, strict=False)

        print(f"Loaded LoRA weights from: {load_path}")
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

    def get_trainable_parameters(self, model: nn.Module) -> List[nn.Parameter]:
        """
        Get all trainable parameters (LoRA parameters).

        Args:
            model: Model with LoRA adaptations

        Returns:
            List of trainable parameters
        """
        trainable_params = []
        for param in model.parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params

    def print_adaptation_summary(self, model: nn.Module):
        """
        Print a summary of the LoRA adaptation.

        Args:
            model: Model with LoRA adaptations
        """
        param_info = count_lora_parameters(model)

        print("\n" + "=" * 50)
        print("LoRA Adaptation Summary")
        print("=" * 50)
        print(f"Total parameters: {param_info['total_parameters']:,}")
        print(f"Trainable parameters: {param_info['trainable_parameters']:,}")
        print(f"LoRA parameters: {param_info['lora_parameters']:,}")
        print(f"Trainable percentage: {param_info['trainable_percentage']:.4f}%")
        print(f"Adapted modules: {len(self.adapted_modules)}")
        print(f"Target modules: {self.config.target_modules}")
        print(f"LoRA rank: {self.config.r}")
        print(f"LoRA alpha: {self.config.lora_alpha}")
        print(f"LoRA dropout: {self.config.lora_dropout}")
        
        # Show which modules were actually adapted
        if self.adapted_modules:
            print(f"\nAdapted module names:")
            for module_name in self.adapted_modules.keys():
                print(f"  - {module_name}")
        
        # Show parameter efficiency
        if param_info['total_parameters'] > 0:
            efficiency = param_info['total_parameters'] / param_info['trainable_parameters']
            print(f"\nParameter efficiency: {efficiency:.1f}x (only {param_info['trainable_percentage']:.4f}% of parameters are trainable)")
        
        print("=" * 50)


def count_lora_parameters(model: nn.Module) -> Dict[str, Any]:
    """
    Count the number of LoRA parameters in a model.

    Args:
        model: PyTorch model with LoRA layers

    Returns:
        Dictionary with parameter counts
    """
    lora_params = 0
    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if 'lora_' in name:
                lora_params += param.numel()

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "lora_parameters": lora_params,
        "trainable_percentage": 100 * trainable_params / total_params if total_params > 0 else 0.0,
    }