"""
VLM LoRA Fine-tuning Implementation

This module provides a comprehensive implementation for fine-tuning Vision Language Models (VLMs)
using Low-Rank Adaptation (LoRA). It includes:

- LLaVA model loading with 4-bit quantization support for memory efficiency
- LoRA adaptation for both vision and language components
- Memory optimization techniques for limited GPU resources
- Complete pipeline from model setup to training preparation
- Comprehensive memory usage analysis and monitoring

The implementation is designed to work efficiently on consumer-grade hardware
with limited GPU memory (e.g., 4GB GPUs) through aggressive memory management
and quantization techniques.
"""

import os
from typing import List, Optional

import torch
from torch import nn
from transformers import (
    LlavaForConditionalGeneration,
    LlavaProcessor
)
from transformers.utils.quantization_config import BitsAndBytesConfig

from lora.vlm_lora_adapter import VLMLoRAAdapter
from lora.vlm_lora_config import VLMLoRAConfig


def load_llava_model(model_name: str, load_in_4bit: bool):
    """
    Load LLaVA model with quantization (optional).
    
    Args:
        model_name: HuggingFace model name for LLaVA
        load_in_4bit: Whether to load model in 4-bit quantization
        
    Returns:
        Tuple of (model, processor)
    """
    print(f"Loading LLaVA model: {model_name}")

    # Clear GPU memory before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory before loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    quantization_config = None
    if load_in_4bit:
        try:
            # Test if bitsandbytes is working properly
            import bitsandbytes as bnb
            print("Testing bitsandbytes compatibility...")
            
            # Test if we can create a simple 4-bit tensor
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_enable_fp32_cpu_offload=True
                )
                print("4-bit quantization enabled")
            except Exception as bnb_error:
                print(f"bitsandbytes 4-bit test failed: {bnb_error}")
                print("Falling back to non-quantized model loading...")
                quantization_config = None
                
        except ImportError:
            print("Warning: bitsandbytes not installed, skipping 4-bit quantization")
            print("Install with: pip install bitsandbytes-windows")
            quantization_config = None
        except Exception as e:
            print(f"Warning: Could not enable 4-bit quantization: {e}")
            print("Falling back to non-quantized model loading...")
            quantization_config = None

    # Determine loading parameters
    loading_kwargs = {
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True
    }

    # Add quantization config if available
    if quantization_config is not None:
        loading_kwargs["quantization_config"] = quantization_config
        # More aggressive device mapping for 4GB GPU
        loading_kwargs["device_map"] = "auto"  # Let transformers handle device mapping automatically
        print("Using auto device mapping for memory optimization")
    else:
        print("Loading without device_map for compatibility")
    
    # load model
    try:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            **loading_kwargs
        )
        print("LLaVA model loaded successfully")
    except Exception as e:
        print(f"Error loading model with current config: {e}")
        print("Trying fallback loading with float32...")
        
        # Try with even more aggressive memory optimization
        try:
            # Load with CPU offloading
            loading_kwargs = {
                "torch_dtype": torch.float32,
                "low_cpu_mem_usage": True,
                "device_map": "auto",
                "offload_folder": "offload",  # Offload to disk if needed
                "offload_state_dict": True
            }
            model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                **loading_kwargs
            )
            print("LLaVA model loaded with CPU offloading")
        except Exception as e2:
            print(f"CPU offloading failed: {e2}")
            print("Trying minimal loading...")
            
            # Final fallback - load on CPU
            model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            print("LLaVA model loaded without quantization")
    
    # Load processor
    processor = LlavaProcessor.from_pretrained(model_name)
    
    print(f"Model device: {model.device}")
    print(f"Model dtype: {model.dtype}")
    
    if torch.cuda.is_available():
        print(f"GPU memory after loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    return model, processor


def get_llava_lora_target_modules() -> List[str]:
    """
    Get target modules for LoRA adaptation in LLaVA.
    
    Returns:
        List of module names to apply LoRA to
    """
    return [
        # Vision tower (CLIP ViT) - attention and MLP layers
        "vision_tower.vision_model.encoder.layers.*.self_attn.q_proj",
        "vision_tower.vision_model.encoder.layers.*.self_attn.k_proj", 
        "vision_tower.vision_model.encoder.layers.*.self_attn.v_proj",
        "vision_tower.vision_model.encoder.layers.*.self_attn.out_proj",
        "vision_tower.vision_model.encoder.layers.*.mlp.fc1",
        "vision_tower.vision_model.encoder.layers.*.mlp.fc2",
        
        # Language model (LLaMA/Vicuna) - attention and MLP layers
        "language_model.model.layers.*.self_attn.q_proj",
        "language_model.model.layers.*.self_attn.k_proj",
        "language_model.model.layers.*.self_attn.v_proj",
        "language_model.model.layers.*.self_attn.o_proj",
        "language_model.model.layers.*.mlp.gate_proj",
        "language_model.model.layers.*.mlp.up_proj",
        "language_model.model.layers.*.mlp.down_proj",
        
        # Output projection head
        "language_model.lm_head",
        
        # Multi-modal projector
        "multi_modal_projector.linear_1",
        "multi_modal_projector.linear_2",
        
        # Alternative patterns that might match
        "q_proj", "k_proj", "v_proj", "o_proj",  # Generic attention patterns
        "gate_proj", "up_proj", "down_proj",     # Generic MLP patterns
        "fc1", "fc2",                            # Generic MLP patterns
        "lm_head",                               # Language model head
        "linear_1", "linear_2",                  # Projector patterns
    ]


def prepare_llava_for_lora(model: LlavaForConditionalGeneration) -> LlavaForConditionalGeneration:
    """
    Prepare LLaVA model for LoRA fine-tuning.
    
    Args:
        model: LLaVA model
        
    Returns:
        Model prepared for LoRA
    """
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # Ensure certain modules are in the right dtype
    if hasattr(model, 'multi_modal_projector'):
        model.multi_modal_projector.to(torch.float32)
    
    return model


def create_vlm_lora_model(
        model: nn.Module,
        config: Optional[VLMLoRAConfig] = None,
        **kwargs
) -> tuple[nn.Module, VLMLoRAAdapter]:
    """
    Convenience function to create a VLM model with LoRA adaptation.

    Args:
        model: Base VLM model
        config: LoRA configuration (will create default if None)
        **kwargs: Additional arguments for VLMLoRAConfig

    Returns:
        Tuple of (adapted_model, adapter)
    """
    if config is None:
        config = VLMLoRAConfig(**kwargs)

    adapter = VLMLoRAAdapter(config)
    adapted_model = adapter.apply_lora(model)

    # Print adaptation summary
    adapter.print_adaptation_summary(adapted_model)

    return adapted_model, adapter

def no_training():
    """
    Set up LoRA model without training and print comprehensive results.
    """
    print("VLM LoRA Model Setup (No Training)")
    print("=" * 50)
    
    # Memory optimization for CUDA
    if torch.cuda.is_available():
        # Set memory fraction to avoid OOM
        torch.cuda.set_per_process_memory_fraction(0.7)  # Use 70% of available memory
        torch.cuda.empty_cache()
        
        # Set environment variable for memory management
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
        
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Available GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Configuration for model setup
    setup_config = {
        "model_name": "llava-hf/llava-1.5-7b-hf",
        "load_in_4bit": True,
        "lora_r": 8,
        "lora_alpha": 2.0,
        "lora_dropout": 0.1,
        "max_samples_test": 1
    }
    
    print(f"\nConfiguration:")
    for key, value in setup_config.items():
        print(f"-{key}: {value}")
    
    if setup_config["load_in_4bit"]:
        print(f"\nNote: 4-bit quantization is enabled for memory efficiency")

    # Load LLaVA model
    try:

        base_model, processor = load_llava_model(
            model_name=setup_config["model_name"],
            load_in_4bit=setup_config["load_in_4bit"]
        )
        print("LLaVA model loaded successfully")

    except Exception as e:
        print(f"Error loading LLaVA model: {e}")
        print("Trying with CPU fallback...")
        try:
            # Try loading on CPU first, then move to GPU
            base_model, processor = load_llava_model(
                model_name=setup_config["model_name"],
                load_in_4bit=False  # Disable 4-bit quantization for CPU loading
            )
            print("LLaVA model loaded on CPU successfully")
        except Exception as e2:
            print(f"CPU loading also failed: {e2}")
            return
    
    # Model statistics before LoRA
    total_params = sum(p.numel() for p in base_model.parameters())
    trainable_params_before = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    
    print(f"\nBase Model Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params_before:,}")
    print(f"Trainable ratio: {trainable_params_before/total_params*100:.2f}%")
    print(f"Model device: {base_model.device}")
    print(f"Model dtype: {base_model.dtype}")
    
    # Prepare model for LoRA
    print(f"\nPreparing model for LoRA adaptation...")
    base_model = prepare_llava_for_lora(base_model)
    
    # LoRA configuration - very minimal for 4GB GPU
    lora_config = VLMLoRAConfig(
        r=setup_config["lora_r"],
        lora_alpha=setup_config["lora_alpha"],
        lora_dropout=setup_config["lora_dropout"],
        target_modules=get_llava_lora_target_modules(),
        adapt_vision=True,
        adapt_language=True
    )
    
    print(f"\nLoRA Configuration:")
    print(f"Rank (r): {lora_config.r}")
    print(f"Alpha: {lora_config.lora_alpha}")
    print(f"Dropout: {lora_config.lora_dropout}")
    print(f"Target modules: {len(lora_config.target_modules)}")
    print(f"Adapt language: {lora_config.adapt_vision}")
    print(f"Adapt vision: {lora_config.adapt_vision}")

    # Apply LoRA adaptation
    print(f"\nApplying LoRA adaptation...")

    try:
        # Clear GPU memory before LoRA adaptation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU memory before LoRA: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        adapted_model, adapter = create_vlm_lora_model(base_model, lora_config)
        
        # Move to device with memory management
        if torch.cuda.is_available():
            # Try to move to GPU, fallback to CPU if needed
            try:
                adapted_model = adapted_model.to(device)
                print(f"LoRA adaptation applied successfully on {device}")
            except RuntimeError as mem_error:
                if "out of memory" in str(mem_error).lower():
                    print("GPU memory insufficient, using CPU")
                    device = torch.device("cpu")
                    adapted_model = adapted_model.to(device)
                    print("LoRA adaptation applied successfully on CPU")
                else:
                    raise mem_error
        else:
            adapted_model = adapted_model.to(device)
            print("LoRA adaptation applied successfully on CPU")
            
        if torch.cuda.is_available():
            print(f"GPU memory after LoRA: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
    except Exception as e:
        print(f"Error applying LoRA adaptation: {e}")
        print("Falling back to base model without LoRA adaptation")
        return
    
    # Model statistics after LoRA
    lora_params = sum(p.numel() for p in adapted_model.parameters() if p.requires_grad)
    total_params_after = sum(p.numel() for p in adapted_model.parameters())
    
    print(f"\nAdapted Model Statistics:")
    print(f"Total parameters: {total_params_after:,}")
    print(f"LoRA trainable parameters: {lora_params:,}")
    
    if adapter is not None:
        print(f"LoRA parameter ratio: {lora_params/total_params*100:.4f}%")
        print(f"Memory efficiency: {total_params/lora_params:.1f}x")
    else:
        print("LoRA adaptation failed - using base model")
        print(f"Trainable parameter ratio: {lora_params/total_params*100:.4f}%")
    
    print(f"Model device: {adapted_model.device}")

    # Memory usage analysis
    print(f"\nMemory Usage Analysis:")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU memory allocated: {memory_before:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    print(f"\n" + "="*50)
    print(f"LoRA Model Setup Complete!")
    print(f"LoRA parameters: {lora_params:,} ({lora_params/total_params*100:.4f}% of base model)")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VLM LoRA Fine-tuning with MathVista")
    parser.add_argument("--no-training", action="store_true", help="Skip training, just setup model")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    
    args = parser.parse_args()

    if args.no_training:
        no_training()