"""
Example: VLM LoRA Fine-tuning

This example demonstrates how to use the LoRA implementation for fine-tuning
a Vision Language Model (VLM). It includes a complete pipeline from model
setup to training and evaluation.
"""

import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    LlavaForConditionalGeneration,
    LlavaProcessor
)
from transformers.utils.quantization_config import BitsAndBytesConfig

import wandb
from dataset.MathVistaDataset import MathVistaDataset
from lora.vlm_lora_adapter import VLMLoRAAdapter
from lora.vlm_lora_config import VLMLoRAConfig
from lora_trainer import LoRATrainer, create_optimizer, create_scheduler


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


def llava_loss_function(outputs, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Loss function for LLaVA training.
    
    Args:
        outputs: Model outputs (LLaVA output object)
        batch: Batch of data
        
    Returns:
        Loss value
    """

    # return loss directly from output
    if hasattr(outputs, 'loss') and outputs.loss is not None:
        return outputs.loss
    
    # manual loss computation
    logits = outputs.logits
    labels = batch['labels']
    attention_mask = batch['attention_mask']
    
    # Shift labels for causal language modeling
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_attention_mask = attention_mask[:, 1:].contiguous()
    
    # Flatten
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    shift_attention_mask = shift_attention_mask.view(-1)
    
    # Compute loss only on non-padded tokens
    loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')
    loss = loss * shift_attention_mask

    if shift_attention_mask.sum() > 0:
        loss = loss.sum() / shift_attention_mask.sum()
    else:
        loss = loss.sum()
    
    return loss


def init_wandb(project_name: str = "vlm-lora", run_name: Optional[str] = None, config: Optional[Dict] = None):
    """
    Initialize Weights & Biases for experiment tracking.
    
    Args:
        project_name: Name of the wandb project
        run_name: Name of this specific run
        config: Configuration dictionary to log
        
    Returns:
        wandb run object
    """
    # Set default run name if not provided
    if run_name is None:
        import uuid
        run_name = f"llava-lora-{uuid.uuid4().hex[:8]}"
    
    # Initialize wandb
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        tags=["llava", "lora", "mathvista", "vlm"],
        notes="LLaVA LoRA fine-tuning on MathVista dataset"
    )
    
    print(f"🚀 Wandb initialized: {run.url}")
    return run


def train(args=None):
    """
    Main function demonstrating the complete LoRA fine-tuning pipeline.
    
    Args:
        args: Command line arguments (optional)
    """
    print("VLM LoRA Fine-tuning Example with MathVista Dataset")
    print("=" * 50)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Training configuration
    training_config = {
        "model_name": "llava-hf/llava-1.5-7b-hf",
        "load_in_4bit": True,
        "lora_r": 16,
        "lora_alpha": 32.0,
        "lora_dropout": 0.1,
        "learning_rate": 5e-4,
        "weight_decay": 0.01,
        "batch_size": 2,
        "num_epochs": 3,
        "gradient_accumulation_steps": 2,
        "max_grad_norm": 1.0,
        "warmup_ratio": 0.1,
        "save_dir": "./vlm_lora_mathvista_checkpoints",
        "logging_steps": 5,
        "eval_steps": 20,
        "save_steps": 50,
        "max_samples_train": 100,
        "max_samples_val": 50
    }
    
    # Initialize wandb
    project_name = args.wandb_project if args else "vlm-lora"
    run_name = args.wandb_run_name if args else None
    
    wandb_run = init_wandb(
        project_name=project_name,
        run_name=run_name,
        config=training_config
    )
    
    # Log system info
    wandb.log({
        "system/device": str(device),
        "system/cuda_available": torch.cuda.is_available(),
        "system/cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "system/torch_version": torch.__version__,
    })
    
    # Load LLaVA model
    print("\nLoading LLaVA model...")
    base_model, processor = load_llava_model(
        model_name=training_config["model_name"],
        load_in_4bit=training_config["load_in_4bit"]
    )
    
    # Log model info
    total_params = sum(p.numel() for p in base_model.parameters())
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    
    wandb.log({
        "model/total_parameters": total_params,
        "model/trainable_parameters": trainable_params,
        "model/model_name": training_config["model_name"],
        "model/load_in_4bit": training_config["load_in_4bit"]
    })
    
    # Prepare model for LoRA
    base_model = prepare_llava_for_lora(base_model)
    print(f"Base model parameters: {total_params:,}")
    
    # LoRA configuration for LLaVA
    lora_config = VLMLoRAConfig(
        r=training_config["lora_r"],
        lora_alpha=training_config["lora_alpha"],
        lora_dropout=training_config["lora_dropout"],
        target_modules=get_llava_lora_target_modules(),
        adapt_vision=True,
        adapt_language=True
    )
    
    # Log LoRA configuration
    wandb.log({
        "lora/r": lora_config.r,
        "lora/alpha": lora_config.lora_alpha,
        "lora/dropout": lora_config.lora_dropout,
        "lora/adapt_vision": lora_config.adapt_vision,
        "lora/adapt_language": lora_config.adapt_language
    })
    
    # Apply LoRA adaptation
    print("\nApplying LoRA adaptation...")
    adapted_model, adapter = create_vlm_lora_model(base_model, lora_config)
    adapted_model = adapted_model.to(device)
    
    # Ensure adapted_model is a proper model
    assert hasattr(adapted_model, 'generate'), "adapted_model must have generate method"
    
    # Log LoRA adaptation info
    lora_params = sum(p.numel() for p in adapted_model.parameters() if p.requires_grad)
    wandb.log({
        "lora/trainable_parameters": lora_params,
        "lora/parameter_ratio": lora_params / total_params
    })

    # Create datasets
    print("\nCreating MathVista datasets...")
    train_dataset = MathVistaDataset(
        split="testmini", 
        max_samples=training_config["max_samples_train"], 
        processor=processor
    )
    val_dataset = MathVistaDataset(
        split="testmini", 
        max_samples=training_config["max_samples_val"], 
        processor=processor
    )
    
    # Log dataset info
    wandb.log({
        "dataset/train_samples": len(train_dataset),
        "dataset/val_samples": len(val_dataset),
        "dataset/split": "testmini"
    })
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=training_config["batch_size"], shuffle=True)  # type: ignore
    val_loader = DataLoader(val_dataset, batch_size=training_config["batch_size"], shuffle=False)  # type: ignore
    
    # Create optimizer and scheduler
    print("\nSetting up training...")
    optimizer = create_optimizer(
        model=adapted_model,
        adapter=adapter,
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        optimizer_type="adamw"
    )
    
    num_training_steps = len(train_loader) * training_config["num_epochs"]
    num_warmup_steps = int(num_training_steps * training_config["warmup_ratio"])
    
    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_type="cosine",
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps
    )
    
    # Log training setup
    wandb.log({
        "training/num_training_steps": num_training_steps,
        "training/num_warmup_steps": num_warmup_steps,
        "training/batch_size": training_config["batch_size"],
        "training/gradient_accumulation_steps": training_config["gradient_accumulation_steps"],
        "training/learning_rate": training_config["learning_rate"],
        "training/weight_decay": training_config["weight_decay"]
    })
    
    # Create trainer
    trainer = LoRATrainer(
        model=adapted_model,
        adapter=adapter,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=str(device),
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        max_grad_norm=training_config["max_grad_norm"],
        save_dir=training_config["save_dir"],
        logging_steps=training_config["logging_steps"],
        eval_steps=training_config["eval_steps"],
        save_steps=training_config["save_steps"]
    )
    
    # Test model before training
    print("\nTesting model before training...")
    adapted_model.eval()
    with torch.no_grad():
        sample_batch = next(iter(val_loader))
        sample_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample_batch.items()}
        
        # LLaVA forward pass
        outputs = adapted_model(
            input_ids=sample_batch['input_ids'],
            attention_mask=sample_batch['attention_mask'],
            pixel_values=sample_batch['pixel_values'],
            labels=sample_batch['labels']
        )
        loss = llava_loss_function(outputs, sample_batch)
        print(f"Initial loss: {loss.item():.4f}")
        
        # Log initial loss
        wandb.log({"eval/initial_loss": loss.item()})
    
    # Train the model
    print("\nStarting training...")
    training_history = trainer.train(
        num_epochs=training_config["num_epochs"],
        loss_fn=llava_loss_function
    )
    
    # Test model after training
    print("\nTesting model after training...")
    adapted_model.eval()
    with torch.no_grad():
        outputs = adapted_model(
            input_ids=sample_batch['input_ids'],
            attention_mask=sample_batch['attention_mask'],
            pixel_values=sample_batch['pixel_values'],
            labels=sample_batch['labels']
        )
        loss = llava_loss_function(outputs, sample_batch)
        print(f"Final loss: {loss.item():.4f}")
        
        # Log final loss
        wandb.log({"eval/final_loss": loss.item()})
    
    # Save the final model
    print("\nSaving final model...")
    final_weights_path = "./vlm_mathvista_lora_final_weights.pth"
    adapter.save_lora_weights(adapted_model, final_weights_path)
    
    # Log model artifacts
    wandb.save(final_weights_path)
    
    # Demonstrate merging weights
    print("\nDemonstrating weight merging...")
    merged_model_path = "./vlm_mathvista_merged_model.pth"
    adapter.merge_and_save(adapted_model, merged_model_path)
    wandb.save(merged_model_path)
    
    # Print final statistics
    print("\nTraining Summary:")
    print(f"Training steps: {trainer.global_step}")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Final learning rate: {trainer.optimizer.param_groups[0]['lr']:.2e}")
    
    # Log final statistics
    wandb.log({
        "training/final_steps": trainer.global_step,
        "training/best_val_loss": trainer.best_val_loss,
        "training/final_learning_rate": trainer.optimizer.param_groups[0]['lr']
    })
    
    # Sample predictions
    print("\nSample predictions:")
    adapted_model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 3:  # Show 3 samples
                break
            
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Generate predictions
            generation_outputs = adapted_model.generate(  # type: ignore
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                pixel_values=batch['pixel_values'],
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=processor.tokenizer.eos_token_id  # type: ignore
            )
            
            # Decode predictions
            predictions = processor.tokenizer.batch_decode(generation_outputs, skip_special_tokens=True)  # type: ignore
            
            print(f"\nSample {i+1}:")
            print(f"Question: {batch['question'][0]}")
            print(f"True Answer: {batch['answer'][0]}")
            print(f"Model prediction: {predictions[0]}")
            
            # Log sample predictions
            wandb.log({
                f"predictions/sample_{i+1}/question": batch['question'][0],
                f"predictions/sample_{i+1}/true_answer": batch['answer'][0],
                f"predictions/sample_{i+1}/model_prediction": predictions[0]
            })
    
    # Plot training curves (if matplotlib is available)
    try:
        plot_path = "./mathvista_llava_training_curves.png"
        trainer.plot_training_curves(plot_path)
        wandb.save(plot_path)
    except ImportError:
        print("Matplotlib not available, skipping plot generation.")
    
    # Finish wandb run
    wandb.finish()
    
    print("\nMathVista LoRA fine-tuning completed")
    print("Check the generated checkpoints and training curves.")
    print(f"View experiment at: {wandb_run.url}")

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
    else:
        try:
            # Set environment variable to disable wandb if requested
            if args.no_wandb:
                os.environ["WANDB_MODE"] = "disabled"
                print(" Weights & Biases logging disabled")
            train(args)
        except Exception as e:
            print(f"\nError during training: {e}")
            import traceback
            traceback.print_exc() 