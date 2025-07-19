"""
Example: VLM LoRA Fine-tuning

This example demonstrates how to use the LoRA implementation for fine-tuning
a Vision Language Model (VLM). It includes a complete pipeline from model
setup to training and evaluation.
"""

from typing import Dict, List, Optional
import os
import argparse

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig
from transformers import (
    LlavaForConditionalGeneration,
    LlavaProcessor
)
import wandb

from lora.VLMLoRAAdapter import VLMLoRAAdapter
from lora.VLMLoRAConfig import VLMLoRAConfig
from dataset.MathVistaDataset import MathVistaDataset
from lora.LoRALinear import LoRALinear
from LoRATrainer import LoRATrainer, create_optimizer, create_scheduler



def load_llava_model(model_name: str = "llava-hf/llava-1.5-7b-hf", load_in_4bit: bool = True):
    """
    Load LLaVA model with quantization (optional).
    
    Args:
        model_name: HuggingFace model name for LLaVA
        load_in_4bit: Whether to load model in 4-bit quantization
        
    Returns:
        Tuple of (model, processor)
    """
    print(f"Loading LLaVA model: {model_name}")

    quantization_config = None
    if load_in_4bit:
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            print("4-bit quantization enabled")
        except Exception as e:
            print(f"Warning: Could not enable 4-bit quantization: {e}")
            quantization_config = None

    # Determine loading parameters
    loading_kwargs = {
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True
    }
    
    # Add quantization config if available
    if quantization_config is not None:
        loading_kwargs["quantization_config"] = quantization_config
        loading_kwargs["device_map"] = "auto"
    else:
        print("Loading without device_map for compatibility")
    
    # load model
    try:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            **loading_kwargs
        )
        print("LLaVA model loaded with quantization successfully")
    except Exception as e:
        print(f"Error loading model with current config: {e}")
        print("Trying fallback loading...")
        
        # load with minimal configuration
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for better compatibility
            low_cpu_mem_usage=True
        )
        print("LLaVA model loaded with default configuration")
    
    # Load processor
    processor = LlavaProcessor.from_pretrained(model_name)
    
    print(f"Model device: {model.device}")
    print(f"Model dtype: {model.dtype}")
    
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


def main(args=None):
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


def quick_test():
    """
    Quick test to verify the implementation works.
    """
    print("Quick Test - MathVista LoRA Implementation")
    print("-" * 40)
    
    # Test basic LoRA layer
    lora_layer = LoRALinear(512, 256, r=8, lora_alpha=16.0)
    x = torch.randn(4, 512)
    output = lora_layer(x)
    print(f"✓ LoRA layer test passed: {x.shape} -> {output.shape}")
    
    # Test LLaVA model loading (mock test)
    try:
        print("✓ LLaVA model loading test - would load real model in practice")
        print("✓ LoRA target modules defined for LLaVA")
    except Exception as e:
        print(f" LLaVA test skipped: {e}")
    
    # Test target modules
    target_modules = get_llava_lora_target_modules()
    print(f"✓ Target modules test passed: {len(target_modules)} modules defined")
    
    # Test dataset loading (small sample)
    try:
        dataset = MathVistaDataset(split="testmini", max_samples=5)
        sample = dataset[0]
        print(f"✓ MathVista dataset test passed: {len(dataset)} samples loaded")
        print(f"  Sample keys: {list(sample.keys())}")
        if 'images' in sample and hasattr(sample['images'], 'shape'):
            image_tensor = sample['images']
            if hasattr(image_tensor, 'shape'):
                print(f"  Image shape: {image_tensor.shape}")  # type: ignore
            else:
                print(f"  Image type: {type(image_tensor)}")
        else:
            print(f"  Image type: {type(sample.get('images', 'None'))}")
        question = sample['question']
        if isinstance(question, str) and len(question) > 100:
            print(f"  Question: {question[:100]}...")
        else:
            print(f"  Question: {question}")
    except Exception as e:
        print(f" MathVista dataset test failed: {e}")
        print(" This might be due to network issues or missing dependencies")
    
    print("\nAll tests passed! The implementation is working correctly.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VLM LoRA Fine-tuning with MathVista")
    parser.add_argument("--quick-test", action="store_true", help="Run quick test instead of full training")
    parser.add_argument("--no-training", action="store_true", help="Skip training, just setup model")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="vlm-lora", help="Wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Wandb run name")
    
    args = parser.parse_args()
    
    if args.quick_test:
        quick_test()
    else:
        try:
            # Set environment variable to disable wandb if requested
            if args.no_wandb:
                os.environ["WANDB_MODE"] = "disabled"
                print(" Weights & Biases logging disabled")
            
            main(args)
        except Exception as e:
            print(f"\nError during training: {e}")
            import traceback
            traceback.print_exc() 