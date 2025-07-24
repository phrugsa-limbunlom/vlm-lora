"""
LoRA Training Utilities

This module provides training utilities for LoRA fine-tuning, including
a trainer class, optimization functions, and evaluation metrics.
"""
import os
import time
from typing import Dict, List, Optional, Callable, Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from lora.vlm_lora_adapter import VLMLoRAAdapter


class LoRATrainer:
    """
    Trainer class for LoRA fine-tuning with comprehensive training utilities.
    """
    
    def __init__(
        self,
        model: nn.Module,
        adapter: VLMLoRAAdapter,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        save_dir: str = "./lora_checkpoints",
        logging_steps: int = 10,
        eval_steps: int = 100,
        save_steps: int = 500,
    ):
        """
        Initialize LoRA trainer.
        
        Args:
            model: Model with LoRA adaptations
            adapter: VLM LoRA adapter instance
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            optimizer: Optimizer (will create Adam if None)
            scheduler: Learning rate scheduler (optional)
            device: Device to train on
            gradient_accumulation_steps: Steps for gradient accumulation
            max_grad_norm: Maximum gradient norm for clipping
            save_dir: Directory to save checkpoints
            logging_steps: Steps between logging
            eval_steps: Steps between evaluation
            save_steps: Steps between saving checkpoints
        """
        self.model = model.to(device)
        self.adapter = adapter
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.save_dir = save_dir
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup optimizer
        if optimizer is None:
            trainable_params = adapter.get_trainable_parameters(model)
            self.optimizer = optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)
        else:
            self.optimizer = optimizer
        
        self.scheduler = scheduler
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'steps': []
        }
        
        # Print model info
        self.adapter.print_adaptation_summary(self.model)
    
    def train_epoch(self, loss_fn: Callable) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            loss_fn: Loss function
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            outputs = self.model(**batch)
            loss = loss_fn(outputs, batch)
            
            # Normalize loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.max_grad_norm
                )
                
                # Optimizer step
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.logging_steps == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    current_loss = loss.item() * self.gradient_accumulation_steps
                    
                    self.training_history['steps'].append(self.global_step)
                    self.training_history['train_loss'].append(current_loss)
                    self.training_history['learning_rate'].append(current_lr)
                    
                    # Log to wandb
                    wandb.log({
                        'train/loss': current_loss,
                        'train/learning_rate': current_lr,
                        'train/step': self.global_step,
                        'train/epoch': self.epoch
                    })
                    
                    progress_bar.set_postfix({
                        'loss': f"{current_loss:.4f}",
                        'lr': f"{current_lr:.2e}",
                        'step': self.global_step
                    })
                
                # Evaluation
                if self.val_loader is not None and self.global_step % self.eval_steps == 0:
                    val_metrics = self.evaluate(loss_fn)
                    self.training_history['val_loss'].append(val_metrics['val_loss'])
                    
                    # Log validation metrics to wandb
                    wandb.log({
                        'val/loss': val_metrics['val_loss'],
                        'val/step': self.global_step,
                        'val/epoch': self.epoch
                    })
                    
                    # Save best model
                    if val_metrics['val_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val_loss']
                        self.save_checkpoint("best_model")
                        
                        # Log best validation loss
                        wandb.log({
                            'val/best_loss': self.best_val_loss,
                            'val/best_step': self.global_step
                        })
                    
                    self.model.train()  # Return to training mode
                
                # Save checkpoint
                if self.global_step % self.save_steps == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}")
                    
                    # Log checkpoint info
                    wandb.log({
                        'checkpoint/step': self.global_step,
                        'checkpoint/epoch': self.epoch
                    })
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
        
        return {
            'train_loss': total_loss / num_batches,
            'global_step': self.global_step
        }
    
    def evaluate(self, loss_fn: Callable) -> Dict[str, float]:
        """
        Evaluate the model on validation set.
        
        Args:
            loss_fn: Loss function
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                batch = self._move_batch_to_device(batch)
                
                outputs = self.model(**batch)
                loss = loss_fn(outputs, batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        val_loss = total_loss / num_batches
        print(f"Validation Loss: {val_loss:.4f}")
        
        return {'val_loss': val_loss}
    
    def train(
        self,
        num_epochs: int,
        loss_fn: Callable,
        eval_fn: Optional[Callable] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
            loss_fn: Loss function
            eval_fn: Optional evaluation function
            
        Returns:
            Training history
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Training on device: {self.device}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train for one epoch
            train_metrics = self.train_epoch(loss_fn)
            
            # Evaluate
            if self.val_loader is not None:
                val_metrics = self.evaluate(loss_fn)
            else:
                val_metrics = {}
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_metrics['train_loss']:.4f}")
            if val_metrics:
                print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            
            # Log epoch summary to wandb
            wandb.log({
                'epoch/train_loss': train_metrics['train_loss'],
                'epoch/val_loss': val_metrics.get('val_loss', None),
                'epoch/epoch': epoch + 1,
                'epoch/total_epochs': num_epochs
            })
            
            # Save end-of-epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch + 1}")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        
        # Log final training summary
        wandb.log({
            'training/total_time_seconds': total_time,
            'training/total_steps': self.global_step,
            'training/final_train_loss': train_metrics['train_loss'],
            'training/best_val_loss': self.best_val_loss
        })
        
        return self.training_history
    
    def save_checkpoint(self, checkpoint_name: str):
        """
        Save training checkpoint.
        
        Args:
            checkpoint_name: Name for the checkpoint
        """
        checkpoint_path = os.path.join(self.save_dir, f"{checkpoint_name}.pth")
        
        # Save LoRA weights only (more efficient)
        self.adapter.save_lora_weights(self.model, checkpoint_path)
        
        # Save training state
        training_state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'optimizer_state': self.optimizer.state_dict(),
        }
        
        if self.scheduler is not None:
            training_state['scheduler_state'] = self.scheduler.state_dict()
        
        training_state_path = os.path.join(self.save_dir, f"{checkpoint_name}_training_state.pth")
        torch.save(training_state, training_state_path)
        
        print(f"Saved checkpoint: {checkpoint_name}")
        
        # Log checkpoint to wandb
        wandb.save(checkpoint_path)
        wandb.save(training_state_path)
    
    def load_checkpoint(self, checkpoint_name: str):
        """
        Load training checkpoint.
        
        Args:
            checkpoint_name: Name of the checkpoint to load
        """
        checkpoint_path = os.path.join(self.save_dir, f"{checkpoint_name}.pth")
        training_state_path = os.path.join(self.save_dir, f"{checkpoint_name}_training_state.pth")
        
        # Load LoRA weights
        self.adapter.load_lora_weights(self.model, checkpoint_path)
        
        # Load training state
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location=self.device)
            
            self.global_step = training_state['global_step']
            self.epoch = training_state['epoch']
            self.best_val_loss = training_state['best_val_loss']
            self.training_history = training_state['training_history']
            
            self.optimizer.load_state_dict(training_state['optimizer_state'])
            
            if self.scheduler is not None and 'scheduler_state' in training_state:
                self.scheduler.load_state_dict(training_state['scheduler_state'])
        
        print(f"Loaded checkpoint: {checkpoint_name}")
    
    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to specified device."""
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """
        Plot training curves.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.training_history['steps']:
            print("No training history to plot.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves
        ax1.plot(self.training_history['steps'], self.training_history['train_loss'], 
                label='Train Loss', color='blue')
        if self.training_history['val_loss']:
            # Create steps for validation loss (assuming it's logged every eval_steps)
            val_steps = [i * self.eval_steps for i in range(len(self.training_history['val_loss']))]
            ax1.plot(val_steps, self.training_history['val_loss'], 
                    label='Val Loss', color='red')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Learning rate curve
        ax2.plot(self.training_history['steps'], self.training_history['learning_rate'], 
                color='green')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved training curves to: {save_path}")
            
            # Log plot to wandb
            wandb.log({"training_curves": wandb.Image(save_path)})
        else:
            plt.show()

def create_optimizer(
        model: nn.Module,
        adapter: VLMLoRAAdapter,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        optimizer_type: str = "adamw"
) -> optim.Optimizer:
    """
    Create optimizer for LoRA training.

    Args:
        model: Model with LoRA adaptations
        adapter: VLM LoRA adapter
        learning_rate: Learning rate
        weight_decay: Weight decay
        optimizer_type: Type of optimizer ("adamw", "adam", "sgd")

    Returns:
        Optimizer instance
    """
    trainable_params = adapter.get_trainable_parameters(model)

    if optimizer_type.lower() == "adamw":
        return optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == "adam":
        return optim.Adam(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == "sgd":
        return optim.SGD(trainable_params, lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(
        optimizer: optim.Optimizer,
        scheduler_type: str = "cosine",
        num_training_steps: int = 1000,
        num_warmup_steps: int = 100,
        **kwargs
) -> Any:
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer instance
        scheduler_type: Type of scheduler ("cosine", "linear", "step")
        num_training_steps: Total number of training steps
        num_warmup_steps: Number of warmup steps
        **kwargs: Additional scheduler arguments

    Returns:
        Scheduler instance
    """
    if scheduler_type.lower() == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, T_max=num_training_steps, **kwargs)
    elif scheduler_type.lower() == "linear":
        from torch.optim.lr_scheduler import LinearLR
        return LinearLR(optimizer, start_factor=1.0, end_factor=0.1,
                        total_iters=num_training_steps, **kwargs)
    elif scheduler_type.lower() == "step":
        from torch.optim.lr_scheduler import StepLR
        return StepLR(optimizer, step_size=num_training_steps // 3, gamma=0.1, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")