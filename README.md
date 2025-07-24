# VLM LoRA Fine-tuning with LLaVA

This repository provides a comprehensive implementation of LoRA (Low-Rank Adaptation) for fine-tuning Vision Language Models (VLMs), specifically optimized for **LLaVA** (Large Language and Vision Assistant).

## 🚀 Key Features

- **LLaVA Integration**: Uses LLaVA-1.5-7B model from HuggingFace
- **Memory-Efficient Training**: 4-bit quantization with BitsAndBytesConfig
- **Comprehensive LoRA Support**: Targets vision tower, language model, and projector layers
- **Production-Ready**: Includes proper data processing, training loops, and evaluation
- **MathVista Dataset**: Demonstrates mathematical reasoning in visual contexts

## 📦 Installation

### Basic Installation

```bash
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126  # FOR GPU SUPPORT
```
### Key Dependencies

- `transformers>=4.37.0` - HuggingFace transformers with LLaVA support
- `torch>=2.0.0` - PyTorch with CUDA support
- `bitsandbytes>=0.41.0` - 4-bit quantization
- `peft>=0.8.0` - Parameter-efficient fine-tuning
- `datasets>=2.0.0` - HuggingFace datasets
- `accelerate>=0.20.0` - HuggingFace accelerate for distributed training
- `wandb>=0.15.0` - Weights & Biases for experiment tracking

**Note**: If bitsandbytes installation fails, the code will automatically fall back to loading without quantization. This will use more memory but still work. Make sure to install PyTorch with CUDA support.

To test whether PyTorch has CUDA support

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

## 🏗️ Architecture

### LLaVA Model Integration

The implementation uses the LLaVA-1.5-7B model:

```python
# Load LLaVA with 4-bit quantization
model, processor = load_llava_model(
    model_name="llava-hf/llava-1.5-7b-hf",
    load_in_4bit=True
)
```

### LoRA Target Modules

Optimized for LLaVA architecture:

```python
target_modules = [
    # Vision Tower (CLIP ViT)
    "vision_tower.vision_model.encoder.layers.*.self_attn.q_proj",
    "vision_tower.vision_model.encoder.layers.*.self_attn.k_proj",
    "vision_tower.vision_model.encoder.layers.*.self_attn.v_proj",
    "vision_tower.vision_model.encoder.layers.*.self_attn.out_proj",
    "vision_tower.vision_model.encoder.layers.*.mlp.fc1",
    "vision_tower.vision_model.encoder.layers.*.mlp.fc2",
    
    # Language Model (LLaMA/Vicuna)
    "language_model.model.layers.*.self_attn.q_proj",
    "language_model.model.layers.*.self_attn.k_proj",
    "language_model.model.layers.*.self_attn.v_proj",
    "language_model.model.layers.*.self_attn.o_proj",
    "language_model.model.layers.*.mlp.gate_proj",
    "language_model.model.layers.*.mlp.up_proj",
    "language_model.model.layers.*.mlp.down_proj",
    
    # Output projection head
    "language_model.lm_head",
    
    # Multi-modal Projector
    "multi_modal_projector.linear_1",
    "multi_modal_projector.linear_2",
]
```

## 📊 Dataset Processing

### MathVista Dataset

The implementation includes a comprehensive MathVista dataset processor:

```python
# Load MathVista dataset
dataset = MathVistaDataset(
    data_path="path/to/mathvista",
    processor=processor,
    max_length=512
)

# Create data loaders
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
```

### LLaVA-Compatible Format

The dataset processing works with LLaVA's expected input format:

```python
# Process with LLaVA processor
inputs = processor(
    text=text_prompt,
    images=image,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512
)
```

## 🔧 Training Configuration

### Optimized LoRA Settings

```python
lora_config = VLMLoRAConfig(
    r=16,                    # Higher rank for better performance
    lora_alpha=32.0,         # Scaling factor
    lora_dropout=0.1,        # Dropout for regularization
    target_modules=get_llava_lora_target_modules(),
    adapt_vision=True,       # Adapt vision encoder
    adapt_language=True      # Adapt language model
)
```

### Memory-Efficient Training

- **4-bit Quantization**: Reduces memory usage by ~75%
- **Gradient Checkpointing**: Saves memory during backpropagation
- **Parameter Freezing**: Only LoRA parameters are trainable
- **Mixed Precision**: Uses FP16 for efficiency

## 🚀 Usage

### Basic Training

```python
# Load LLaVA model
model, processor = load_llava_model("llava-hf/llava-1.5-7b-hf")

# Apply LoRA adaptation
model, adapter = create_vlm_lora_model(model, lora_config)

# Train with MathVista dataset
trainer = LoRATrainer(model, adapter, train_loader, val_loader)
trainer.train(num_epochs=3, loss_fn=llava_loss_function)
```

### Quick Test

```python
python main.py --quick-test
```

### Full Training

```python
python main.py
```

## 📈 Performance

### Memory Usage

- **Original LLaVA-7B**: ~28GB VRAM
- **With 4-bit + LoRA**: ~8GB VRAM
- **Trainable Parameters**: ~1.5% of total parameters

### Training Speed

- **4-bit quantization**: 2-3x faster training
- **LoRA adaptation**: 10x fewer parameters to optimize
- **Gradient checkpointing**: Memory-efficient backpropagation

## 🔍 Model Architecture

### LLaVA Components

1. **Vision Tower**: CLIP ViT-L/14 (336px)
2. **Language Model**: Vicuna-7B (LLaMA-based)
3. **Multi-modal Projector**: 2-layer MLP
4. **LoRA Adapters**: Applied to all linear layers

### LoRA Integration

- **Vision Encoder**: Attention and MLP layers
- **Language Model**: Self-attention and feed-forward layers
- **Projector**: Linear transformation layers

## 📚 Files Overview

- `main.py`: Complete LLaVA fine-tuning pipeline with training and evaluation
- `LoRATrainer.py`: Training utilities, optimizer, and training loop
- `lora/`: LoRA implementation modules
  - `VLMLoRAConfig.py`: Configuration for VLM LoRA adaptation
  - `VLMLoRAAdapter.py`: Main LoRA adapter for VLM models
  - `LoRAConfig.py`: Base LoRA configuration
  - `LoRALinear.py`: LoRA linear layer implementation
  - `LoRALayer.py`: Base LoRA layer
- `dataset/`: Dataset processing
  - `MathVistaDataset.py`: MathVista dataset loader and processor
- `requirements.txt`: Optimized dependencies for LLaVA


## 🚨 Hardware Requirements with 4-bit Quantization

- **GPU**: 8GB VRAM minimum
- **RAM**: 16GB system memory
- **Storage**: 20GB free space

## 📖 Advanced Usage

### Custom Dataset

```python
# Create custom dataset
class CustomVLMDataset(Dataset):
    def __init__(self, data_path, processor):
        self.processor = processor
        # Load your data here
    
    def __getitem__(self, idx):
        # Return LLaVA-compatible format
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'labels': labels
        }
```

### Model Deployment

```python
# Save LoRA weights
adapter.save_lora_weights(model, "llava_lora_weights.pth")

# Load for inference
adapter.load_lora_weights(model, "llava_lora_weights.pth")
model.eval()
```

### Experiment Tracking

The implementation includes Weights & Biases integration for experiment tracking:

```python
# Initialize wandb
import wandb
wandb.init(project="vlm-lora", name="llava-finetuning")

# Log metrics during training
wandb.log({
    "loss": loss.item(),
    "learning_rate": scheduler.get_last_lr()[0]
})
```

## 🔧 Configuration Options

### LoRA Configuration

```python
config = VLMLoRAConfig(
    r=16,                           # LoRA rank
    lora_alpha=32.0,                # Scaling factor
    lora_dropout=0.1,               # Dropout rate
    target_modules=None,            # Auto-detect for LLaVA
    adapt_vision=True,              # Adapt vision encoder
    adapt_language=True,            # Adapt language model
    bias="none",                    # Bias adaptation
    task_type="CAUSAL_LM"           # Task type
)
```

### Training Configuration

```python
# Training parameters
training_args = {
    "num_epochs": 3,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "gradient_accumulation_steps": 4,
    "max_grad_norm": 1.0,
    "save_steps": 500,
    "eval_steps": 500,
    "logging_steps": 10
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---