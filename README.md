# VLM LoRA Setup and Demos

LoRA (Low-Rank Adaptation) implementation for Vision Language Models using LLaVA-1.5-7B. Focuses on LoRA setup, demonstrations, and testing.

## 🚀 Features

- **LLaVA Integration**: LLaVA-1.5-7B with 4-bit quantization
- **LoRA Setup**: Complete LoRA adapter for VLM models
- **Memory Efficient**: ~8GB VRAM (vs 28GB original)
- **Testing Suite**: Comprehensive component tests

## 📦 Installation

```bash
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

**Dependencies**: `transformers>=4.37.0`, `torch>=2.0.0`, `bitsandbytes>=0.41.0`, `peft>=0.8.0`

## 🚀 Quick Start

### LoRA Setup

```python
from main import load_llava_model, create_vlm_lora_model
from lora.vlm_lora_config import VLMLoRAConfig

# Load model
model, processor = load_llava_model("llava-hf/llava-1.5-7b-hf", load_in_4bit=True)

# Apply LoRA
config = VLMLoRAConfig(r=16, lora_alpha=32.0)
model, adapter = create_vlm_lora_model(model, config)
```

### Run Demos

```bash
python main.py --no-training
```

## 📁 Project Structure

```
├── main.py                 # LLaVA model loading & LoRA setup
├── lora/                  # LoRA implementation
│   ├── vlm_lora_adapter.py
│   ├── vlm_lora_config.py
│   ├── lora_linear.py
│   ├── lora_layer.py
│   └── lora_config.py
└── requirements.txt
```

## 🧪 Testing

```bash
python test_lora_dtype_fix.py
python test_gpu_usage.py
python test_device_mapping.py
python test_memory_allocation.py
```

## 🔧 Configuration

```python
config = VLMLoRAConfig(
    r=16,                    # LoRA rank
    lora_alpha=32.0,         # Scaling factor
    lora_dropout=0.1,        # Dropout
    adapt_vision=True,       # Adapt vision encoder
    adapt_language=True      # Adapt language model
)
```

## 📊 Performance

- **Memory**: 8GB VRAM (vs 28GB original)
- **Parameters**: ~1.5% trainable (LoRA only)
- **Speed**: 2-3x faster with 4-bit quantization

## 🚨 Requirements

- **GPU**: 8GB VRAM minimum
- **RAM**: 16GB system memory
- **Storage**: 20GB free space

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
