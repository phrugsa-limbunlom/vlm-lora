#!/usr/bin/env python3
"""
Weights & Biases (wandb) Setup and Configuration

This script helps you set up wandb for the VLM LoRA project and provides
examples of how to use wandb for experiment tracking.
"""

import os
import sys
import subprocess
from typing import Dict, Any


def check_wandb_installation():
    """Check if wandb is installed and accessible."""
    try:
        import wandb
        print(f"✅ Wandb is installed (version: {wandb.__version__})")
        return True
    except ImportError:
        print("❌ Wandb is not installed")
        return False


def install_wandb():
    """Install wandb if not already installed."""
    print("Installing wandb...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb>=0.15.0"])
        print("✅ Wandb installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install wandb")
        return False


def setup_wandb_login():
    """Guide user through wandb login process."""
    print("\n🔐 Wandb Login Setup")
    print("=" * 40)
    
    # Check if already logged in
    try:
        import wandb
        api = wandb.Api()
        # Try to access user info
        user = api.default_entity
        print(f"✅ Already logged in as: {user}")
        return True
    except Exception:
        pass
    
    print("You need to log in to wandb to track your experiments.")
    print("1. Go to https://wandb.ai/authorize")
    print("2. Create an account or log in")
    print("3. Copy your API key")
    print("4. Run: wandb login")
    
    response = input("\nWould you like to run 'wandb login' now? (y/n): ")
    if response.lower() in ['y', 'yes']:
        try:
            subprocess.run(["wandb", "login"], check=True)
            print("✅ Wandb login successful")
            return True
        except subprocess.CalledProcessError:
            print("❌ Wandb login failed")
            return False
        except FileNotFoundError:
            print("❌ Wandb CLI not found. Please install wandb first.")
            return False
    
    return False


def create_wandb_config():
    """Create a sample wandb configuration file."""
    config_content = """# wandb_config.yaml
# Sample configuration for VLM LoRA experiments

# Project settings
project_name: "vlm-lora"
entity: null  # Set to your wandb username if needed

# Experiment tracking
log_model: true
log_code: true
log_artifacts: true

# Hyperparameter tracking
track_hyperparameters: true

# System monitoring
monitor_gpu: true
monitor_system: true

# Logging frequency
log_interval: 10  # Log every N steps
eval_interval: 50  # Evaluate every N steps

# Artifacts
save_model_artifacts: true
save_training_curves: true
save_predictions: true
"""
    
    config_path = "wandb_config.yaml"
    with open(config_path, "w") as f:
        f.write(config_content)
    
    print(f"✅ Created sample wandb config: {config_path}")
    return config_path


def show_usage_examples():
    """Show examples of how to use wandb with the project."""
    examples = """
📚 Wandb Usage Examples
=======================

1. Basic Training with Wandb:
   python main.py

2. Custom Project Name:
   python main.py --wandb-project "my-vlm-experiments"

3. Custom Run Name:
   python main.py --wandb-run-name "llava-mathvista-v1"

4. Disable Wandb:
   python main.py --no-wandb

5. Quick Test (no wandb):
   python main.py --quick-test

6. Environment Variables:
   export WANDB_PROJECT="vlm-lora"
   export WANDB_RUN_NAME="experiment-1"
   python main.py

📊 What Gets Logged:
- Training loss and learning rate
- Validation loss and metrics
- Model parameters and architecture
- System information (GPU, memory)
- Training curves and plots
- Model checkpoints and artifacts
- Sample predictions
- Training time and efficiency metrics

🔧 Advanced Configuration:
- Modify wandb_config.yaml for custom settings
- Use wandb.init() with custom parameters
- Add custom metrics with wandb.log()
- Track custom artifacts with wandb.save()

📈 Viewing Results:
- Check the wandb URL printed during training
- Visit https://wandb.ai/[username]/[project]
- Compare runs and hyperparameters
- Download artifacts and models
"""
    
    print(examples)


def main():
    """Main setup function."""
    print("🚀 Wandb Setup for VLM LoRA Project")
    print("=" * 50)
    
    # Check installation
    if not check_wandb_installation():
        install_choice = input("Would you like to install wandb? (y/n): ")
        if install_choice.lower() in ['y', 'yes']:
            if not install_wandb():
                print("❌ Setup failed. Please install wandb manually.")
                return
        else:
            print("❌ Wandb is required for experiment tracking.")
            return
    
    # Setup login
    if not setup_wandb_login():
        print("⚠️ Wandb login not completed. You can still run experiments but they won't be tracked.")
    
    # Create config
    config_path = create_wandb_config()
    
    # Show examples
    show_usage_examples()
    
    print("\n🎉 Wandb setup completed!")
    print(f"📁 Configuration file: {config_path}")
    print("🚀 You can now run: python main.py")


if __name__ == "__main__":
    main() 