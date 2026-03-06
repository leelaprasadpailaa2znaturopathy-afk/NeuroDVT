import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def count_parameters(model):
    """Counts the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_training_curves(history_path='./checkpoints/history.json'):
    import json
    import os
    if not os.path.exists(history_path):
        print(f"No history found at {history_path}")
        return
        
    with open(history_path, 'r') as f:
        history = json.load(f)
        
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'r-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['test_acc'], 'b-', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    print("Learning curves saved as learning_curves.png")

def compare_model_sizes(models_dict):
    """
    Compares parameter counts for different models.
    models_dict: {'ModelName': model_instance}
    """
    print("\nParameter Efficiency Comparison:")
    print("-" * 40)
    for name, model in models_dict.items():
        params = count_parameters(model)
        print(f"{name:20}: {params:,} parameters")
    print("-" * 40)

# Optional GradCAM hook set up could go here
# For now, keeping it basic to ensure stability
