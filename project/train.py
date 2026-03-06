import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import json
import time

from dataset_loader import get_dataloaders
from models.dvt_model import create_dvt_model

def train():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name = 'cifar10'
    batch_size = 64
    epochs = 100
    lr = 0.003
    weight_decay = 0.05
    save_dir = './checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    print(f"Using device: {device}")

    # Data
    train_loader, test_loader, num_classes = get_dataloaders(
        dataset_name=dataset_name, batch_size=batch_size
    )

    # Model
    model = create_dvt_model(num_classes=num_classes).to(device)
    
    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training Logs
    history = {
        'train_loss': [],
        'test_acc': [],
        'epochs': epochs,
        'lr': lr
    }

    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # PhD Tip: Gradient Clipping for Transformer stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / len(train_loader)
        history['train_loss'].append(epoch_loss)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        history['test_acc'].append(accuracy)
        
        print(f"Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        scheduler.step()
        
        # Save best model
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            
        # Save training logs periodically
        with open(os.path.join(save_dir, 'history.json'), 'w') as f:
            json.dump(history, f)

    print(f"Training complete. Best accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train()
