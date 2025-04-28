import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from .model import EMGHybridModel
from .data_loader import load_and_preprocess_data, EMGDataset

BATCH_SIZE = 64
EPOCHS = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(data_root):
    os.makedirs("models", exist_ok=True)

    segments, labels = load_and_preprocess_data(data_root)
    X_train, X_test, y_train, y_test = train_test_split(
        segments, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_dataset = EMGDataset(X_train, y_train, augment=True)
    test_dataset = EMGDataset(X_test, y_test, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    model = EMGHybridModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss, correct, total = 0, 0, 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE).squeeze()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pbar.set_postfix({'loss': train_loss/(total/BATCH_SIZE), 'acc': 100.*correct/total})

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE).squeeze()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_acc = 100 * val_correct / val_total
        print(f'Validation Accuracy: {val_acc:.2f}%')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'models/best_model.pth')

        scheduler.step(val_loss)

    print(f'Best Validation Accuracy: {best_acc:.2f}%')
    return model

if __name__ == "__main__":
    trained_model = train_model(data_root="path/to/your/data") 
