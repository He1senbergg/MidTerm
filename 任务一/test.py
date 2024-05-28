import os
import time
import torch
import argparse
import pandas as pd
from PIL import Image
import torch.nn as nn
from model import load_model
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, Dataset
from dataloader import data_transforms, load_data
from model import load_model, train_model

def parse_args():
    parser = argparse.ArgumentParser(description="Test a ResNet model on the CUB-200-2011 dataset.")
    parser.add_argument('--data_dir', type=str, default=r'/mnt/ly/models/deep_learning/mid_term/data/CUB_200_2011', help='Path to the CUB-200-2011 dataset directory.')
    parser.add_argument('--model_path', type=str, default=None, help='Path to a saved model checkpoint to continue training.')
    return parser.parse_args()

def main():
    args = parse_args()
    
    data_dir = args.data_dir
    batch_size = 32
    model_path = args.model_path

    transform = data_transforms()
    train_loader, val_loader = load_data(data_dir, transform, batch_size)

    criterion = nn.CrossEntropyLoss()
    model = load_model(200, model_path=model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 验证步骤
    model.eval()
    val_loss = 0.0
    corrects = 0

    # 开始验证计时
    val_start_time = time.time()
    model.to(device)
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)

    val_end_time = time.time()
    val_elapsed_time = val_end_time - val_start_time

    val_loss = val_loss / len(val_loader.dataset)
    val_acc = corrects.double() / len(val_loader.dataset)
    print(f'Test Loss: {val_loss:.4f}, Test Accuracy: {val_acc:.4f}, Test Time: {val_elapsed_time:.2f}s')

if __name__ == '__main__':
    main()
