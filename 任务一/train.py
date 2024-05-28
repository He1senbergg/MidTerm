import torch
import argparse
from torch import nn
from model import load_model, train_model
from torchvision.models import ResNet18_Weights
from dataloader import data_transforms, load_data

def parse_args():
    parser = argparse.ArgumentParser(description="Train a ResNet model on the CUB-200-2011 dataset.")
    parser.add_argument('--num_classes', type=int, default=200, help='Number of classes in the dataset.')
    parser.add_argument('--data_dir', type=str, default=r'/mnt/ly/models/deep_learning/mid_term/data/CUB_200_2011', help='Path to the CUB-200-2011 dataset directory.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=70, help='Number of epochs for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for the SGD optimizer.')
    parser.add_argument('--weights', type=str, default='IMAGENET1K_V1', help='Pre-trained weights to use.')
    parser.add_argument('--model_path', type=str, default=None, help='Path to a saved model checkpoint to continue training.')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam'], default='SGD', help='Optimizer to use (SGD or Adam).')
    parser.add_argument('--logdir', type=str, default="/mnt/ly/models/deep_learning/mid_term/tensorboard/1", help='Directory to save TensorBoard logs.')
    parser.add_argument('--save_dir', type=str, default="/mnt/ly/models/deep_learning/mid_term/model", help='Directory to save model checkpoints.')
    parser.add_argument('--scratch', type=bool, default=False, help='Train the model from scratch.')
    parser.add_argument('--decay', type=float, default=1e-3, help='Weight decay for the optimizer.')
    parser.add_argument('--milestones', type=list, default=None, help='List of epochs to decrease the learning rate.')
    parser.add_argument('--gamma', type=float, default=0.1, help='Factor to decrease the learning rate.')
    return parser.parse_args()

def main():
    args = parse_args()
    
    num_classes = args.num_classes
    data_dir = args.data_dir
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    momentum = args.momentum
    logdir = args.logdir
    save_dir = args.save_dir
    model_path = args.model_path
    scratch = args.scratch
    decay = args.decay
    milestones = args.milestones
    gamma = args.gamma

    if milestones:
        for milestone in milestones:
            if milestone > num_epochs:
                raise ValueError("Milestone epoch cannot be greater than the total number of epochs.")
            
    if args.weights == 'IMAGENET1K_V1':
        weights = ResNet18_Weights.IMAGENET1K_V1
    else:
        weights = None

    transform = data_transforms()
    train_loader, val_loader = load_data(data_dir, transform, batch_size)

    model = load_model(num_classes, weights=weights, train_from_scratch = scratch, model_path=model_path)
    criterion = nn.CrossEntropyLoss()
    
    if scratch or model_path:
        parameters = [{"params": model.parameters(), "lr": learning_rate}]
    else:
        parameters = [
        {"params": model.fc.parameters(), "lr": learning_rate},
        {"params": [param for name, param in model.named_parameters() if "fc" not in name], "lr": learning_rate*0.1}
        ]

    # 根据命令行参数选择优化器
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(parameters, momentum=momentum, weight_decay=decay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(parameters, weight_decay=decay)

    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, logdir, save_dir, milestones, gamma)

if __name__ == '__main__':
    main()