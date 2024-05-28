import os
import time
import torch
import torch.nn as nn
from torch.optim import Optimizer
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
from torch.utils.tensorboard import SummaryWriter

def load_model(num_classes=200, weights=ResNet18_Weights.IMAGENET1K_V1, train_from_scratch = False, model_path=None):
    if train_from_scratch:
        # 判断是否给了相互冲突的参数
        if model_path:
            raise ValueError("Cannot specify model_path when train_from_scratch is True.")
        # 如果设置了从头训练，则加载未经训练的模型
        model = models.resnet18()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_path:
        # 如果提供了模型路径，则加载模型权重
        model = models.resnet18()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(torch.load(model_path))
    else:
        # 否则加载预训练模型
        model = models.resnet18(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                criterion: nn.Module, optimizer: Optimizer, num_epochs: int = 10, 
                logdir: str ='/mnt/ly/models/deep_learning/mid_term/tensorboard/cfm-1',
                save_dir: str ='/mnt/ly/models/deep_learning/mid_term/model',
                milestones: list = None, gamma: float = 0.1):

    if milestones is None:
        milestones = [int(num_epochs * 0.5), int(num_epochs * 0.75)]
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.to(device)

    writer = SummaryWriter(log_dir=logdir)

    # 添加模型图
    init_img = torch.zeros((1, 3, 224, 224)).to(device)  # 假设输入图像尺寸为 (3, 224, 224)
    writer.add_graph(model, init_img)

    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        corrects = 0
        total = 0

        # 如果 optimizer 是 SGD 优化器，则执行后续操作
        if isinstance(optimizer, torch.optim.SGD):
            # 更新学习率
            if epoch+1 in milestones:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= gamma

        # 开始训练计时
        train_start_time = time.time()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = corrects.double() / total

        # 结束训练计时
        train_end_time = time.time()
        train_elapsed_time = train_end_time - train_start_time
        print(f'Epoch {epoch+1}/{num_epochs}, \nTrain Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}, Training Time: {train_elapsed_time:.2f}s')

        # 将训练loss和accuracy写入TensorBoard
        writer.add_scalar('Loss/Train Loss', epoch_loss, epoch)
        writer.add_scalar('Accuracy/Train Accuracy', epoch_acc, epoch)
        # writer.add_scalar('Training Time', train_elapsed_time, epoch)

        # 将当前学习率写入TensorBoard
        lr_i = 1
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            writer.add_scalar(f'Learning Rate/{lr_i}', current_lr, epoch)
            lr_i += 1

        # 验证步骤
        model.eval()
        val_loss = 0.0
        corrects = 0

        # 开始验证计时
        val_start_time = time.time()

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

        # 将验证loss和accuracy写入TensorBoard
        writer.add_scalar('Loss/Test Loss', val_loss, epoch)
        writer.add_scalar('Accuracy/Test Accuracy', val_acc, epoch)
        # writer.add_scalar('Validation Time', val_elapsed_time, epoch)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            file_path = f"{epoch+1}_{best_val_acc}.pth"
            # # 删除之前保存的模型
            # os.remove(os.path.join(save_dir, file_path))
            torch.save(model.state_dict(), os.path.join(save_dir, file_path))
    
    writer.flush()
    writer.close()