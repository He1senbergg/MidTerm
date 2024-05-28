import os
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

class CUB200Dataset(Dataset):
    def __init__(self, root_dir, is_train=True, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = pd.read_csv(os.path.join(root_dir, 'images.txt'), sep=' ', names=['img_id', 'img_name'])
        self.labels = pd.read_csv(os.path.join(root_dir, 'image_class_labels.txt'), sep=' ', names=['img_id', 'label'])
        self.train_test_split = pd.read_csv(os.path.join(root_dir, 'train_test_split.txt'), sep=' ', names=['img_id', 'is_train'])

        # 过滤出训练数据或测试数据
        self.data = self.images.merge(self.labels, on='img_id')
        self.data = self.data.merge(self.train_test_split, on='img_id')
        self.data = self.data[self.data['is_train'] == (1 if is_train else 0)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 'images', self.data.iloc[idx]['img_name'])
        image = Image.open(img_name).convert('RGB')
        label = self.data.iloc[idx]['label'] - 1  # 标签从0开始

        if self.transform:
            image = self.transform(image)

        return image, label

def data_transforms():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform

def load_data(data_dir, transforms, batch_size=32):
    train_dataset = CUB200Dataset(root_dir=data_dir, is_train=True, transform=transforms)
    val_dataset = CUB200Dataset(root_dir=data_dir, is_train=False, transform=transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader