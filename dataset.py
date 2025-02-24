import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

class CIFAR10Dataset(Dataset):
  def __init__(self, root_dir, train=True, transform=None):
    self.root_dir = root_dir
    self.transform = transform
    self.data = []
    self.labels = []
    
    # 加载数据文件
    if train:
      files = [f"data_batch_{i}" for i in range(1, 6)]
    else:
      files = ["test_batch"]
        
    for file in files:
      file_path = os.path.join(root_dir, file)
      with open(file_path, 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
        self.data.append(entry['data'])
        self.labels.extend(entry['labels' if 'labels' in entry else 'fine_labels'])
    
    # 合并数据
    self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
    self.data = self.data.transpose((0, 2, 3, 1))  # (N, H, W, C)
    self.labels = np.array(self.labels)
    
    # 加载类别名称
    meta_path = os.path.join(root_dir, 'batches.meta')
    with open(meta_path, 'rb') as f:
      meta = pickle.load(f, encoding='latin1')
      self.classes = meta['label_names' if 'label_names' in meta else 'fine_label_names']

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    img = self.data[idx]
    label = self.labels[idx]
    
    # 转换为PIL Image进行变换
    img = transforms.ToPILImage()(img)
    
    if self.transform:
        img = self.transform(img)
        
    return img, label

def get_cifar10_loaders(batch_size=64, val_ratio=0.1, data_dir='cifar-10-batches-py'):
  # 数据预处理
  train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
  ])
  
  test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
  ])

  # 加载数据集
  train_set = CIFAR10Dataset(
    root_dir=data_dir,
    train=True,
    transform=train_transform
  )
  
  test_set = CIFAR10Dataset(
    root_dir=data_dir,
    train=False,
    transform=test_transform
  )

  # 划分验证集
  val_size = int(len(train_set) * val_ratio)
  train_size = len(train_set) - val_size
  train_set, val_set = random_split(train_set, [train_size, val_size])

  # 创建DataLoader
  train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
  val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
  test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
  
  return train_loader, val_loader, test_loader

if __name__ == '__main__':
  # 测试数据加载
  train_loader, val_loader, test_loader = get_cifar10_loaders()
  
  print(f"训练集样本数量: {len(train_loader.dataset)}")
  print(f"验证集样本数量: {len(val_loader.dataset)}")
  print(f"测试集样本数量: {len(test_loader.dataset)}")
  
  # 检查第一个批次的数据
  sample_data, sample_labels = next(iter(train_loader))
  print("\n数据形状:", sample_data.shape)
  print("标签形状:", sample_labels.shape)
  print("类别示例:", train_loader.dataset.dataset.classes[sample_labels[0].item()])