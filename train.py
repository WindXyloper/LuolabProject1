import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from dataset import get_cifar10_loaders
from model import create_model

def parse_args():
  """解析训练专用参数"""
  parser = argparse.ArgumentParser(description='训练参数')
  parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'mlp'], help='模型架构')
  parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
  parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
  parser.add_argument('--lr', type=float, default=0.001,help='学习率')
  parser.add_argument('--optimizer', type=str, default='adam',choices=['adam', 'sgd', 'rmsprop'],help='优化器选择')
  parser.add_argument('--val_ratio', type=float, default=0.1,help='验证集比例')
  parser.add_argument('--log_dir', type=str, default='runs',help='TensorBoard日志目录')
  parser.add_argument('--save_dir', type=str, default='checkpoints',help='模型保存目录')
  parser.add_argument('--checkpoint', type=str, default=None,help='加载的模型检查点路径')
  return parser.parse_args()

def train_epoch(model, loader, criterion, optimizer, device):
  """训练单个epoch"""
  model.train()
  total_loss = 0.0
  for inputs, labels in loader:
    inputs, labels = inputs.to(device), labels.to(device)
    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    total_loss += loss.item() * inputs.size(0)
  return total_loss / len(loader.dataset)

def validate(model, loader, criterion, device):
  """验证模型"""
  model.eval()
  total_loss = 0.0
  correct = 0
  with torch.no_grad():
    for inputs, labels in loader:
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      total_loss += loss.item() * inputs.size(0)
      _, preds = torch.max(outputs, 1)
      correct += (preds == labels).sum().item()
  return total_loss / len(loader.dataset), correct / len(loader.dataset)

def main():
  args = parse_args()
  # print("\n===== 环境检查 =====")
  # print(f"PyTorch版本: {torch.__version__}")
  # print(f"CUDA可用: {torch.cuda.is_available()}")
  if torch.cuda.is_available():
    # print(f"GPU数量: {torch.cuda.device_count()}")
    # print(f"当前GPU: {torch.cuda.current_device()}")
    # print(f"设备名称: {torch.cuda.get_device_name(0)}")
    device = torch.device('cuda')
  else:
    raise RuntimeError("未检测到可用GPU，终止训练！")
  # print("==================\n")
  
  # 初始化目录
  os.makedirs(args.save_dir, exist_ok=True)
  
  # 创建唯一日志目录（含微秒）
  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
  log_dir_name = f"{args.model}_{timestamp}"
  log_dir = os.path.join(os.path.abspath(args.log_dir), log_dir_name)
  
  # 稳健的目录创建
  retry = 3
  while retry > 0:
    try:
      os.makedirs(log_dir, exist_ok=False)
      break
    except FileExistsError:
      timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
      log_dir = os.path.join(args.log_dir, f"{args.model}_{timestamp}")
      retry -= 1
  else:
    raise RuntimeError("无法创建唯一日志目录")
  
  # 显式设置权限
  os.chmod(log_dir, 0o755)
  
  # 验证目录
  # print("日志目录绝对路径:", os.path.abspath(log_dir))
  assert os.path.isdir(log_dir), "目录创建失败"
  
  writer = SummaryWriter(log_dir)
  
  # 加载数据
  train_loader, val_loader, _ = get_cifar10_loaders(
      batch_size=args.batch_size, 
      val_ratio=args.val_ratio
  )
  
  # 创建模型
  model = create_model(args.model, device=device)
  if args.checkpoint:
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state'])
    epoch_start = checkpoint['epoch'] + 1
    optimizer = getattr(optim, args.optimizer.capitalize())(
      model.parameters(), lr=args.lr)
    optimizer.load_state_dict(checkpoint['optimizer_state'])
  else:
    epoch_start = 0
    optimizer = getattr(optim, args.optimizer.capitalize())(
      model.parameters(), lr=args.lr)
  
  criterion = nn.CrossEntropyLoss()
  
  best_acc = 0.0
  for epoch in range(args.epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    
    # 保存最佳模型
    if val_acc > best_acc:
      best_acc = val_acc
      torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'args': vars(args)
      }, os.path.join(args.save_dir, f'best_{args.model}.pth'))
    
    print(f"Epoch {epoch+1}/{args.epochs}")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

  writer.close()

if __name__ == '__main__':
    main()