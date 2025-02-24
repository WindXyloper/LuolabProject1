import torch
import torch.nn as nn
import torch.nn.init as init

class CNN(nn.Module):
  def __init__(self, dropout_rate=0.5, init_method='he'):
    super(CNN, self).__init__()
    
    # 特征提取层
    self.features = nn.Sequential(
      nn.Conv2d(3, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2),  # 16x16
      
      nn.Conv2d(32, 64, kernel_size=3, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2),  # 8x8
      
      nn.Conv2d(64, 128, kernel_size=3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2)  # 4x4
    )
    
    # 分类器
    self.classifier = nn.Sequential(
      nn.Linear(128 * 4 * 4, 512),
      nn.BatchNorm1d(512),
      nn.ReLU(inplace=True),
      nn.Dropout(dropout_rate),
      nn.Linear(512, 10)
    )
    
    # 初始化权重
    self._initialize_weights(init_method)

  def forward(self, x):
    x = self.features(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x

  def _initialize_weights(self, method):
    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if method == 'he':
          init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif method == 'xavier':
          init.xavier_normal_(m.weight)
        if m.bias is not None:
          init.constant_(m.bias, 0)

class MLP(nn.Module):
  def __init__(self, hidden_dim=256):
      super().__init__()
      self.net = nn.Sequential(
          nn.Flatten(),
          nn.Linear(32*32*3, hidden_dim),
          nn.ReLU(),
          nn.Dropout(0.3),
          nn.Linear(hidden_dim, 10)
      )
  def forward(self, x): return self.net(x)

  def _initialize_weights(self, method):
    for m in self.modules():
      if isinstance(m, nn.Linear):
        if method == 'he':
          init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif method == 'xavier':
          init.xavier_normal_(m.weight)
        init.constant_(m.bias, 0)

def create_model(model_name, device='cuda', **kwargs):
  model_dict = {
    'cnn': CNN,
    'mlp': MLP
  }
  
  assert model_name in model_dict, f"无效模型名称，可选: {list(model_dict.keys())}"
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model_dict[model_name](**kwargs).to(device)
  
  # print(f"[{model_name.upper()} 结构]")
  # print(model)
  # print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
  # print("设备:", device)
  # print("-"*50)
  
  return model

if __name__ == '__main__':
  # 测试模型创建
  cnn_model = create_model('cnn', dropout_rate=0.3)
  dummy_input = torch.randn(2, 3, 32, 32).to('cuda')
  print("\nCNN输出形状:", cnn_model(dummy_input).shape)
  
  mlp_model = create_model('mlp', hidden_dims=[1024, 512], dropout_rate=0.5)
  print("\nMLP输出形状:", mlp_model(dummy_input).shape)