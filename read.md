

**模型的架构：**
CNN：
CNN(
  (features): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU(inplace=True)
    (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (8): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): ReLU(inplace=True)
    (11): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Linear(in_features=2048, out_features=512, bias=True)
    (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
可训练参数数量: 1,148,938

MLP结构：
MLP(
  (net): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=3072, out_features=256, bias=True)
    (2): ReLU()
    (3): Dropout(p=0.3, inplace=False)
    (4): Linear(in_features=256, out_features=10, bias=True)
  )
)
可训练参数数量: 789,258


**CNN训练：**

训练格式：

```
python train.py --model cnn --epochs 30 --batch_size 128
```

恢复训练：

```
python train.py --model cnn --epochs 30 --batch_size 128 --checkpoint checkpoints/best_cnn.pth
```

测试：

```
python test.py --checkpoint checkpoints/best_cnn.pth
```

**MLP训练：**

训练格式：

```
python train.py --model mlp --epochs 30 --batch_size 256 --lr 0.001 --optimizer adam --val_ratio 0.2 --log_dir mlp_logs
```

恢复训练：

```
python train.py --model mlp --checkpoint checkpoints/best_mlp.pth --epochs 40 --lr 0.0001
```

测试：

```
python test.py --checkpoint mlp_checkpoints/best_mlp.pth
```

多模型对比：

```
python test.py --compare checkpoints/best_cnn.pth checkpoints/best_mlp.pth
```
