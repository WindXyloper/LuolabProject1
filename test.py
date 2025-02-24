# test.py
import argparse
import torch
from dataset import get_cifar10_loaders
from model import create_model

def parse_args():
    """解析测试参数"""
    parser = argparse.ArgumentParser(description='测试参数')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint', type=str, help='单模型测试检查点路径')
    group.add_argument('--compare', nargs='+', help='对比多个模型检查点路径')
    parser.add_argument('--batch_size', type=int, default=256,help='测试批次大小')
    return parser.parse_args()

def evaluate(model, loader, device):
    """评估模型"""
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
    return correct / len(loader.dataset)

def compare_models(checkpoints, test_loader, device):
    """模型对比分析"""
    results = {}
    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path)
        model = create_model(ckpt['args']['model']).to(device)
        model.load_state_dict(ckpt['model_state'])
        acc = evaluate(model, test_loader, device)
        results[ckpt_path.split('/')[-1]] = acc
    
    print("\n===== 模型对比 =====")
    for name, acc in sorted(results.items()):
        model_type = name.split('_')[1].upper()
        print(f"{model_type:<6} | Accuracy: {acc:.2%}")

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载测试数据
    _, _, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        val_ratio=0.0
    )

    if args.compare:
        compare_models(args.compare, test_loader, device)
    else:
        ckpt = torch.load(args.checkpoint)
        model = create_model(ckpt['args']['model']).to(device)
        model.load_state_dict(ckpt['model_state'])
        acc = evaluate(model, test_loader, device)
        print(f"\nTest Accuracy: {acc:.4f}")

if __name__ == '__main__':
    main()