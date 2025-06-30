import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_utils.prune_quantize_resnet import get_resnet18, prune_model, quantize_model

def get_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])
    train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return DataLoader(train, batch_size=batch_size, shuffle=True), DataLoader(test, batch_size=batch_size)

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x,y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds==y).sum().item()
            total += y.size(0)
    return correct/total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_loaders()
    model = get_resnet18(num_classes=10).to(device)

    # Single-step training for demo
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    model.train()
    x_batch, y_batch = next(iter(train_loader))
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    optimizer.zero_grad()
    loss = criterion(model(x_batch), y_batch)
    loss.backward()
    optimizer.step()

    acc_base = evaluate(model, test_loader, device)
    print(f"Baseline accuracy: {acc_base:.4f}")

    pruned = prune_model(model, amount=0.5)
    acc_pruned = evaluate(pruned, test_loader, device)
    print(f"Pruned accuracy: {acc_pruned:.4f}")

    quantized = quantize_model(pruned)
    quantized.to(device)
    acc_quant = evaluate(quantized, test_loader, device)
    print(f"Quantized accuracy: {acc_quant:.4f}")

    torch.save(pruned.state_dict(), 'resnet_pruned.pth')
    torch.save(quantized.state_dict(), 'resnet_quantized.pth')

if __name__=='__main__':
    main()
