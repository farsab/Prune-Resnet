import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision import models

def get_resnet18(num_classes=10, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def prune_model(model, amount=0.5):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, 'weight', amount=amount)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            prune.remove(module, 'weight')
    return model

def quantize_model(model):
    return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
