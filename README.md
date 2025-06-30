# Prune-Resnet
=======
# ResNet18 Pruning & Quantization Toolkit

## Project Title
Compressing ResNet18: L1 Pruning & Dynamic Quantization

## Description
A production-ready toolkit to demonstrate:
1. **Pruning**: Unstructured L1 pruning of fully-connected layers in ResNet18.
2. **Quantization**: Dynamic quantization for efficient int8 inference.

Demo runs on CIFAR-10 with minimal training to showcase accuracy trade-offs.

## Dataset
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html):  
- 50,000 train  
- 10,000 test  

## Run
```bash
python main.py

