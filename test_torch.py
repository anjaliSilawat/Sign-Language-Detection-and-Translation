import torch
import torchvision

print("Torch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)

# Simple tensor operation to test Torch
x = torch.tensor([1.0, 2.0, 3.0])
print("Tensor:", x)

# Test torchvision (shouldn't cause issues if only importing)
from torchvision import models
print("Torchvision models imported successfully.")
