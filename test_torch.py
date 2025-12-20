import torch
import numpy as np

print("Torch version :", torch.__version__)
print("Numpy version :", np.__version__)

x = torch.randn(1, 1, 64, 64)
conv = torch.nn.Conv2d(1, 32, 3, padding=1)
y = conv(x)
print("Output shape :", y.shape)
