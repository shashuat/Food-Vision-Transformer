import torch
import torchvision

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

train_dir = './data/train'
test_dir = './data/test'

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms