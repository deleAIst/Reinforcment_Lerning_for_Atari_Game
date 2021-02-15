import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class DQN(nn.Module):
    def __init__(self, img_shape: int, num_actions: int, learning_rate: float):
        super().__init__()
        self.img_shape = img_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.internal_model = self.build_model()

    def build_model(self):
        conv_1 = nn.Conv2d(self.img_shape, 32, 8, 4, 4)
        conv_2 = nn.Conv2d(32, 64, 4, 2, 2)
        conv_3 = nn.Conv2d(64, 64, 3, 1, 1)
        relu = nn.ReLU()
        line = nn.Linear(64, 256)
