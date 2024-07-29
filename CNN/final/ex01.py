import torch
import torch.nn as nn
import numpy as np
import torchvision.datasets as dataset
import torchvision.transforms as transform
from torch.utils.data import DataLoader
from torchsummary import summary

# Training dataset 다운로드
cifar100_train = dataset.CIFAR100(root = "./", # 데이터셋을 저장할 위치
                            train = True,
                            transform = transform.ToTensor(),
                            download = True)
# Testing dataset 다운로드
cifar100_test = dataset.CIFAR100(root = "./",
                            train = False,
                            transform = transform.ToTensor(),
                            download = True)

class modelOne(nn.Module):
    def __init__(self):
        super(modelOne, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)

        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv2_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)

        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 100)

        # 파라미터를 가지지 않은 layer는 한 번만 선언해도 문제 없음
        self.relu = nn.ReLU()
        self.avgPool2d = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # convolution layers
        out = self.relu(self.conv1_1(x))
        out = self.relu(self.conv1_2(out))
        out = self.avgPool2d(out)

        out = self.relu(self.conv2_1(out))
        out = self.relu(self.conv2_2(out))
        out = self.avgPool2d(out)

        out = self.relu(self.conv3_1(out))
        out = self.relu(self.conv3_2(out))
        out = self.avgPool2d(out)

        # 평탄화
        out = out.reshape(-1, 4096)

        # fully connected layers
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)

        return out


model = modelOne().to('cuda')
summary(model, (3, 32, 32))
print(summary)
