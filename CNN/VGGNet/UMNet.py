# 패키지 선언
import torch
import torch.nn as nn
import numpy as np
import torchvision.datasets as dataset
import torchvision.transforms as transform
from torch.utils.data import DataLoader

# Dataset 선언
# Training dataset 다운로드
cifar10_train = dataset.CIFAR10(root="./",  # 데이터셋을 저장할 위치
                                train=True,
                                transform=transform.ToTensor(),
                                download=True)

# Testing dataset 다운로드
cifar10_test = dataset.CIFAR10(root="./",
                               train=False,
                               transform=transform.ToTensor(),
                               download=True)

# CIFAR10 데이터셋 형상 확인
from matplotlib import pyplot as plt

print(len(cifar10_train))  # training dataset 개수 확인

first_data = cifar10_train[1]
print(first_data[0].shape)  # 두번째 data의 형상 확인
print(first_data[1])  # 두번째 data의 정답 확인

plt.imshow(first_data[0].permute(1, 2, 0))
plt.show()


# 모델 정의
class UMNet(nn.Module):
    def __init__(self):
        super(UMNet, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(in_channels=35, out_channels=32, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(in_channels=99, out_channels=128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

        # Skip connection을 위한 convolution layer 선언
        self.conv_skip1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv_skip2 = nn.Conv2d(in_channels=35, out_channels=64, kernel_size=3, padding=1)
        self.conv_skip3 = nn.Conv2d(in_channels=99, out_channels=256, kernel_size=3, padding=1)

        # 파라미터를 가지지 않은 layer는 한 번만 선언해도 문제 없음
        self.relu = nn.ReLU()
        self.MaxPool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        # Channel Attention 을 위한 convolution layer 선언
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.caconv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.caconv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Skip connection
        skip = self.relu(self.conv_skip1(x))
        out = self.relu(self.conv1_1(x))
        out = self.relu(self.conv1_2(out))

        out = out + skip

        # Dense connection
        out = torch.cat([x, out], dim=1)

        out = self.MaxPool2d(out)

        # Skip connection
        skip = self.relu(self.conv_skip2(out))
        out1 = self.relu(self.conv2_1(out))
        out1 = self.relu(self.conv2_2(out1))
        # Channel Attention (CA)
        weight = self.GAP(out1)
        weight = self.relu(self.caconv1(weight))
        weight = self.sigmoid(self.caconv2(weight))
        out1 = out1 * weight

        out1 = out1 + skip

        # Dense connection
        out1 = torch.cat([out, out1], dim=1)

        out1 = self.MaxPool2d(out1)
        # Skip connection
        skip = self.relu(self.conv_skip3(out1))
        out = self.relu(self.conv3_1(out1))
        out = self.relu(self.conv3_2(out))
        out = out + skip

        out = self.MaxPool2d(out)
        # 평탄화
        out = out.reshape(out.size(0), -1)

        # fully connected layers
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)

        return out


# Hyper-parameter 지정
batch_size = 100
learning_rate = 0.1
training_epochs = 20
loss_function = nn.CrossEntropyLoss()
network = UMNet()
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)
data_loader = DataLoader(dataset=cifar10_train,
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=True)

# 학습을 위한 반복문 진행
network = network.to('cuda:0')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)

    for img, label in data_loader:
        img = img.to('cuda:0')
        label = label.to('cuda:0')

        pred = network(img)

        loss = loss_function(pred, label)
        optimizer.zero_grad()  # gradient 초기화
        loss.backward()
        optimizer.step()

        avg_cost += loss / total_batch

    print('Epoch: %d Loss = %f' % (epoch + 1, avg_cost))

print('Learning finished')

# 정답률 확인
network = network.to('cpu')
with torch.no_grad():  # test에서는 기울기 계산 제외
    img_test = torch.tensor(np.transpose(cifar10_test.data, (0, 3, 1, 2))) / 255.
    label_test = torch.tensor(cifar10_test.targets)

    prediction = network(img_test)  # 전체 test data를 한번에 계산

    correct_prediction = torch.argmax(prediction, 1) == label_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())
