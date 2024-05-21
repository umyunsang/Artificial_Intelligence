import torch
import torch.nn as nn
import torchvision.datasets as dataset
import torchvision.transforms as transform
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Conv2d 실습
input_channel = 1

# 테스트 입력 텐서 생성
test_tensor = torch.rand(1, 1, 5, 5).to(device)  # Batch size, Channel, Height, Width
print(test_tensor.size())
print(test_tensor)


class testModel_channel1(nn.Module):
    def __init__(self):
        super(testModel_channel1, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x):
        y = self.conv1(x)
        return y


model = testModel_channel1().to(device)
out = model(test_tensor)
print(out.size())
print(out)

# input channel = 3, output channel = 64
test_tensor = torch.rand(1, 3, 32, 32).to(device)


class testModel_channel3(nn.Module):
    def __init__(self):
        super(testModel_channel3, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)

    def forward(self, x):
        y = self.conv1(x)
        return y


model = testModel_channel3().to(device)
out = model(test_tensor)
print(out.size())

# 3 Convolution layer
# in_channel = 1, out_channel = 32
# in_channel = 32, out_channel = 64
# in_channel = 64, out_channel = 128

test_tensor = torch.rand(1, 1, 32, 32).to(device)


class testModel_layer3(nn.Module):
    def __init__(self):
        super(testModel_layer3, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        y = self.pool(self.relu(self.conv3(x)))
        return y


model = testModel_layer3().to(device)
out = model(test_tensor)
print(out.size())

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
print(len(cifar10_train))  # training dataset 개수 확인

first_data = cifar10_train[1]
print(first_data[0].shape)  # 두번째 data의 형상 확인
print(first_data[1])  # 두번째 data의 정답 확인

plt.imshow(first_data[0].permute(1, 2, 0))
plt.show()


# CNN 모델 정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(64, 16, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.reshape(x, (-1, 5 * 5 * 16))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        y = self.fc3(x)
        return y


# Hyper-parameters 지정
batch_size = 100
learning_rate = 0.1
training_epochs = 30
loss_function = nn.CrossEntropyLoss()
network = CNN().to(device)
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)
data_loader = DataLoader(dataset=cifar10_train,
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=True)

# Perceptron 학습을 위한 반복문 선언
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)

    for img, label in data_loader:
        img = img.to(device)
        label = label.to(device)

        pred = network(img)
        loss = loss_function(pred, label)
        optimizer.zero_grad()  # gradient 초기화
        loss.backward()
        optimizer.step()
        avg_cost += loss / total_batch

    print('Epoch: %d Loss = %f' % (epoch + 1, avg_cost))

print('Learning finished')

# 학습이 완료된 모델을 이용해 정답률 확인
network = network.to('cpu')
with torch.no_grad():  # test에서는 기울기 계산 제외
    network.eval()
    img_test = torch.tensor(np.transpose(cifar10_test.data, (0, 3, 1, 2))) / 255
    label_test = torch.tensor(cifar10_test.targets)

    prediction = network(img_test)  # 전체 test data를 한번에 계산

    correct_prediction = torch.argmax(prediction, 1) == label_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())
# Weight parameter 저장하기/불러오기
# torch.save(network.state_dict(), "./cnn_cifar10.pth")

# 1. Pooling layer 변경: Average pooling  Max pooling
# 2. Convolutional layer channel 개수 변경: 6  32, 64