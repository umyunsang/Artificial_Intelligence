import torch
import torch.nn as nn
import torchvision.datasets as dataset
import torchvision.transforms as transform
from torch.utils.data import DataLoader

# Training dataset 다운로드
mnist_train = dataset.MNIST(root="./",  # 데이터셋을 저장할 위치
                            train=True,
                            transform=transform.ToTensor(),
                            download=True)
# Testing dataset 다운로드
mnist_test = dataset.MNIST(root='./',
                           train=False,
                           transform=transform.ToTensor(),
                           download=True)


# Single Layer Perceptron 모델 정의
class SLP(nn.Module):
    def __init__(self):
        super(SLP, self).__init__()

        self.fc = nn.Linear(in_features=784, out_features=10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        y = self.fc(x)

        return y


# Hyper-parameters 지정
batch_size = 100
learning_rate = 0.1
training_epochs = 15
loss_function = nn.CrossEntropyLoss()
network = SLP()
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

data_loader = DataLoader(dataset=mnist_train,
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=True)

# Perceptron 학습을 위한 반복문 선언
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)

    for img, label in data_loader:
        pred = network(img)

        loss = loss_function(pred, label)
        optimizer.zero_grad()  # gradient 초기화
        loss.backward()
        optimizer.step()

        avg_cost += loss / total_batch

    print('Epoch: %d  LR: %f  Loss = %f' % (epoch + 1, optimizer.param_groups[0]['lr'], avg_cost))
    # scheduler.step()

print('Learning finished')

# 학습이 완료된 모델을 이용해 정답률 확인
with torch.no_grad():  # test에서는 기울기 계산 제외

    img_test = mnist_test.data.float()
    label_test = mnist_test.targets

    prediction = network(img_test)  # 전체 test data를 한번에 계산

    correct_prediction = torch.argmax(prediction, 1) == label_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())
