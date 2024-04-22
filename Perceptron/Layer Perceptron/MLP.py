import torch
import torch.nn as nn
import torchvision.datasets as dataset
import torchvision.transforms as transform
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 2. 데이터셋 준비
mnist_train = dataset.MNIST(root="./", train=True, transform=transform.ToTensor(), download=True)
mnist_test = dataset.MNIST(root="./", train=False, transform=transform.ToTensor(), download=True)

# 5. Multi Layer Perceptron (MLP) 모델 정의
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        y = self.sigmoid(self.fc1(x))
        y = self.fc2(y)
        return y

# 6. 하이퍼파라미터 설정
batch_size = 100
learning_rate = 0.1
training_epochs = 15
loss_function = nn.CrossEntropyLoss()
network = MLP()
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)

# 7. DataLoader 설정
data_loader = DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)

# 8. 훈련 반복문
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)

    for img, label in data_loader:
        pred = network(img)
        loss = loss_function(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_cost += loss / total_batch

    print('Epoch: %d Loss = %f' % (epoch + 1, avg_cost))

print('Learning finished')

# 9. 학습된 모델을 이용한 정확도 확인
with torch.no_grad():
    img_test = mnist_test.data.float()
    label_test = mnist_test.targets

    prediction = network(img_test)
    correct_prediction = torch.argmax(prediction, 1) == label_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

# 10. 학습된 모델의 가중치 저장
torch.save(network.state_dict(), "pth/mlp_mnist.pth")


# 결과 값
# Epoch: 15 Loss = 0.193479
# Learning finished
# Accuracy: 0.9437999725341797
