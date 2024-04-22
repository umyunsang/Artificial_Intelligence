# 패키지 선언
import torch
import torch.nn as nn
import torchvision.datasets as dataset
import torchvision.transforms as transform
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

# 데이터셋 준비
mnist_train = dataset.MNIST(root="./", train=True, transform=transform.ToTensor(), download=True)
mnist_test = dataset.MNIST(root="./", train=False, transform=transform.ToTensor(), download=True)

# 데이터셋 확인
print(len(mnist_train))  # 훈련 데이터셋의 샘플 개수를 출력합니다.
first_data = mnist_train[0]  # 첫 번째 데이터를 가져옵니다.
print(first_data[0].shape)  # 이미지의 형태를 출력합니다.
print(first_data[1])  # 해당 이미지의 레이블을 출력합니다.
plt.imshow(first_data[0][0, :, :], cmap='gray')  # 이미지를 시각화합니다.
plt.show()

# 이미지 전처리
first_img = first_data[0]
print(first_img.shape)  # 이미지의 형태를 출력합니다.
first_img = first_img.view(-1, 28 * 28)  # 이미지를 1차원으로 평탄화합니다.
print(first_img.shape)  # 변환된 이미지의 형태를 출력합니다.

# Single Layer Perceptron (SLP) 모델 정의
class SLP(nn.Module):
    def __init__(self):
        super(SLP, self).__init__()
        self.fc = nn.Linear(in_features=784, out_features=10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        y = self.fc(x)
        return y

# Hyper-parameters 설정
batch_size = 100
learning_rate = 0.1
training_epochs = 15
loss_function = nn.CrossEntropyLoss()  # 손실 함수로 Cross Entropy Loss를 사용합니다.
network = SLP()  # 앞에서 정의한 SLP 모델을 인스턴스화합니다.
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)  # SGD 옵티마이저를 설정합니다.

# DataLoader 설정
data_loader = DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)

# 훈련 반복문
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)

    for img, label in data_loader:
        pred = network(img)  # 모델에 이미지를 전달하여 예측값을 계산합니다.
        loss = loss_function(pred, label)  # 손실을 계산합니다.
        optimizer.zero_grad()  # 기울기 초기화
        loss.backward()  # 역전파를 통해 기울기 계산
        optimizer.step()  # 옵티마이저로 모델 파라미터 업데이트

        avg_cost += loss / total_batch  # 평균 손실을 계산합니다.

    print('Epoch: %d Loss = %f' % (epoch + 1, avg_cost))  # 에폭마다 평균 손실을 출력합니다.

print('Learning finished')  # 학습 완료 메시지 출력

# 학습된 모델을 이용한 정확도 확인
with torch.no_grad():  # 기울기 계산 제외
    img_test = mnist_test.data.float()  # 테스트 데이터를 float 형태로 변환합니다.
    label_test = mnist_test.targets  # 테스트 데이터의 레이블을 가져옵니다.

    prediction = network(img_test)  # 테스트 데이터에 대한 예측을 계산합니다.
    correct_prediction = torch.argmax(prediction, 1) == label_test  # 정확하게 예측한 경우를 계산합니다.
    accuracy = correct_prediction.float().mean()  # 정확도를 계산합니다.
    print('Accuracy:', accuracy.item())  # 정확도를 출력합니다.

# 학습된 모델의 가중치 저장
torch.save(network.state_dict(), "pth/slp_mnist.pth")

# 결과값
# Epoch: 15 Loss = 0.274448
# Learning finished
# Accuracy: 0.892300009727478