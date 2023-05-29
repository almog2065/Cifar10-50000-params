import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms.autoaugment import AutoAugmentPolicy
from torch.functional import F
import matplotlib.pyplot as plt
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(batch_size):
    transform_train_1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_train_2 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_train_3 = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_train_4 = transforms.Compose([
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_train_5 = transforms.Compose([
        transforms.AutoAugment(AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset_1 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_1)
    train_dataset_2 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_2)
    train_dataset_3 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_3)
    train_dataset_4 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_4)
    train_dataset_5 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_5)
    train_dataset = torch.utils.data.ConcatDataset([train_dataset_1, train_dataset_2])
    train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset_3])
    train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset_4])
    train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset_5])

    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8,
                                              pin_memory=True)

    return train_loader, test_loader


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(16))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=5, padding=2),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.AdaptiveAvgPool2d(2))
        self.fc = nn.Linear(128, 10)
        # self.fc1 = nn.Linear(85, 10)
        self.dropout = nn.Dropout(p=0.25)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        # out = self.dropout(out)
        out = self.fc(out)
        # out = nn.ReLU()(out)
        # out = self.fc1(out)
        return out


def train(epochs, model, train_loader, test_loader, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_train = []
    loss_test = []
    acc_train = []
    acc_test = []
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for epoch in range(epochs):
        model.train()
        curr_loss_train = 0
        curr_loss_test = 0

        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = images.to(device)
                labels = labels.to(device)

            # Forward + Backward + Optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            curr_loss_train += loss.item()
            outputs = nn.LogSoftmax(dim=1)(outputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            # curr_loss_train += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if (i + 1) % 100 == 0:
            #     print('Epoch [%d/%d], Iter [%d] Loss: %.4f'
            #           % (epoch + 1, epochs, i + 1, loss.data))
        acc_train.append(1 - correct / total)
        loss_train.append(curr_loss_train / i+1)

        model.eval()

        correct = 0
        total = 0

        for j, (images, labels) in enumerate(test_loader):
            if torch.cuda.is_available():
                images = images.to(device)
                labels = labels.to(device)

            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, labels)
            curr_loss_test += loss.item()
            outputs = nn.LogSoftmax(dim=1)(outputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        acc_test.append(1 - correct / total)
        loss_test.append(curr_loss_test / j+1)
        print('Test Accuracy of the model on the 10000 test images: %f %%' % (100 * (correct / total)))
        print(f'epoch : {epoch + 1}')
        if correct / total > 0.8:
            print("The end")
            with open("model_q1.pkl", "wb") as f:
                pickle.dump(model, f)
            return loss_train, acc_train, loss_test, acc_test


def plot_all(loss_train, acc_train, loss_test, acc_test):
    plt.plot(loss_train, label="train", color="red")
    plt.plot(loss_test, label="test", color="yellow")
    plt.title("loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss or error")
    plt.show()
    plt.plot(acc_train, label="train", color="red")
    plt.plot(acc_test, label="test", color="yellow")
    plt.title("error")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss or error")
    plt.show()


def main():
    num_epochs = 50
    batch_size = 100
    train_loader, test_loader = load_data(batch_size)
    cnn = CNN()
    print('number of parameters: ', sum(param.numel() for param in cnn.parameters()))

    if torch.cuda.is_available():
        cnn = cnn.to(device)

    loss_train, acc_train, loss_test, acc_test = train(num_epochs, cnn, train_loader, test_loader)
    plot_all(loss_train, acc_train, loss_test, acc_test)


if __name__ == '__main__':
    main()
