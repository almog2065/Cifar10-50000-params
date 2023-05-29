import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import pickle
import io
from q1_206571135_train import CNN


def load_test(batch_size):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8,
                                              pin_memory=True)

    return test_loader


class pickle_cpu(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=lambda storage, loc: storage)
        else:
            return super().find_class(module, name)


def evaluate_model_q1():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cuda:0":
        model = pickle.load(open('model_q1.pkl', 'rb'))
    else:
        model = pickle_cpu(open('model_q1.pkl', 'rb')).load()

    model.to(device)
    model.eval()
    test_loader = load_test(batch_size=100)

    correct = 0
    total = 0
    for j, (images, labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images = images.to(device)
            labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images)
        outputs = nn.LogSoftmax(dim=1)(outputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    return 1 - correct / total


if __name__ == '__main__':
    print(f'The average error of the model: {evaluate_model_q1()}')
