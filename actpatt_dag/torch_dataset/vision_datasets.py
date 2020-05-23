from torchvision import datasets, transforms

def get_MNIST(train):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    d = datasets.MNIST(
        root='dataset/',
        download=True,
        train=train,
        transform=transform)

    return d

def get_CIFAR10(train):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    d = datasets.CIFAR10(
        root='dataset/',
        download=True,
        train=train,
        transform=transform)

    return d