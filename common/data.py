import torch
import torchvision
import torchvision.transforms.v2 as Tv2

def loadDataset(dataset="mnist", dataset_dir="datasets"):
    toTensor = Tv2.Compose([
        Tv2.ToImage(), 
        Tv2.ToDtype(torch.float32, scale=True)]
    )
    
    if dataset == "mnist":
        train_and_valid_data = torchvision.datasets.MNIST(
            root=dataset_dir, train=True, download=True, transform=toTensor)
        test_data = torchvision.datasets.MNIST(
            root=dataset_dir, train=False, download=True, transform=toTensor)
    elif dataset == "fashion_mnist":
        train_and_valid_data = torchvision.datasets.FashionMNIST(
            root=dataset_dir, train=True, download=True, transform=toTensor)
        test_data = torchvision.datasets.FashionMNIST(
            root=dataset_dir, train=False, download=True, transform=toTensor)

    train_data, valid_data = torch.utils.data.random_split(
        train_and_valid_data, [0.9, 0.1])

    return train_data, valid_data, test_data