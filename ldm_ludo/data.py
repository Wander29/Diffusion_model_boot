

"""The `DiffusionDataset` function takes an image dataset, and for each image, 
it scales the pixel values between –1 and +1, then picks a random time step 
between 1 and _T_ and adds the corresponding noise, then it returns a tuple 
with the input and the target. The input is a `DiffusionSample`, and the target is 
the unscaled noise `eps`:"""

import torch
import torchvision
import torchvision.transforms.v2 as Tv2
from collections import namedtuple
from ldm_ludo import diff_model as dm

def loadDataset(dataset="mnist"):
    toTensor = Tv2.Compose([Tv2.ToImage(), Tv2.ToDtype(torch.float32, scale=True)])
    
    if dataset == "mnist":
        train_and_valid_data = torchvision.datasets.MNIST(
            root="datasets", train=True, download=True, transform=toTensor)
        test_data = torchvision.datasets.MNIST(
            root="datasets", train=False, download=True, transform=toTensor)
    elif dataset == "fashion_mnist":
        train_and_valid_data = torchvision.datasets.FashionMNIST(
            root="datasets", train=True, download=True, transform=toTensor)
        test_data = torchvision.datasets.FashionMNIST(
            root="datasets", train=False, download=True, transform=toTensor)

    torch.manual_seed(42)
    train_data, valid_data = torch.utils.data.random_split(
        train_and_valid_data, [0.9, 0.1])

    return train_data, valid_data, test_data

class DiffusionDataset:
    def __init__(self, dataset, T, alpha_bars):
        self.dataset = dataset
        self.T = T
        self.alpha_bars = alpha_bars

    def __getitem__(self, i):
        x0, _ = self.dataset[i]
        x0 = (x0 * 2) - 1  # scale from –1 to +1
        t = torch.randint(1, self.T + 1, size=[1])
        xt, eps = dm.forward_diffusion(self.alpha_bars, x0, t)
        return DiffusionSample(xt, t), eps

    def __len__(self):
        return len(self.dataset)

"""Each input is a `DiffusionSample` instance containing the noisy image, and the corresponding time step."""
class DiffusionSample(namedtuple("DiffusionSampleBase", ["xt", "t"])):
    def to(self, device):
        return DiffusionSample(self.xt.to(device), self.t.to(device))

def sanity_check(data_loader):
    """
        Take a look at a few training samples
        with the corresponding noise to predict, 
        and the original images, which we get by subtracting 
        the appropriately scaled noise from the 
        appropriately scaled noisy image`:
    """
    sample, eps = next(iter(data_loader))  # get the first batch
    plots.plot_sanity_check(alpha_bars, sample, eps, device)
