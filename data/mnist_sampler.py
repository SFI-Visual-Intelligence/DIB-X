import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import paths
import torchvision

class SampleMnistData28():
    def __init__(self):
        self.path_to_echograms = paths.path_to_echograms()
    def for_train(self):
        mnist_tr = torchvision.datasets.MNIST(self.path_to_echograms, train=True,
                                              transform=torchvision.transforms.ToTensor(), download=True)
        return mnist_tr

    def for_val(self):
        mnist_val = torchvision.datasets.MNIST(self.path_to_echograms, train=False,
                                              transform=torchvision.transforms.ToTensor(), download=True)
        return mnist_val

    def for_test(self):
        mnist_te = torchvision.datasets.MNIST(self.path_to_echograms, train=False,
                                              transform=torchvision.transforms.ToTensor(), download=True)
        return mnist_te
