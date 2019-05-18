from copy import deepcopy

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


def get_data_loaders(seed, batch_size, num_train=13006, num_test=None):
    """Returns data loaders for MNIST binary classification task: 1 vs 7.
    
    The data loaders are formatted in a dict like this
    {
      "train": the train set,
      "neighbor": a dataset neighoring the train set,
      "test": the test set
    }
    and have train set of size num_train, and test set 10% as big unless otherwise
    specified by num_test.
  
    Note that all input images have been projected onto the unit ball.
    """
    # MNIST dataset (images and labels) projected onto unit ball
    project_to_unit_ball = lambda tensor: tensor / tensor.norm()
    transform = torchvision.transforms.Compose(
            (transforms.ToTensor(), project_to_unit_ball)
            )
    train_dataset = torchvision.datasets.MNIST(root='../../data', 
                                               train=True, 
                                               transform=transform,
                                               download=True)
  
    test_dataset = torchvision.datasets.MNIST(root='../../data', 
                                              train=False, 
                                              transform=transform)
  
    # make this a binary task by removing labels besides 1 and 7
    # 7 is considered the "positive" class in this task
    mask = (train_dataset.targets == 7) | (train_dataset.targets == 1)
    train_dataset.data = train_dataset.data[mask]
    train_dataset.targets = train_dataset.targets[mask]
    train_dataset.targets = (train_dataset.targets == 7).float()
    mask = (test_dataset.targets == 7) | (test_dataset.targets == 1)
    test_dataset.data = test_dataset.data[mask]
    test_dataset.targets = test_dataset.targets[mask]
    test_dataset.targets = (test_dataset.targets == 7).float()
  
    # add an attr that specifies input dim (will be helpful elsewhere in codebase)
    num_pixels = next(iter(train_dataset))[0].numel()
    for dset in train_dataset, test_dataset:
        setattr(dset, 'num_pixels', num_pixels)
  
    # subsampled MNIST
    num_test = num_test or int(.1 * num_train)  # handles None case
    if not num_train in range(0, 13007):
        raise ValueError("num_train must be between 1 and 13006")
    if not num_test in range(0, 2164):
        raise ValueError("num_test must be between 1 and 2163")
  
    # remove the second neighbor; will be swapped in for the first neighbor later
    rng = np.random.RandomState(seed)
    neighbor2_idx = rng.choice(np.arange(len(train_dataset)))
    neighbor2_img = train_dataset.data[neighbor2_idx]
    neighbor2_label = train_dataset.targets[neighbor2_idx]
    all_idx = torch.arange(len(train_dataset.data))
    train_dataset.data = train_dataset.data[all_idx != neighbor2_idx]
    train_dataset.targets = train_dataset.targets[all_idx != neighbor2_idx]
  
    # reduce down to desired number of examples
    train_idx = rng.choice(np.arange(num_train), num_train, False)
    test_idx = rng.choice(np.arange(num_test), num_test, False)
    train_dataset.data = train_dataset.data[train_idx]
    test_dataset.data = test_dataset.data[test_idx]
    train_dataset.targets = train_dataset.targets[train_idx]
    test_dataset.targets = test_dataset.targets[test_idx]
  
    # format the neighbor inputs/outputs as tuples
    neighbor1_idx = rng.choice(np.arange(len(train_dataset)))
    neighbor1 = (train_dataset.data[neighbor1_idx],
            train_dataset.targets[neighbor1_idx])
    neighbor2 = neighbor2_img, neighbor2_label
    neighbors = neighbor1, neighbor2  # return tuple of (img, target) pairs
  
    # carry out the swap
    neighbor_dataset = deepcopy(train_dataset)
    neighbor_dataset.data[neighbor1_idx] = neighbor2_img
    neighbor_dataset.targets[neighbor1_idx] = neighbor2_label
  
    # Data loader (input pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True)
  
    neighbor_loader = torch.utils.data.DataLoader(dataset=neighbor_dataset, 
                                                  batch_size=batch_size, 
                                                  shuffle=True)
  
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=batch_size, 
                                              shuffle=False)
  
    loaders = dict(train=train_loader, neighbor=neighbor_loader, test=test_loader) 
  
    return loaders, neighbors

