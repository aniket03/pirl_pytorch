import os
import argparse

import torch
import torchvision

import numpy as np
import pandas as pd

from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import SubsetRandomSampler

from dataset_helpers import def_train_transform, def_test_transform
from resnet import resnet18
from train_test_helper import ModelTrainTest

if __name__ == '__main__':

    # Training arguments
    parser = argparse.ArgumentParser(description='CIFAR10 Train test script')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay constant (default: 5e-4)')
    parser.add_argument('--experiment-name', type=str, default='e1_resnet18_')
    args = parser.parse_args()

    # Identify device for holding tensors and carrying out computations
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the file_path where trained model will be saved
    model_file_path = os.path.join('weights/', args.experiment_name)

    # Define train_set, val_set and test_set objects
    train_set = torchvision.datasets.CIFAR10(root='./cifar_data', train=True,
                                            download=True, transform=def_train_transform)
    # val_set is created separately because it uses separate data_transform and sampling validation set
    # from train_set would have resulted in using data_transform used with train_set
    val_set = torchvision.datasets.CIFAR10(root='./cifar_data', train=True,
                                           download=True, transform=def_test_transform)
    test_set = torchvision.datasets.CIFAR10(root='./cifar_data', train=False, download=True,
                                           transform=def_test_transform)

    # Define train, validation and test data loaders
    len_train_val_set = len(train_set)
    train_val_indices = list(range(len_train_val_set))
    np.random.shuffle(train_val_indices)

    count_train = 45000

    train_indices = train_val_indices[:count_train]
    val_indices = train_val_indices[count_train:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, sampler=train_sampler,
                                               num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=100, sampler=val_sampler,
                                             num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=8)

    # Print sample batches that would be returned by the train_data_loader
    dataiter = iter(train_loader)
    X, y = dataiter.__next__()
    print (X.size())
    print (y.size())

    # Train required model using data loaders defined above
    num_outputs = 10
    epochs = args.epochs
    lr = args.lr
    weight_decay_const = args.weight_decay

    # If using Resnet18
    model_to_train = resnet18(num_classes=num_outputs, siamese_deg=None)

    # Set device on which training is done. Plus optimizer to use.
    model_to_train.to(device)
    sgd_optimizer = optim.SGD(model_to_train.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay_const)
    scheduler = ReduceLROnPlateau(sgd_optimizer, 'min', patience=2, verbose=True, min_lr=1e-5)

    # Start training
    model_train_test_obj = ModelTrainTest(model_to_train, device, model_file_path)
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    for epoch_no in range(epochs):
        train_loss, train_acc, val_loss, val_acc = model_train_test_obj.train(
            sgd_optimizer, epoch_no, params_max_norm=4,
            train_data_loader=train_loader, val_data_loader=val_loader
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        scheduler.step(val_loss)

    observations_df = pd.DataFrame()
    observations_df['epoch count'] = [i for i in range(1, args.epochs + 1)]
    observations_df['train loss'] = train_losses
    observations_df['val loss'] = val_losses
    observations_df['train acc'] = train_accs
    observations_df['val acc'] = val_accs
    observations_file_path = args.experiment_name + '_observations.csv'
    observations_df.to_csv(observations_file_path)




