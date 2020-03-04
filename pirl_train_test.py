import os
import argparse

import torch
import torchvision

import numpy as np
import pandas as pd

from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import SubsetRandomSampler

from common_constants import PAR_DATA_DIR, PAR_WEIGHTS_DIR, PAR_OBSERVATIONS_DIR
from dataset_helpers import def_train_transform, def_test_transform
from get_dataset import GetCIFARDataForPIRL
from models import pirl_resnet
from train_test_helper import ModelTrainTest, PIRLModelTrainTest


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


if __name__ == '__main__':

    # Training arguments
    parser = argparse.ArgumentParser(description='CIFAR10 Train test script for PIRL')
    parser.add_argument('--model-type', type=str, default='res18', help='The network architecture to employ as backbone')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay constant (default: 5e-4)')
    parser.add_argument('--patience-for-lr-decay', type=int, default=10)
    parser.add_argument('--count-negatives', type=int, default=6400,
                        help='No of samples in memory bank of negatives')
    parser.add_argument('--beta', type=float, default=0.5, help='Exponential running average constant'
                                                                'in memory bank update')
    parser.add_argument('--temp-parameter', type=float, default=0.07, help='Temperature parameter in NCE probability')
    parser.add_argument('--experiment-name', type=str, default='e1_resnet18_')
    args = parser.parse_args()

    # Identify device for holding tensors and carrying out computations
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the file_path where trained model will be saved
    model_file_path = os.path.join(PAR_WEIGHTS_DIR, args.experiment_name)

    # Get train_val data
    list_data_batch_files = [
        os.path.join(PAR_DATA_DIR, 'cifar-10-batches-py/data_batch_{}'.format(batch_ind))
        for batch_ind in range(1, 6)
    ]

    data_batches = []
    for data_batch_file in list_data_batch_files:
        data_dict = unpickle(data_batch_file)
        data_batches.append(data_dict[b'data'])
    data = np.vstack(data_batches)
    data = data.reshape((data.shape[0], 3, 32, 32))
    data = data.transpose((0, 2, 3, 1))  # convert to h-w-channels format
    print ('Train-val data shape', data.shape)

    # Get Test data
    test_data_file = os.path.join(PAR_DATA_DIR, 'cifar-10-batches-py/test_batch')
    test_data_dict = unpickle(test_data_file)
    test_data = test_data_dict[b'data']
    test_data = test_data.reshape((test_data.shape[0], 3, 32, 32))
    test_data = test_data.transpose((0, 2, 3, 1))  # convert to h-w-channels format
    print('Test data shape', test_data.shape)

    # Define train_set, val_set and test_set objects
    train_set = GetCIFARDataForPIRL(data)
    val_set = GetCIFARDataForPIRL(data)
    test_set = GetCIFARDataForPIRL(test_data)

    # Define train, validation and test data loaders
    len_train_val_set = len(train_set)
    train_val_indices = list(range(len_train_val_set))
    np.random.shuffle(train_val_indices)

    count_train = 45000

    train_indices = train_val_indices[:count_train]
    val_indices = train_val_indices[count_train:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler,
                                               num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, sampler=val_sampler,
                                             num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # Print sample batches that would be returned by the train_data_loader
    dataiter = iter(train_loader)
    X, y = dataiter.__next__()
    print (X[0].size())
    print (X[1].size())
    print (y.size())

    # Train required model using data loaders defined above
    epochs = args.epochs
    lr = args.lr
    weight_decay_const = args.weight_decay

    # If using Resnet18
    model_to_train = pirl_resnet(args.model_type)

    # Set device on which training is done. Plus optimizer to use.
    model_to_train.to(device)
    sgd_optimizer = optim.SGD(model_to_train.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay_const)
    scheduler = ReduceLROnPlateau(sgd_optimizer, 'min', patience=args.patience_for_lr_decay,
                                  verbose=True, min_lr=1e-4)

    # Start training
    all_images_mem = np.random.randn(len_train_val_set, 128)
    model_train_test_obj = PIRLModelTrainTest(
        model_to_train, device, model_file_path, all_images_mem, train_indices, val_indices, args.count_negatives,
        args.temp_parameter, args.beta
    )
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    for epoch_no in range(epochs):
        train_loss, train_acc, val_loss, val_acc = model_train_test_obj.train(
            sgd_optimizer, epoch_no, params_max_norm=4,
            train_data_loader=train_loader, val_data_loader=val_loader,
            no_train_samples=len(train_indices), no_val_samples=len(val_indices)
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
    observations_file_path = os.path.join(PAR_OBSERVATIONS_DIR, args.experiment_name + '_observations.csv')
    observations_df.to_csv(observations_file_path)
