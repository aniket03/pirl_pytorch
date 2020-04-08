import os
import argparse

import torch
import torchvision

import numpy as np
import pandas as pd

from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import SubsetRandomSampler

from common_constants import PAR_WEIGHTS_DIR
from experiment_logger import log_experiment
from get_dataset import GetSTL10DataForPIRL
from models import pirl_resnet
from random_seed_setter import set_random_generators_seed
from train_test_helper import PIRLModelTrainTest


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


if __name__ == '__main__':

    # Training arguments
    parser = argparse.ArgumentParser(description='STL10 Train test script for PIRL task')
    parser.add_argument('--model-type', type=str, default='res18', help='The network architecture to employ as backbone')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay constant (default: 5e-4)')
    parser.add_argument('--tmax-for-cos-decay', type=int, default=70)
    parser.add_argument('--warm-start', type=bool, default=False)
    parser.add_argument('--count-negatives', type=int, default=6400,
                        help='No of samples in memory bank of negatives')
    parser.add_argument('--beta', type=float, default=0.5, help='Exponential running average constant'
                                                                'in memory bank update')
    parser.add_argument('--only-train', type=bool, default=False,
                        help='If true utilize the entire unannotated STL10 dataset for training.')
    parser.add_argument('--non-linear-head', type=bool, default=False,
                        help='If true apply non-linearity to the output of function heads '
                             'applied to resnet image representations')
    parser.add_argument('--temp-parameter', type=float, default=0.07, help='Temperature parameter in NCE probability')
    parser.add_argument('--cont-epoch', type=int, default=1, help='Epoch to start the training from, helpful when using'
                                                                  'warm start')
    parser.add_argument('--experiment-name', type=str, default='e1_resnet18_')
    args = parser.parse_args()

    # Set random number generation seed for all packages that generate random numbers
    set_random_generators_seed()

    # Identify device for holding tensors and carrying out computations
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the file_path where trained model will be saved
    model_file_path = os.path.join(PAR_WEIGHTS_DIR, args.experiment_name + '_epoch_100')

    # Get train_val image file_paths
    base_images_dir = 'stl10_data/unlabelled'
    file_names_list = os.listdir(base_images_dir)
    file_names_list = [file_name for file_name in file_names_list if file_name[-4:] == 'jpeg']
    file_paths_list = [os.path.join(base_images_dir, file_name) for file_name in file_names_list]

    # Define train_set, val_set objects
    train_set = GetSTL10DataForPIRL(file_paths_list)
    val_set = GetSTL10DataForPIRL(file_paths_list)

    # Define train and validation data loaders
    len_train_val_set = len(train_set)
    train_val_indices = list(range(len_train_val_set))
    np.random.shuffle(train_val_indices)

    if args.only_train is False:
        count_train = 70000
    else:
        count_train = 100000

    train_indices = train_val_indices[:count_train]
    val_indices = train_val_indices[count_train:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler,
                                               num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, sampler=val_sampler,
                                             num_workers=8)

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
    model_to_train = pirl_resnet(args.model_type, args.non_linear_head)

    # Set device on which training is done. Plus optimizer to use.
    model_to_train.to(device)
    sgd_optimizer = optim.SGD(model_to_train.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay_const)
    scheduler = CosineAnnealingLR(sgd_optimizer, args.tmax_for_cos_decay, eta_min=1e-4, last_epoch=-1)

    # Initialize model weights with a previously trained model if using warm start
    if args.warm_start and os.path.exists(model_file_path):
        model_to_train.load_state_dict(torch.load(model_file_path, map_location=device))

    # Start training
    all_images_mem = np.random.randn(len_train_val_set, 128)
    model_train_test_obj = PIRLModelTrainTest(
        model_to_train, device, model_file_path, all_images_mem, train_indices, val_indices, args.count_negatives,
        args.temp_parameter, args.beta, args.only_train
    )
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    for epoch_no in range(args.cont_epoch, args.cont_epoch + epochs):
        train_loss, train_acc, val_loss, val_acc = model_train_test_obj.train(
            sgd_optimizer, epoch_no, params_max_norm=4,
            train_data_loader=train_loader, val_data_loader=val_loader,
            no_train_samples=len(train_indices), no_val_samples=len(val_indices)
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        scheduler.step()

    # Log train-test results
    log_experiment(args.experiment_name, args.epochs, train_losses, val_losses, train_accs, val_accs)
