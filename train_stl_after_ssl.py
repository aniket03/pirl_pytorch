import os
import argparse

import torch
import torchvision

import numpy as np
import pandas as pd

from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import SubsetRandomSampler

from common_constants import PAR_WEIGHTS_DIR
from dataset_helpers import def_train_transform_stl, def_test_transform, get_file_paths_n_labels, hflip_data_transform, \
    darkness_jitter_transform, lightness_jitter_transform, rotations_transform, all_in_transform
from experiment_logger import log_experiment
from get_dataset import GetSTL10Data
from models import classifier_resnet, pirl_resnet
from network_helpers import copy_weights_between_models
from random_seed_setter import set_random_generators_seed
from train_test_helper import ModelTrainTest

if __name__ == '__main__':

    # Training arguments
    parser = argparse.ArgumentParser(description='CIFAR10 Train test script')
    parser.add_argument('--model-type', type=str, default='res18',
                        help='The network architecture to employ as backbone')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay constant (default: 5e-4)')
    parser.add_argument('--patience-for-lr-decay', type=int, default=10)
    parser.add_argument('--full-fine-tune', type=bool, default=False)
    parser.add_argument('--experiment-name', type=str, default='e1_pirl_sup_')
    parser.add_argument('--pirl-model-name', type=str)
    args = parser.parse_args()

    # Set random number generation seed for all packages that generate random numbers
    set_random_generators_seed()

    # Identify device for holding tensors and carrying out computations
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define file path with trained SSL model and file_path where trained classification model
    # will be saved
    pirl_file_path = os.path.join(PAR_WEIGHTS_DIR, args.pirl_model_name)
    model_file_path = os.path.join(PAR_WEIGHTS_DIR, args.experiment_name)

    # Get train-val file paths and labels for STL10
    par_train_val_images_dir = './stl10_data/train'
    train_val_file_paths, train_val_labels = get_file_paths_n_labels(par_train_val_images_dir)
    print ('Train val file paths count', len(train_val_file_paths))
    print ('Train val labels count', len(train_val_labels))

    # Split file paths into train and val file paths
    len_train_val_set = len(train_val_file_paths)
    train_val_indices = list(range(len_train_val_set))
    np.random.shuffle(train_val_indices)

    count_train = 4200

    train_indices = train_val_indices[:count_train]
    val_indices = train_val_indices[count_train:]

    train_val_file_paths = np.array(train_val_file_paths)
    train_val_labels = np.array(train_val_labels)
    train_file_paths, train_labels = train_val_file_paths[train_indices], train_val_labels[train_indices]
    val_file_paths, val_labels = train_val_file_paths[val_indices], train_val_labels[val_indices]

    # Define train_set, and val_set objects
    train_set = ConcatDataset(
        [GetSTL10Data(train_file_paths, train_labels, def_train_transform_stl),
         GetSTL10Data(train_file_paths, train_labels, hflip_data_transform),
         GetSTL10Data(train_file_paths, train_labels, darkness_jitter_transform),
         GetSTL10Data(train_file_paths, train_labels, lightness_jitter_transform),
         GetSTL10Data(train_file_paths, train_labels, rotations_transform),
         GetSTL10Data(train_file_paths, train_labels, all_in_transform)]
    )

    # train_set = GetSTL10Data(train_val_file_paths, train_val_labels, all_in_transform)
    val_set = GetSTL10Data(val_file_paths, val_labels, def_test_transform)

    # Define train, validation and test data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=100, num_workers=8)

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

    # Define model_to_train and inherit weights from pre-trained SSL model
    model_to_train = classifier_resnet(args.model_type, num_classes=num_outputs)
    pirl_model = pirl_resnet(args.model_type)
    pirl_model.load_state_dict(torch.load(pirl_file_path, map_location=device))
    weight_copy_success = copy_weights_between_models(pirl_model, model_to_train)

    if not weight_copy_success:
        print ('Weight copy between SSL and classification net failed. Pls check !!')
        exit()

    # Freeze all layers except fully connected in classification net
    for name, param in model_to_train.named_parameters():
        if name[:7] == 'resnet_':
            param.requires_grad = False

    # # To test what is trainable status of each layer
    # for name, param in model_to_train.named_parameters():
    #     print (name, param.requires_grad)

    # Set device on which training is done. Plus optimizer to use.
    model_to_train.to(device)
    sgd_optimizer = optim.SGD(model_to_train.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay_const)
    scheduler = ReduceLROnPlateau(sgd_optimizer, 'min', patience=args.patience_for_lr_decay,
                                  verbose=True, min_lr=1e-5)

    # Start training
    model_train_test_obj = ModelTrainTest(model_to_train, device, model_file_path)
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

    # Log train-test results
    log_experiment(args.experiment_name + '_lin_clf', args.epochs, train_losses, val_losses, train_accs, val_accs)

    # Check if layers beyond last fully connected are to be fine tuned
    if args.full_fine_tune:
        for name, param in model_to_train.named_parameters():
            param.requires_grad = True

    # Reset optimizer and learning rate scheduler
    sgd_optimizer = optim.SGD(model_to_train.parameters(), lr=0.01, momentum=0.9, weight_decay=weight_decay_const)
    scheduler = ReduceLROnPlateau(sgd_optimizer, 'min', patience=args.patience_for_lr_decay,
                                  verbose=True, min_lr=1e-5)

    # Re-start training
    model_train_test_obj = ModelTrainTest(model_to_train, device, model_file_path)
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

    # Log train-test results
    log_experiment(args.experiment_name + '_full_ft', args.epochs, train_losses, val_losses, train_accs, val_accs)
