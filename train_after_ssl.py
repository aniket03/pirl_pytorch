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
from models import classifier_resnet, pirl_resnet
from network_helpers import copy_weights_between_models
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
    parser.add_argument('--experiment-name', type=str, default='e1_pirl_sup_')
    parser.add_argument('--pirl-model-name', type=str)
    args = parser.parse_args()

    # Identify device for holding tensors and carrying out computations
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define file path with trained SSL model and file_path where trained classification model
    # will be saved
    pirl_file_path = os.path.join(PAR_WEIGHTS_DIR, args.pirl_model_name)
    model_file_path = os.path.join(PAR_WEIGHTS_DIR, args.experiment_name)

    # Define train_set, val_set and test_set objects
    train_set = torchvision.datasets.CIFAR10(root=PAR_DATA_DIR, train=True,
                                            download=True, transform=def_train_transform)
    # val_set is created separately because it uses separate data_transform and sampling validation set
    # from train_set would have resulted in using data_transform used with train_set
    val_set = torchvision.datasets.CIFAR10(root=PAR_DATA_DIR, train=True,
                                           download=True, transform=def_test_transform)
    test_set = torchvision.datasets.CIFAR10(root=PAR_DATA_DIR, train=False, download=True,
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

    # Define model_to_train and inherit weights from pre-trained SSL model
    model_to_train = classifier_resnet(args.model_type, num_classes=num_outputs)
    pirl_model = pirl_resnet(args.model_type)
    pirl_model.load_state_dict(torch.load(pirl_file_path, map_location='cpu'))
    weight_copy_success = copy_weights_between_models(model_to_train, pirl_model)

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

    observations_df = pd.DataFrame()
    observations_df['epoch count'] = [i for i in range(1, args.epochs + 1)]
    observations_df['train loss'] = train_losses
    observations_df['val loss'] = val_losses
    observations_df['train acc'] = train_accs
    observations_df['val acc'] = val_accs
    observations_file_path = os.path.join(PAR_OBSERVATIONS_DIR, args.experiment_name + '_observations.csv')
    observations_df.to_csv(observations_file_path)
