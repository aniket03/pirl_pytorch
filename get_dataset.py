import os
import numpy as np
import torch
from PIL import Image

from torch.utils.data import Dataset

from common_constants import PAR_DATA_DIR
from dataset_helpers import get_nine_crops, pirl_full_img_transform, pirl_jigsaw_patch_transform


class GetCIFARDataForPIRL(Dataset):
    'Characterizes PyTorch Dataset object'
    def __init__(self, data):
        'Initialization'
        self.data = data

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Select sample and convert to tensor object
        np_image = self.data[index]
        image = Image.fromarray(np_image)
        image_tensor = pirl_full_img_transform(image)

        # Get nine crops for the image
        nine_crops = get_nine_crops(image)

        # Form the jigsaw order for this image
        original_order = np.arange(9)
        permuted_order = np.copy(original_order)
        np.random.shuffle(permuted_order)

        # Permut the 9 patches obtained from the image
        permuted_patches_arr = [None] * 9
        for patch_pos, patch in zip(permuted_order, nine_crops):
            permuted_patches_arr[patch_pos] = patch

        # Apply data transforms
        # TODO: Remove hard coded values from here
        tensor_patches = torch.zeros(9, 3, 10, 10)
        for ind, patch in enumerate(permuted_patches_arr):
            patch_tensor = pirl_jigsaw_patch_transform(patch)
            tensor_patches[ind] = patch_tensor

        return [image_tensor, tensor_patches], index


if __name__ == '__main__':
    # Lets just test the GetCIFARDataForPIRL class

    list_data_batch_files = [
        os.path.join(PAR_DATA_DIR, 'cifar-10-batches-py/data_batch_{}'.format(batch_ind))
        for batch_ind in range(1, 6)
    ]

    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    data_batches = []
    for data_batch_file in list_data_batch_files:
        data_dict = unpickle(data_batch_file)
        data_batches.append(data_dict[b'data'])
    data = np.vstack(data_batches)
    data = data.reshape((data.shape[0], 3, 32, 32))
    data = data.transpose((0, 2, 3, 1))  # convert to h-w-channels format
    print (data.shape)

    train_dataset = GetCIFARDataForPIRL(data)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128,
                                               num_workers=8)

    # # Print sample batches that would be returned by the train_data_loader
    dataiter = iter(train_loader)
    X, y = dataiter.__next__()
    print (X[0].size())
    print (X[1].size())
    print (y.size())


