import os
import numpy as np
import torch
from PIL import Image

from torch.utils.data import Dataset

from dataset_helpers import get_nine_crops, pirl_full_img_transform, pirl_stl10_jigsaw_patch_transform


class GetSTL10Data(Dataset):
    'Characterizes PyTorch Dataset object'
    def __init__(self, file_paths, labels, transform):
        'Initialization'
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.file_paths)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Select one file_path and convert to tensor object
        image = Image.open(self.file_paths[index])
        image_tensor = self.transform(image)
        label = self.labels[index]

        return image_tensor, label


class GetSTL10DataForPIRL(Dataset):
    'Characterizes PyTorch Dataset object'
    def __init__(self, file_paths):
        'Initialization'
        self.file_paths = file_paths

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.file_paths)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Select one file_path and convert to tensor object
        image = Image.open(self.file_paths[index])
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
        tensor_patches = torch.zeros(9, 3, 30, 30)
        for ind, patch in enumerate(permuted_patches_arr):
            patch_tensor = pirl_stl10_jigsaw_patch_transform(patch)
            tensor_patches[ind] = patch_tensor

        return [image_tensor, tensor_patches], index



if __name__ == '__main__':

    # Lets test the GetSTL10DataForPIRL class
    print("Testing for GetSTL10DataForPIRL")
    base_images_dir = 'stl10_data/unlabelled'
    file_names_list = os.listdir(base_images_dir)
    file_names_list = [file_name for file_name in file_names_list if file_name[-4:] == 'jpeg']

    file_paths_list = [os.path.join(base_images_dir, file_name) for file_name in file_names_list]
    ssl_dataset = GetSTL10DataForPIRL(file_paths_list)
    ssl_loader = torch.utils.data.DataLoader(ssl_dataset, batch_size=128, num_workers=8)

    dataiter = iter(ssl_loader)
    X, y = dataiter.__next__()
    print(X[0].size())
    print(X[1].size())
    print(y.size())
