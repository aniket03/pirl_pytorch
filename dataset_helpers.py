import os

from torchvision.transforms import transforms

def_train_transform_stl = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

hflip_data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

darkness_jitter_transform = transforms.Compose([
    transforms.ColorJitter(brightness=[0.5, 0.9]),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

lightness_jitter_transform = transforms.Compose([
    transforms.ColorJitter(brightness=[1.1, 1.5]),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

rotations_transform = transforms.Compose([
    transforms.RandomRotation(degrees=25),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

all_in_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=[0.5, 1.5]),
    transforms.RandomRotation(degrees=25),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

pirl_full_img_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

pirl_stl10_jigsaw_patch_transform = transforms.Compose([
    transforms.RandomCrop(30, padding=1),
    transforms.ColorJitter(brightness=[0.5, 1.5]),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def get_file_paths_n_labels(par_images_dir):
    """
    Returns all file paths for images in a directory (par_images_dir).
    The par_images_dir is supposed to only have sub directories with each sub-directory representing a label
    And in turn each sub-directory holding images pertaining to the label it represents
    """

    label_names = [dir_name for dir_name in os.listdir(par_images_dir)
                   if os.path.isdir(os.path.join(par_images_dir, dir_name))]
    label_dir_paths = [os.path.join(par_images_dir, dir_name) for dir_name in label_names]

    file_paths = []
    labels = []

    for label_name, label_dir in zip(label_names, label_dir_paths):
        file_names = [file_name for file_name in os.listdir(label_dir) if file_name[-4:]=='jpeg']
        file_paths += [os.path.join(label_dir, file_name) for file_name in file_names]
        labels += [int(label_name) - 1] * len(file_names)

    return file_paths, labels


def get_nine_crops(pil_image):
    """
    Get nine crops for a square pillow image. That is height and width of the image should be same.
    :param pil_image: pillow image
    :return: List of pillow images. The nine crops
    """
    w, h = pil_image.size
    diff = int(w/3)

    r_vals = [0, diff, 2 * diff]
    c_vals = [0, diff, 2 * diff]

    list_patches = []

    for r in r_vals:
        for c in c_vals:

            left = c
            top = r
            right = c + diff
            bottom = r + diff

            patch = pil_image.crop((left, top, right, bottom))
            list_patches.append(patch)

    return list_patches
