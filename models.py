import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50


class ClassificationResNet(nn.Module):

    def __init__(self, resnet_module, num_classes):
        super(ClassificationResNet, self).__init__()
        self.resnet_module = resnet_module
        self.fc = nn.Linear(512, num_classes)

    def forward(self, input_batch):

        # Data returned by data loaders is of the shape (batch_size, no_channels, h_patch, w_patch)
        resnet_feat_vectors = self.resnet_module(input_batch)
        final_feat_vectors = torch.flatten(resnet_feat_vectors, 1)
        x = F.log_softmax(self.fc(final_feat_vectors))

        return x


def get_base_resnet_module(model_type):
    """
    Returns the backbone network for required resnet architecture, specified as model_type
    :param model_type: Can be either of {res18, res34, res50}
    """

    if model_type == 'res18':
        original_model = resnet18(pretrained=False)
    elif model_type == 'res34':
        original_model = resnet34(pretrained=False)
    else:
        original_model = resnet50(pretrained=False)
    base_resnet_module = nn.Sequential(*list(original_model.children())[:-1])

    return base_resnet_module


def classifier_resnet(model_type, num_classes):
    """
    Returns a classification network with backbone belonging to the family of ResNets
    :param model_type: Specifies which resnet network to employ. Can be one of {res18, res34, res50}
    :param num_classes: The number of classes that the final network classifies it inputs into.
    """

    base_resnet_module = get_base_resnet_module(model_type)

    return ClassificationResNet(base_resnet_module, num_classes)


class PIRLResnet(nn.Module):
    def __init__(self, resnet_module, non_linear_head=False):
        super(PIRLResnet, self).__init__()
        self.resnet_module = resnet_module
        self.lin_project_1 = nn.Linear(512, 128)
        self.lin_project_2 = nn.Linear(128 * 9, 128)
        if non_linear_head:
            self.lin_project_3 = nn.Linear(128, 128)  # Will only be used if non_linear_head is True
        self.non_linear_head = non_linear_head

    def forward(self, i_batch, i_t_patches_batch):
        """
        :param i_batch: Batch of images
        :param i_t_patches_batch: Batch of transformed image patches (jigsaw transformation)
        """

        # Run I and I_t through resnet
        vi_batch = self.resnet_module(i_batch)
        vi_batch = torch.flatten(vi_batch, 1)
        vi_t_patches_batch = [self.resnet_module(i_t_patches_batch[:, patch_ind, :, :, :])
                              for patch_ind in range(9)]
        vi_t_patches_batch = [torch.flatten(vi_t_patches_batch[patch_ind], 1)
                              for patch_ind in range(9)]

        # Run resnet features for I and I_t via lin_project_1 layer
        vi_batch = self.lin_project_1(vi_batch)
        vi_t_patches_batch = [self.lin_project_1(vi_t_patches_batch[patch_ind])
                              for patch_ind in range(9)]

        # Concatenate together lin_project_1 outputs for patches of I_t
        vi_t_patches_concatenated = torch.cat(vi_t_patches_batch, 1)

        # Run concatenated feature vector for I_t through lin_project_2 layer
        vi_t_batch = self.lin_project_2(vi_t_patches_concatenated)

        # Run final feature vectors obtained for I and I_t through non-linearity (if specified)
        if self.non_linear_head:
            vi_batch = self.lin_project_3(F.relu(vi_batch))
            vi_t_batch = self.lin_project_3(F.relu(vi_t_batch))

        return vi_batch, vi_t_batch


def pirl_resnet(model_type, non_linear_head=False):
    """
    Returns a network which supports Pre-text invariant representation learning
    with backbone belonging to the family of ResNets
    :param model_type: Specifies which resnet network to employ. Can be one of {res18, res34, res50}
    :param non_linear_head: If true apply non-linearity to the output of function heads
    applied to resnet image representations
    """

    base_resnet_module = get_base_resnet_module(model_type)

    return PIRLResnet(base_resnet_module, non_linear_head)


if __name__ == '__main__':
    pr = pirl_resnet('res18', non_linear_head=True)  # non_linear_head can be True or False either.
    image_batch = torch.randn(32, 3, 64, 64)
    tr_img_patch_batch = torch.randn(32, 9, 3, 32, 32)

    result1, result2 = pr.forward(image_batch, tr_img_patch_batch)

    print (result1.size())
    print (result2.size())
