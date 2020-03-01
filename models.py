import torch
import torch.nn as nn
import torch.nn.functional as F

from base_resnet import ResNet, BasicBlock


class ClassificationResNet(nn.Module):

    def __init__(self, resnet_module, num_classes):
        super(ClassificationResNet, self).__init__()
        self.resnet_module = resnet_module
        self.fc = nn.Linear(512, num_classes)

    def forward(self, input_batch):

        # Data returned by data loaders is of the shape (batch_size, no_channels, h_patch, w_patch)
        final_feat_vectors = self.resnet_module(input_batch)
        x = F.log_softmax(self.fc(final_feat_vectors))

        return x


def classifier_resnet18(num_classes, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    base_resnet18 = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return ClassificationResNet(base_resnet18, num_classes)


class PIRLResnet(nn.Module):
    def __init__(self, resnet_module):
        super(PIRLResnet, self).__init__()
        self.resnet_module = resnet_module
        self.lin_project_1 = nn.Linear(512, 128)
        self.lin_project_2 = nn.Linear(128 * 9, 128)

    def forward(self, i_batch, i_t_patches_batch):
        """
        :param i_batch: Batch of images
        :param i_t_patches_batch: Batch of transformed image patches (jigsaw transformation)
        """

        # Run I and I_t through resnet
        vi_batch = self.resnet_module(i_batch)
        vi_t_patches_batch = [self.resnet_module(i_t_patches_batch[:, patch_ind, :, :, :])
                              for patch_ind in range(9)]

        # Run resnet features for I and I_t via lin_project_1 layer
        vi_batch = self.lin_project_1(vi_batch)
        vi_t_patches_batch = [self.lin_project_1(vi_t_patches_batch[patch_ind])
                              for patch_ind in range(9)]

        # Concatenate together lin_project_1 outputs for patches of I_t
        vi_t_patches_concatenated = torch.cat(vi_t_patches_batch, 1)

        # Run concatenated feature vector for I_t through lin_project_2 layer
        vi_t_batch = self.lin_project_2(vi_t_patches_concatenated)

        return vi_batch, vi_t_batch


def pirl_resnet18(**kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    base_resnet18 = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return PIRLResnet(base_resnet18)


if __name__ == '__main__':
    pr = pirl_resnet18()
    image_batch = torch.randn(32, 3, 64, 64)
    tr_img_patch_batch = torch.randn(32, 9, 3, 32, 32)

    result1, result2 = pr.forward(image_batch, tr_img_patch_batch)

    print (result1.size())
    print (result2.size())
