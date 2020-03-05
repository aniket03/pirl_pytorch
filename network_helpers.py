import torch

from models import pirl_resnet, classifier_resnet


def test_copy_weights(m1, m2):
    """
    Tests that weights copied from m1 into m2, are actually refected in m2
    """
    m1_state_dict = m1.state_dict()
    m2_state_dict = m2.state_dict()
    weight_copy_flag = 1
    for name, param in m1_state_dict.items():
        if name in m2_state_dict:
            if not torch.all(torch.eq(param.data, m2_state_dict[name].data)):
                print("Something is incorrect for layer {} in 2nd model", name)
                weight_copy_flag = 0

    if weight_copy_flag:
        print('All is well')

    return 1


def copy_weights_between_models(m1, m2):
    """
    Copy weights for layers common between m1 and m2.
    From m1 => m2
    """

    # Load state dictionaries for m1 model and m2 model
    m1_state_dict = m1.state_dict()
    m2_state_dict = m2.state_dict()

    # Set the m2 model's weights with trained m1 model weights
    for name, param in m1_state_dict.items():
        if name not in m2_state_dict:
            continue
        else:
            m2_state_dict[name] = param.data
    m2.load_state_dict(m2_state_dict)

    # Test that model m2 **really** has got updated weights
    return test_copy_weights(m1, m2)


if __name__ == '__main__':

    pr = pirl_resnet('res18')
    cr = classifier_resnet('res18', num_classes=10)

    copy_success = copy_weights_between_models(pr, cr)

