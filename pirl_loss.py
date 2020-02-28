import torch
import numpy as np


def get_img_pair_probs(vi_batch, vi_t_batch, all_images_mem, batch_img_indices, all_img_indices, temp_parameter):
    """
    Returns the probability that feature representation for image I and I_t belong to same distribution.
    :param vi_batch: Feature representation for batch of images I
    :param vi_t_batch: Feature representation for batch containing transformed versions of I.
    :param all_images_mem: Memory bank of feature representations for other images
    :param batch_img_indices: Indices of images present in provided batch
    :param all_img_indices: image indices of the images present in memory bank
    :param temp_parameter: The temperature parameter
    :return:
    """

    # Find images that will be used in memory bank of negatives
    mn_indices = list(set(all_img_indices) - set(batch_img_indices))[:6400]
    mn_arr = all_images_mem[mn_indices]

    # L2 normalize vi, vi_t and memory bank representations
    vi_norm_arr = torch.norm(vi_batch, dim=1, keepdim=True)
    vi_t_norm_arr = torch.norm(vi_t_batch, dim=1, keepdim=True)
    mn_norm_arr = torch.norm(mn_arr, dim=1, keepdim=True)

    vi_batch = vi_batch / vi_norm_arr
    vi_t_batch = vi_t_batch/ vi_t_norm_arr
    mn_arr = mn_arr / mn_norm_arr

    # Find cosine similarities
    sim_vi_vi_t_arr = (vi_batch @ vi_t_batch.t()).diagonal()
    sim_vi_t_mn_mat = (vi_t_batch @ mn_arr.t())

    # Fine exponentiation of similarity arrays
    exp_sim_vi_vi_t_arr = torch.exp(sim_vi_vi_t_arr / temp_parameter)
    exp_sim_vi_t_mn_mat = torch.exp(sim_vi_t_mn_mat / temp_parameter)

    # Sum exponential similarities of I_t with different images from memory bank of negatives
    sum_exp_sim_vi_t_mn_arr = torch.sum(exp_sim_vi_t_mn_mat, 1)

    # Find batch probabilities arr
    batch_prob_arr = exp_sim_vi_vi_t_arr / (exp_sim_vi_vi_t_arr + sum_exp_sim_vi_t_mn_arr)

    return batch_prob_arr


def loss_pirl(vi_batch, vi_t_batch, all_images_mem, batch_img_indices, all_img_indices, temp_parameter):
    img_pair_probs_arr = get_img_pair_probs(vi_batch, vi_t_batch, all_images_mem,
                                            batch_img_indices, all_img_indices, temp_parameter)

    neg_log_img_pair_probs = -1 * torch.log(img_pair_probs_arr)
    loss_i_i_t = torch.sum(neg_log_img_pair_probs) / neg_log_img_pair_probs.size()[0]

    mem_rep_in_batch_imgs = all_images_mem[batch_img_indices]
    img_mem_rep_probs_arr = get_img_pair_probs(vi_batch, mem_rep_in_batch_imgs, all_images_mem,
                                            batch_img_indices, all_img_indices, temp_parameter)
    neg_log_img_mem_rep_probs_arr = -1 * torch.log(img_mem_rep_probs_arr)
    loss_i_mem_i = torch.sum(neg_log_img_mem_rep_probs_arr) / neg_log_img_mem_rep_probs_arr.size()[0]

    loss = (loss_i_i_t + loss_i_mem_i) / 2

    return  loss


if __name__ == '__main__':
    # Test get_img_pair_probs function
    vi_batch = torch.randn(256, 128)
    vi_t_batch = torch.randn(256, 128)
    all_images_mem = torch.randn(50000, 128)
    batch_img_indices = np.arange(25000, 25000 + 256)
    all_images_indices = np.arange(50000)
    temp_parameter = 1.5
    prob_vector = get_img_pair_probs(vi_batch, vi_t_batch, all_images_mem, batch_img_indices,
                                     all_images_indices, temp_parameter)
    print (prob_vector.shape)

    loss_val = loss_pirl(
        vi_batch, vi_t_batch, all_images_mem, batch_img_indices, all_images_indices, temp_parameter
    )

    print (loss_val)


