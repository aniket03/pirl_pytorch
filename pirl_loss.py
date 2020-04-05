import torch
import numpy as np


def get_img_pair_probs(vi_batch, vi_t_batch, mn_arr, temp_parameter):
    """
    Returns the probability that feature representation for image I and I_t belong to same distribution.
    :param vi_batch: Feature representation for batch of images I
    :param vi_t_batch: Feature representation for batch containing transformed versions of I.
    :param mn_arr: Memory bank of feature representations for negative images for current batch
    :param temp_parameter: The temperature parameter
    """

    # Define constant eps to ensure training is not impacted if norm of any image rep is zero
    eps = 1e-6

    # L2 normalize vi, vi_t and memory bank representations
    vi_norm_arr = torch.norm(vi_batch, dim=1, keepdim=True)
    vi_t_norm_arr = torch.norm(vi_t_batch, dim=1, keepdim=True)
    mn_norm_arr = torch.norm(mn_arr, dim=1, keepdim=True)

    vi_batch = vi_batch / (vi_norm_arr + eps)
    vi_t_batch = vi_t_batch/ (vi_t_norm_arr + eps)
    mn_arr = mn_arr / (mn_norm_arr + eps)

    # Find cosine similarities
    sim_vi_vi_t_arr = (vi_batch @ vi_t_batch.t()).diagonal()
    sim_vi_t_mn_mat = (vi_t_batch @ mn_arr.t())

    # Fine exponentiation of similarity arrays
    exp_sim_vi_vi_t_arr = torch.exp(sim_vi_vi_t_arr / temp_parameter)
    exp_sim_vi_t_mn_mat = torch.exp(sim_vi_t_mn_mat / temp_parameter)

    # Sum exponential similarities of I_t with different images from memory bank of negatives
    sum_exp_sim_vi_t_mn_arr = torch.sum(exp_sim_vi_t_mn_mat, 1)

    # Find batch probabilities arr
    batch_prob_arr = exp_sim_vi_vi_t_arr / (exp_sim_vi_vi_t_arr + sum_exp_sim_vi_t_mn_arr + eps)

    return batch_prob_arr


def loss_pirl(img_pair_probs_arr, img_mem_rep_probs_arr):
    """
    Returns the average of [-log(prob(img_pair_probs_arr)) - log(prob(img_mem_rep_probs_arr))]
    :param img_pair_probs_arr: Prob vector of batch of images I and I_t to belong to same data distribution.
    :param img_mem_rep_probs_arr: Prob vector of batch of I and mem_bank_rep of I to belong to same data distribution
    """

    # Get 1st term of loss
    neg_log_img_pair_probs = -1 * torch.log(img_pair_probs_arr)
    loss_i_i_t = torch.sum(neg_log_img_pair_probs) / neg_log_img_pair_probs.size()[0]

    # Get 2nd term of loss
    neg_log_img_mem_rep_probs_arr = -1 * torch.log(img_mem_rep_probs_arr)
    loss_i_mem_i = torch.sum(neg_log_img_mem_rep_probs_arr) / neg_log_img_mem_rep_probs_arr.size()[0]

    loss = (loss_i_i_t + loss_i_mem_i) / 2

    return  loss


if __name__ == '__main__':
    # Test get_img_pair_probs function
    vi_batch = torch.randn(256, 128)
    vi_t_batch = torch.randn(256, 128)
    mn_arr = torch.randn(6400, 128)
    mem_rep_of_batch_imgs = torch.randn(256, 128)
    temp_parameter = 1.5

    # Prob vector between I and I_t
    img_pair_probs_arr = get_img_pair_probs(vi_batch, vi_t_batch, mn_arr, temp_parameter)
    print (img_pair_probs_arr.shape)

    # Prob vector between I and mem bank representation of I
    img_mem_rep_probs_arr = get_img_pair_probs(vi_batch, mem_rep_of_batch_imgs, mn_arr, temp_parameter)
    print (img_mem_rep_probs_arr.shape)

    # Final loss
    loss_val = loss_pirl(img_pair_probs_arr, img_mem_rep_probs_arr)

    print (loss_val)


