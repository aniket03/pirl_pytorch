import torch
import torch.nn.functional as F

import numpy as np

from torch.nn.utils import clip_grad_norm_

from pirl_loss import loss_pirl, get_img_pair_probs


def get_count_correct_preds(network_output, target):

    score, predicted = torch.max(network_output, 1)  # Returns max score and the index where max score was recorded
    count_correct = (target == predicted).sum().float()  # So that when accuracy is computed, it is not rounded to int

    return count_correct


def get_count_correct_preds_pretext(img_pair_probs_arr, img_mem_rep_probs_arr):
    """
    Get count of correct predictions for pre-text task
    :param img_pair_probs_arr: Prob vector of batch of images I and I_t to belong to same data distribution.
    :param img_mem_rep_probs_arr: Prob vector of batch of I and mem_bank_rep of I to belong to same data distribution
    """

    avg_probs_arr = (1/2) * (img_pair_probs_arr + img_mem_rep_probs_arr)
    count_correct = (avg_probs_arr >= 0.5).sum().float()  # So that when accuracy is computed, it is not rounded to int

    return count_correct.item()


class PIRLModelTrainTest():

    def __init__(self, network, device, model_file_path, all_images_mem, train_image_indices,
                 val_image_indices, count_negatives, temp_parameter, beta, only_train=False, threshold=1e-4):
        super(PIRLModelTrainTest, self).__init__()
        self.network = network
        self.device = device
        self.model_file_path = model_file_path
        self.threshold = threshold
        self.train_loss = 1e9
        self.val_loss = 1e9
        self.all_images_mem = torch.tensor(all_images_mem, dtype=torch.float).to(device)
        self.train_image_indices = train_image_indices.copy()
        self.val_image_indices = val_image_indices.copy()
        self.count_negatives = count_negatives
        self.temp_parameter = temp_parameter
        self.beta = beta
        self.only_train = only_train

    def train(self, optimizer, epoch, params_max_norm, train_data_loader, val_data_loader,
              no_train_samples, no_val_samples):
        self.network.train()
        train_loss, correct, cnt_batches = 0, 0, 0

        for batch_idx, (data_batch, batch_img_indices) in enumerate(train_data_loader):

            # Separate input image I batch and transformed image I_t batch (jigsaw patches) from data_batch
            i_batch, i_t_patches_batch = data_batch[0], data_batch[1]

            # Set device for i_batch, i_t_patches_batch and batch_img_indices
            i_batch, i_t_patches_batch = i_batch.to(self.device), i_t_patches_batch.to(self.device)
            batch_img_indices = batch_img_indices.to(self.device)

            # Forward pass through the network
            optimizer.zero_grad()
            vi_batch, vi_t_batch = self.network(i_batch, i_t_patches_batch)

            # Prepare memory bank of negatives for current batch
            np.random.shuffle(self.train_image_indices)
            mn_indices_all = np.array(list(set(self.train_image_indices) - set(batch_img_indices)))
            np.random.shuffle(mn_indices_all)
            mn_indices = mn_indices_all[:self.count_negatives]
            mn_arr = self.all_images_mem[mn_indices]

            # Get memory bank representation for current batch images
            mem_rep_of_batch_imgs = self.all_images_mem[batch_img_indices]

            # Get prob for I, I_t to belong to same data distribution.
            img_pair_probs_arr = get_img_pair_probs(vi_batch, vi_t_batch, mn_arr, self.temp_parameter)

            # Get prob for I and mem_bank_rep of I to belong to same data distribution
            img_mem_rep_probs_arr = get_img_pair_probs(vi_batch, mem_rep_of_batch_imgs, mn_arr, self.temp_parameter)

            # Compute loss => back-prop gradients => Update weights
            loss = loss_pirl(img_pair_probs_arr, img_mem_rep_probs_arr)
            loss.backward()

            clip_grad_norm_(self.network.parameters(), params_max_norm)
            optimizer.step()

            # Update running loss and no of pseudo correct predictions for epoch
            correct += get_count_correct_preds_pretext(img_pair_probs_arr, img_mem_rep_probs_arr)
            train_loss += loss.item()
            cnt_batches += 1

            # Update memory bank representation for images from current batch
            all_images_mem_new = self.all_images_mem.clone().detach()
            all_images_mem_new[batch_img_indices] = (self.beta * all_images_mem_new[batch_img_indices]) + \
                                                    ((1 - self.beta) * vi_batch)
            self.all_images_mem = all_images_mem_new.clone().detach()

            del i_batch, i_t_patches_batch, vi_batch, vi_t_batch, mn_arr, mem_rep_of_batch_imgs
            del img_mem_rep_probs_arr, img_pair_probs_arr

        train_loss /= cnt_batches

        if epoch % 10 == 0:
            torch.save(self.network.state_dict(), self.model_file_path + '_epoch_{}'.format(epoch))

        if self.only_train is False:
            val_loss, val_acc = self.test(epoch, val_data_loader, no_val_samples)

            if val_loss < self.val_loss - self.threshold:
                self.val_loss = val_loss
                torch.save(self.network.state_dict(), self.model_file_path)

        else:
            val_loss, val_acc = 0.0, 0.0

        train_acc = correct / no_train_samples

        print('\nAfter epoch {} - Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            epoch, train_loss, correct, no_train_samples, 100. * correct / no_train_samples))

        return train_loss, train_acc, val_loss, val_acc

    def test(self, epoch, test_data_loader, no_test_samples):

        self.network.eval()
        test_loss, correct, cnt_batches = 0, 0, 0

        for batch_idx, (data_batch, batch_img_indices) in enumerate(test_data_loader):

            # Separate input image I batch and transformed image I_t batch (jigsaw patches) from data_batch
            i_batch, i_t_patches_batch = data_batch[0], data_batch[1]

            # Set device for i_batch, i_t_patches_batch and batch_img_indices
            i_batch, i_t_patches_batch = i_batch.to(self.device), i_t_patches_batch.to(self.device)
            batch_img_indices = batch_img_indices.to(self.device)

            # Forward pass through the network
            vi_batch, vi_t_batch = self.network(i_batch, i_t_patches_batch)

            # Prepare memory bank of negatives for current batch
            np.random.shuffle(self.val_image_indices)

            mn_indices_all = np.array(list(set(self.val_image_indices) - set(batch_img_indices)))
            np.random.shuffle(mn_indices_all)
            mn_indices = mn_indices_all[:self.count_negatives]
            mn_arr = self.all_images_mem[mn_indices]

            # Get memory bank representation for current batch images
            mem_rep_of_batch_imgs = self.all_images_mem[batch_img_indices]

            # Get prob for I, I_t to belong to same data distribution.
            img_pair_probs_arr = get_img_pair_probs(vi_batch, vi_t_batch, mn_arr, self.temp_parameter)

            # Get prob for I and mem_bank_rep of I to belong to same data distribution
            img_mem_rep_probs_arr = get_img_pair_probs(vi_batch, mem_rep_of_batch_imgs, mn_arr, self.temp_parameter)

            # Compute loss
            loss = loss_pirl(img_pair_probs_arr, img_mem_rep_probs_arr)

            # Update running loss and no of pseudo correct predictions for epoch
            correct += get_count_correct_preds_pretext(img_pair_probs_arr, img_mem_rep_probs_arr)
            test_loss += loss.item()
            cnt_batches += 1

            # Update memory bank representation for images from current batch
            all_images_mem_new = self.all_images_mem.clone().detach()
            all_images_mem_new[batch_img_indices] = (self.beta * all_images_mem_new[batch_img_indices]) + \
                                                    ((1 - self.beta) * vi_batch)
            self.all_images_mem = all_images_mem_new.clone().detach()


            del i_batch, i_t_patches_batch, vi_batch, vi_t_batch, mn_arr, mem_rep_of_batch_imgs
            del img_mem_rep_probs_arr, img_pair_probs_arr

        test_loss /= cnt_batches
        test_acc = correct / no_test_samples
        print('\nAfter epoch {} - Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            epoch, test_loss, correct, no_test_samples, 100. * correct / no_test_samples))

        return  test_loss, test_acc


class ModelTrainTest():

    def __init__(self, network, device, model_file_path, threshold=1e-4):
        super(ModelTrainTest, self).__init__()
        self.network = network
        self.device = device
        self.model_file_path = model_file_path
        self.threshold = threshold
        self.train_loss = 1e9
        self.val_loss = 1e9

    def train(self, optimizer, epoch, params_max_norm, train_data_loader, val_data_loader,
              no_train_samples, no_val_samples):
        self.network.train()
        train_loss, correct, cnt_batches = 0, 0, 0

        for batch_idx, (data, target) in enumerate(train_data_loader):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            output = self.network(data)

            loss = F.nll_loss(output, target)
            loss.backward()

            clip_grad_norm_(self.network.parameters(), params_max_norm)
            optimizer.step()

            correct += get_count_correct_preds(output, target)
            train_loss += loss.item()
            cnt_batches += 1

            del data, target, output

        train_loss /= cnt_batches
        val_loss, val_acc = self.test(epoch, val_data_loader, no_val_samples)

        if val_loss < self.val_loss - self.threshold:
            self.val_loss = val_loss
            torch.save(self.network.state_dict(), self.model_file_path)

        train_acc = correct / no_train_samples

        print('\nAfter epoch {} - Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            epoch, train_loss, correct, no_train_samples, 100. * correct / no_train_samples))

        return train_loss, train_acc, val_loss, val_acc

    def test(self, epoch, test_data_loader, no_test_samples):
        self.network.eval()
        test_loss = 0
        correct = 0

        for batch_idx, (data, target) in enumerate(test_data_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss

            correct += get_count_correct_preds(output, target)

            del data, target, output

        test_loss /= no_test_samples
        test_acc = correct / no_test_samples
        print('\nAfter epoch {} - Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            epoch, test_loss, correct, no_test_samples, 100. * correct / no_test_samples))

        return  test_loss, test_acc

if __name__ == '__main__':
    img_pair_probs_arr = torch.randn((256,))
    img_mem_rep_probs_arr = torch.randn((256,))
    print (get_count_correct_preds_pretext(img_pair_probs_arr, img_mem_rep_probs_arr))