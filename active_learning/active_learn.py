import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActiveLearning:
    def __init__(self, num_active_points):
        self.num_active_points = num_active_points

    def get_active_points(self, net_current, num_forwards, buildings_dataset, idx_pool):
        loss_iterations = []
        predictions = []
        criterion_item = torch.nn.CrossEntropyLoss(reduction='none')
        net_current.train()
        for i in range(num_forwards):
            with torch.no_grad():
                predicted = net_current(buildings_dataset.input_tensor[idx_pool][:, :-22])
                # print(predicted.shape, predicted)

            target = buildings_dataset.output_tensor[idx_pool]

            loss = criterion_item(predicted, target)
            predicted = F.softmax(predicted, dim=1)
            loss_iterations.append(loss)
            predictions.append(predicted)

        ## Dimensions: samples x iterations
        losses = torch.stack(loss_iterations, dim=1)

        ## Dimensions: samples x iterations x classes
        predicts = torch.stack(predictions, dim=1)
        # print(predicts.shape)

        # Compute predictive entropy
        avg_predicts = torch.mean(predicts, dim=1)
        eps = 1e-9
        avg_probs_clamped = torch.clamp(avg_predicts, min=eps)
        entropy = -torch.sum(avg_probs_clamped * torch.log2(avg_probs_clamped), dim=1)

        # Compute expected entropy
        prob_clamped = torch.clamp(predicts, min=eps)
        entropy_i = -torch.sum(prob_clamped * torch.log2(prob_clamped), dim=2)
        entropy_sum = entropy_i.sum(dim=1) / num_forwards

        # Compute mutual information
        mutual_info = entropy - entropy_sum

        # active points
        n_active_points = self.num_active_points
        mi_values, mi_indices = mutual_info.topk(n_active_points)
        selected_ind = mi_indices.tolist()
        selected_idx_pool = [idx_pool[i] for i in selected_ind]

        return selected_idx_pool
    
    def get_random_points(self, idx_pool):
        # create a list of random numbers as integers from 0 to len(idx_pool)
        n_active_points = self.num_active_points
        random_idx_pool = np.random.choice(idx_pool, n_active_points, replace=False)
        random_idx_pool = random_idx_pool.tolist()
        return random_idx_pool
         
