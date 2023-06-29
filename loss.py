import torch
import torch.nn as nn
import torch.nn.functional as F

def temporal_smoothness_loss(P_s):
    P_s = torch.clamp(P_s, min=1e-7) # avoid log(0) and negative values
    diff = torch.diff(torch.log(P_s), dim=-1) # log likelihood difference between adjacent frames
    diff_sq = torch.square(diff)
    mean_diff_sq = torch.mean(diff_sq, dim=[1, 2]) # mean over frames and classes
    mean_loss = torch.mean(mean_diff_sq) # mean over batch

    return mean_diff_sq

def boundary_alignment_loss(Y_0, P_s):
    # Compute ground truth boundary sequence
    B = (Y_0[:-1] != Y_0[1:]).float()

    # Smooth B to obtain B_bar using a Gaussian filter
    B_bar = ...  # Fill with your specific code

    # Compute boundary probabilities for denoised sequence
    denoised_boundary_prob = 1 - torch.sum(P_s[:-1] * P_s[1:], dim=-1)

    # Compute binary cross-entropy loss
    return F.binary_cross_entropy(denoised_boundary_prob, B_bar)

def calculate_total_loss(ps, batch_target, mask, num_classes):
    cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-100)

    loss = 0
    for p in ps:
        p_reformat = p.transpose(2, 1).contiguous().view(-1, num_classes) # shape: (N, C, L) => (N*L, C)
        batch_target_reformat = batch_target.view(-1) # shape: (N, L) => (N*L)

        loss += cross_entropy_loss(p_reformat, batch_target_reformat)
        loss += temporal_smoothness_loss(p).squeeze()
        

    return loss
