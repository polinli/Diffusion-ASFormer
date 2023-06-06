import torch
import random

def mask(video_features):
    mask_funcs = [no_masking, position_masking, boundary_masking]
    mask_func = random.choice(mask_funcs)
    return mask_func(video_features)

def no_masking(video_features):
    return video_features

def position_masking(video_features):
    num_frames, feature_dim = video_features.size()
    mask = torch.zeros(num_frames, feature_dim)
    masked_features = video_features * mask
    return masked_features

def boundary_masking(video_features, boundaries, distance=5):
    mask = torch.ones(num_frames, feature_dim)
    for boundary in boundaries:
        if boundary > distance and boundary < num_frames - distance:
            mask[boundary-distance:boundary+distance, :] = 0.0
    masked_features = video_features * mask
    return masked_features

def relation_masking(video_features, gt_actions):
    """
    didn't check if it works
    """
    """
    mask the segments belonging to a random action class.
    gt_actions: ground truth action labels for each segment
    """
    mask = torch.ones(num_frames, feature_dim)
    action_classes = torch.unique(gt_actions)
    random_action_class = action_classes[torch.randint(len(action_classes), (1,))]
    mask[gt_actions == random_action_class, :] = 0.0
    masked_features = video_features * mask
    return masked_features