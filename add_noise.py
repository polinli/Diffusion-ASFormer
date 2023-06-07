import torch
from torch import Tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def forward_process(one_hot_batch_target: Tensor, total_steps: Tensor, beta_start = 0.0001, beta_end = 0.04, beta_steps = 1000) -> Tensor:
    # simplified e forward process using the reparameterization trick
    # one_hot_batch_target shape: (N, C, L), add noise on all the elements

    # original beta in q(Xs|Xs-1)
    beta = torch.linspace(beta_start, beta_end, beta_steps) # all hyper-parameters

    # replace beta with alpha
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)[beta_steps-1] # get the last element from the cumprod

    # the ouput mean and variance of the Gaussian distribution
    mean = alpha_bar**0.5 * one_hot_batch_target # scalar
    deviation = (1-alpha_bar)**0.5               # scalar
    epsilon = torch.randn(one_hot_batch_target.size()).to(device) # tensor (gaussion noise)
    output = mean + deviation * epsilon             # corrupted action list

    # Normalize to [0, 1]
    noisy_action_list = (output - output.min()) / (output.max() - output.min())

    return noisy_action_list

def generate_noise(batch_target, num_classes):
    # generate pure gaussion noise with fixed random seed
    torch.manual_seed(19990526)
    noise = torch.randn(batch_target.shape[0], num_classes, batch_target.shape[1]) # (N, C, L)
    
    return noise