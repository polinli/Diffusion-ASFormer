import torch
from torch import Tensor

def forward_process(batch_target: Tensor, total_steps: Tensor, beta_start = 0.0001, beta_end = 0.04, beta_steps = 1000) -> Tensor:
    # simplified e forward process using the reparameterization trick
    # batch_target shape: (N, L), add noise on all the elements

    # original beta in q(Xs|Xs-1)
    beta = torch.linspace(beta_start, beta_end, beta_steps) # all hyper-parameters

    # replace beta with alpha
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)[beta_steps-1] # get the last element from the cumprod

    # the ouput mean and variance of the Gaussian distribution
    mean = alpha_bar**0.5 * batch_target
    deviation = (1-alpha_bar)**0.5
    eps = torch.randn_like(batch_target) # generate gaussion noise

    # corrupted action list. And normalize to [0, 1]
    corrupted_actions = mean + deviation * eps
    corrupted_actions = (output - output.min()) / (output.max() - output.min())

    return corrupted_actions

def generate_noise(batch_size, length):
    # generate pure gaussion noise with fixed random seed
    torch.manual_seed(19990526)
    noise = torch.randn(batch_size, length)
    
    return noise