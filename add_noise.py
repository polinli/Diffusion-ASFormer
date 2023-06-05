import torch

def forwar_process(batch_target: Tensor, total_steps: Tensor, beta_start = 0.0001, beta_end = 0.04, beta_steps) -> Tensor:
    # simplified e forward process using the reparameterization trick
    # batch_target shape: (N, L), add noise on all the elements

    # original beta in q(Xs|Xs-1)
    beta = torch.linspace(beta_start, beta_end, beta_steps) # all hyper-parameters

    # replace beta with alpha
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    # the ouput mean and variance of the Gaussian distribution
    mean = gather(alpha_bar, total_steps)**0.5 * batch_target
    deviation = (1-gather(alpha_bar, total_steps))**0.5
    eps = torch.randn_like(batch_target)

    return mean + deviation * eps

def generate_noise(batch_size, length):
    # generate pure gaussion noise with fixed random seed
    torch.manual_seed(19990526)
    noise = torch.randn(batch_size, length)
    
    return noise