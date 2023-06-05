import torch

def add_noise(y0, s=1000, beta_start=0.0001, beta_end=0.02, beta_schedule='linear'):
    L, C = y0.shape
    epsilon = torch.randn(L, C)

    # Define beta schedule
    beta = torch.linspace(beta_start, beta_end, steps=s)

    # Compute alpha bar
    alpha_bar = torch.cumprod(1 - beta, dim=0)

    # Compute corrupted sequence Y_s
    y_s = torch.sqrt(alpha_bar[s-1]) * y0 + torch.sqrt(1 - alpha_bar[s-1]) * epsilon

    # Normalize action sequences
    y_s = 2 * (y_s - y_s.min()) / (y_s.max() - y_s.min()) - 1

    return y_s


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