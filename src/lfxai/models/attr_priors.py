import torch

def total_var_prior_attr(attrs):
    """
    Pixel Attribution Prior: a Laplace prior zero mean on the differences between 
    attributions of neighbouring pixels, which promotes a high level of smoothness 
    in the attributions by minimizing their total variation. For more info, look 
    at the Improving performance of deep learning models with axiomatic attribution 
    priors and expected gradients, by Erio et al., https://arxiv.org/abs/1906.10670.
    """
    return (
        torch.sum(torch.abs(attrs[:, :, :, :-1] - attrs[:, :, :, 1:])) + 
        torch.sum(torch.abs(attrs[:, :, :-1, :] - attrs[:, :, 1:, :]))
    )