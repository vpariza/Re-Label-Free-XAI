from itertools import product

import numpy as np
import torch


def generate_masks(attr: np.ndarray, mask_size: int, is_normalised: bool = False) -> torch.Tensor:
    """
    Generates mask for images with feature importance scores
    Args:
        attr: feature importance scores
        mask_size: number of pixels masked
        # Extension idea
        is_normalised: boolean flag to set importance of pixel i with (1 - f_i)/max(f)
    Returns:
        Mask hiding most important pixels

    """
    dataset_size, n_chanels, H, W = attr.shape
    attr = torch.from_numpy(
        np.sum(np.abs(attr), axis=1, keepdims=True)
    )  # Sum the attribution over the channels
    masks = torch.ones(attr.shape)
    masks = masks.view(
        dataset_size, -1
    )  # Reshape to make it compatible with torch.topk

    # torch.topk[0] -> returns top values, torch.topk[1] -> returns indices of top values
    # top_pixels = torch.topk(attr.view(dataset_size, -1), mask_size)[1]

    top_attrs = torch.topk(attr.view(dataset_size, -1), mask_size)
    top_pixel_values, top_pixels = top_attrs
    if not is_normalised:
        for feature_id, example_id in product(range(mask_size), range(dataset_size)):
            masks[example_id, top_pixels[example_id, feature_id]] = 0
    else:
        for example_id in range(dataset_size):
            max_feat = max(top_pixel_values[example_id])
            for feature_id in range(mask_size):
                feat = top_pixel_values[example_id, feature_id]
                masks[example_id, top_pixels[example_id, feature_id]] = (1-feat)/max_feat
    masks = masks.view(dataset_size, 1, H, W)
    return masks


def generate_tseries_masks(attr: np.ndarray, mask_size: int) -> torch.Tensor:
    """
    Generates mask for time series with feature importance scores
    Args:
        attr: feature importance scores
        mask_size: number of time steps masked

    Returns:
        Mask hiding most important time steps

    """
    dataset_size, time_steps, n_chanels = attr.shape
    attr = torch.from_numpy(
        np.sum(np.abs(attr), axis=-1, keepdims=True)
    )  # Sum the attribution over the channels
    masks = torch.ones(attr.shape)
    masks = masks.view(
        dataset_size, -1
    )  # Reshape to make it compatible with torch.topk
    top_time_steps = torch.topk(attr.view(dataset_size, -1), mask_size)[1]
    for feature_id, example_id in product(range(mask_size), range(dataset_size)):
        masks[example_id, top_time_steps[example_id, feature_id]] = 0
    masks = masks.view(dataset_size, time_steps, n_chanels)
    return masks