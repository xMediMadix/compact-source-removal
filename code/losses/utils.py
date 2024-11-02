"""Utility functions for loss computations, including the Gram matrix for style loss."""

import torch


def gram_matrix(feat):
    """Compute the Gram matrix for style loss.

    This code is based on:
    https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py

    Args:
        feat (torch.Tensor): Input feature map from a layer of the network.

    Returns:
        torch.Tensor: The computed Gram matrix.
    """
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram
