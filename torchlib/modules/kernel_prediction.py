from collections import OrderedDict

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
# from torch_scatter import scatter_add

from torch.autograd import Variable

from torchlib.image import crop_like


class ApplyKernels(nn.Module):
    """Gather values from tensor, weighted by kernels."""
    def __init__(self, ksize, normalization=None):
        super(ApplyKernels, self).__init__()
        self.ksize = ksize

        if normalization is None:
            assert normalization in ["l1", "sum"], "unknown normalization {}, should be `l1` or `sum`".format(normalization)
        self.normalization = normalization

    def forward(self, kernel, tensor):
        """
        @param kernel:  Kernels to apply to tensor. spatial dimensions are linearized.
        @type  kernel:  tensor, size [bs, k*k, h, w]

        @param tensor:  tensor on which to apply the kernels
        @type  tensor:  tensor, size [bs, c, h, w]

        @return:  a weighted reduction of `tensor` with weights in `kernels` and
                  the sum of weights (None is no normalization)
        @rtype :  tensor
        """

        kh, kw = kernel.shape[2:]
        bs, ci, h, w = tensor.shape
        ksize = self.ksize

        # Crop kernel and input so their sizes match
        needed = kh + ksize - 1
        if needed > h:
            crop = (needed - h) // 2
            if crop > 0:
                kernel = kernel[:, :, crop:-crop, crop:-crop]
            kh, kw = kernel.shape[2:]
        else:
            crop = (h - needed) // 2
            if crop > 0:
                tensor = tensor[:, :, crop:-crop, crop:-crop]

        # -------------------------------------------------------------------------
        # Vectorize the kernel tiles
        kernel = kernel.permute(0, 2, 3, 1)
        kernel = kernel.contiguous().view(bs, 1, kh, kw, ksize*ksize)

        # Split the input buffer in tiles matching the kernels
        tiles = tensor.unfold(2, ksize, 1).unfold(3, ksize, 1)
        tiles = tiles.contiguous().view(bs, ci, kh, kw, ksize*ksize)
        # -------------------------------------------------------------------------

        weighted_sum = th.sum(kernel*tiles, dim=4)

        if self.normalization == "sum":
            kernel_sum = th.sum(kernel, dim=4)
        elif self.normalization == "l1":
            kernel_sum = th.sum(th.abs(kernel), dim=4)
        elif self.normalization == "none":
            kernel_sum = None

        return weighted_sum, kernel_sum
