from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np
import scipy
import torch
from torch import nn
from torch.nn import functional as F



def hungarian_algorithm(cost_matrix: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batch-applies the hungarian algorithm to find a matching that minimizes
    the overall cost. Returns the matching indices as a LongTensor with shape
    (batch size, 2, min(num objects, num slots)). The first column is the row
    indices (the indices of the true objects) while the second column is the
    column indices (the indices of the slots). The row indices are always in
    ascending order, while the column indices are not necessarily. The outputs
    are on the same device as `cost_matrix` but gradients are detached.
    A small example:
                | 4, 1, 3 |
                | 2, 0, 5 |
                | 3, 2, 2 |
                | 4, 0, 6 |
    would result in selecting elements (1,0), (2,2) and (3,1). Therefore, the
    row indices will be [1,2,3] and the column indices will be [0,2,1].
    Args:
        cost_matrix: Tensor of shape (batch size, num objects, num slots).
    Returns:
        A tuple containing:
            - a Tensor with shape (batch size, min(num objects, num slots))
                with the costs of the matches.
            - a LongTensor with shape (batch size, 2, min(num objects,
                num slots)) containing the indices for the resulting matching.

    From https://github.com/addtt/object-centric-library/utils/slot_matching.py.
    """

    # List of tuples of size 2 containing flat arrays
    raw_indices = list(
        map(scipy.optimize.linear_sum_assignment, cost_matrix.cpu().detach().numpy())
    )
    indices = torch.tensor(
        np.array(raw_indices), device=cost_matrix.device, dtype=torch.long
    )
    smallest_cost_matrix = torch.stack(
        [
            cost_matrix[i][indices[i, 0], indices[i, 1]]
            for i in range(cost_matrix.shape[0])
        ]
    )
    return smallest_cost_matrix.to(cost_matrix.device), indices


def calculate_distances(labels, data, size):
    batch_size, _, _ = labels.shape
    
    labels_reshaped = labels.unsqueeze(1).repeat(1, size, 1, 1).reshape(-1, labels.size(2))
    data_reshaped = data.unsqueeze(2).repeat(1, 1, size, 1).reshape(-1, data.size(2))

    bce_losses = F.binary_cross_entropy(data_reshaped, labels_reshaped, reduction='none')

    cost_matrix = bce_losses.view(batch_size, size, size, -1).mean(dim=-1)

    return cost_matrix