import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int

def cardinal_to_xy(cardinal_directions: Float[torch.Tensor, "batch 4 height width"])\
    -> Float[torch.Tensor, "batch 2 height width"]:
    """
    Converts the output of cardinal directions to a resultant vector field.

    Arguments:
        cardinal_directions (torch.Tensor): Tensor of shape [b, 4, h, w] representing the cardinal directions:
            - 0: up (+y)
            - 1: down (-y)
            - 2: left (-x)
            - 3: right (+x)
    
    Raises:
        AssertionError: If the input is not a tensor, does not have 4 dimensions or does not have 4 channels.

    Returns:
        torch.Tensor: Tensor of shape [b, 2, h, w] representing the resultant vector field:
            - 0: x-component (horizontal)
            - 1: y-component (vertical)
    """
    # 0. Input Handling
    assert isinstance(cardinal_directions, torch.Tensor), "Input must be a PyTorch tensor"
    assert cardinal_directions.ndim == 4, "Input tensor must of shape [b, 4, h, w]"
    assert cardinal_directions.shape[1] == 4, "Input tensor must have 4 channels for cardinal directions"

    # 1. Generate the resultant vector field
    up = cardinal_directions[:, 0, :, :]
    down = cardinal_directions[:, 1, :, :]
    left = cardinal_directions[:, 2, :, :]
    right = cardinal_directions[:, 3, :, :]

    # Compute the resultant x and y components
    x = right - left  # Horizontal component
    y = up - down  # Vertical component

    # Stack to form the resultant vector field [b, 2, h, w]
    vector_field = torch.stack((x, y), dim=1)
    return vector_field

def angular_field_loss(vector_field: Float[torch.Tensor, "batch 2 height width"],
                       target_vectors: Float[torch.Tensor, "batch 2"]) -> Float[torch.Tensor, "loss"]:
    """
    Computes the cosine similarity loss between a batched vector field and a batched target vector.

    Arguments:
        vector_field (torch.Tensor): Tensor of shape [b, 2, h, w], predicted vector field:
            - 0: x-component (horizontal)
            - 1: y-component (vertical)
        target_vectors (torch.Tensor): Tensor of shape [b, 2], per-input target vectors

    Raises:
        AssertionError: If the either input is not a tensor, there is a batch size mismatch between inputs,
            either input does not have 4 dimensions, or either input does not have 2 channels.

    Returns:
        float: Scalar cosine similarity loss
    """
    # 0. Input Handling
    assert isinstance(vector_field, torch.Tensor), "Input must be a PyTorch tensor"
    assert vector_field.ndim == 4, "Input tensor must of shape [b, 2, h, w]"
    assert vector_field.shape[1] == 2, "Input tensor must have 2 channels for x and y components"

    assert isinstance(target_vectors, torch.Tensor), "Target must be a PyTorch tensor"
    assert target_vectors.ndim == 2, "Target tensor must of shape [b, 2]"
    assert target_vectors.shape[1] == 2, "Target tensor must have 2 channels for x and y components"

    assert target_vectors.shape[0] == vector_field.shape[0],\
        "Batch sizes of vector field and target vectors must match"

    # 1. Normalize the target vectors and expand to match spatial dimensions
    target_vectors_norm = F.normalize(target_vectors[:, :, None, None], dim=1)  # [b, 2, 1, 1]
    target_vectors_norm_expanded = target_vectors_norm[:,:,None,None]  # [b, 2, 1, 1] -> [b, 2, h, w]

    # 3. Normalize the resultant field for cosine similarity calculation
    vector_field_norm = F.normalize(vector_field, dim=1)

    # 4. Cosine similarity calculation
    cos_sim = torch.sum(vector_field_norm * target_vectors_norm_expanded, dim=1)  # Shape: [b, h, w]

    # 5. Loss calculation (1 - mean similarity)
    loss = 1 - cos_sim.mean()

    return loss