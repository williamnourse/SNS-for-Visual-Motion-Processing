import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int

def cardinal_to_xy(cardinal_directions: Float[torch.Tensor, "batch 4 height width"])\
    -> Float[torch.Tensor, "batch 2 height width"]:
    """
    Converts the output of cardinal directions to a resultant vector field.
    Args:
    :param cardinal_directions: Tensor of shape [b, 4, h, w] representing the cardinal directions:
        - 0: up (+y)
        - 1: down (-y)
        - 2: left (-x)
        - 3: right (+x)
    
    Returns:
    :return: Tensor of shape [b, 2, h, w] representing the resultant vector field:
        - 0: x-component (horizontal)
        - 1: y-component (vertical)
    """
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

def angular_field_loss(cardinal_field: Float[torch.Tensor, "batch 2 height width"],
                       target_vectors: Float[torch.Tensor, "batch 2"]) -> Float[torch.Tensor, "loss"]:
    """
    Computes the cosine similarity loss between a batched vector field and a batched target vector.

    Args:
    - output: Tensor of shape [b, 4, h, w] -> predicted cardinal direction magnitudes:
        - 0: up (+y)
        - 1: down (-y)
        - 2: left (-x)
        - 3: right (+x)
    - target_vectors: Tensor of shape [b, 2] -> per-input target vectors

    Returns:
    - loss: Scalar cosine similarity loss
    """
    b, _, h, w = cardinal_field.shape

    resultant_field = cardinal_to_xy(cardinal_field)

    # 2. Normalize the target vectors and expand to match spatial dimensions
    # target_vectors = F.normalize(target_vectors[:, :, None, None], dim=1)  # [b, 2, 1, 1]
    target_vectors = target_vectors[:,:,None,None]

    # 3. Normalize the resultant field for cosine similarity calculation
    norm_resultant = F.normalize(resultant_field, dim=1)

    # 4. Cosine similarity calculation
    cos_sim = torch.sum(norm_resultant * target_vectors, dim=1)  # Shape: [b, h, w]

    # 5. Loss calculation (1 - mean similarity)
    loss = 1 - cos_sim.mean()

    return loss