import src.functions as functions
import pytest
import torch

def test_cardinal_to_xy_input_tensor():
    """
    Test that the function raises an error when the input is not a tensor.
    """
    with pytest.raises(AssertionError):
        functions.cardinal_to_xy('Not a tensor')
    valid_tensor = torch.zeros([1, 4, 2, 2])
    functions.cardinal_to_xy(valid_tensor)

def test_cardinal_to_xy_tensor_shape():
    """
    Test that the function raises an error when the input tensor does not have 4 dimensions.
    """
    with pytest.raises(AssertionError):
        functions.cardinal_to_xy(torch.zeros([1, 4, 2]))
    with pytest.raises(AssertionError):
        functions.cardinal_to_xy(torch.zeros([1, 4]))
    with pytest.raises(AssertionError):
        functions.cardinal_to_xy(torch.zeros([1]))
    valid_tensor = torch.zeros([1, 4, 2, 2])
    functions.cardinal_to_xy(valid_tensor)

def test_cardinal_to_xy_channel_shape():
    """
    Test that the function raises an error when the input tensor does not have 4 channels.
    """
    with pytest.raises(AssertionError):
        functions.cardinal_to_xy(torch.zeros([1, 3, 2, 2]))
    with pytest.raises(AssertionError):
        functions.cardinal_to_xy(torch.zeros([1, 5, 2, 2]))
    valid_tensor = torch.zeros([1, 4, 2, 2])
    functions.cardinal_to_xy(valid_tensor)

def test_cardinal_to_xy_output_shape():
    """
    Test that the function returns a tensor of the correct shape.
    """
    valid_input = torch.zeros([1, 4, 2, 2])
    output = functions.cardinal_to_xy(valid_input)
    assert output.shape == (1, 2, 2, 2), "Output shape mismatch"

def test_cardinal_to_xy_output_values():
    """
    Test that the function returns the expected output values.
    """
    # Define a batch of cardinal directions
    batch_size = 1
    height = 2
    width = 2
    cardinal = torch.zeros([batch_size, 4, height, width])
    
    # Set the cardinal directions
    up, down, left, right = 0, 1, 2, 3
    cardinal[:, up, :, :] = torch.tensor([[1, 0], [0, 0]])
    cardinal[:, down, :, :] = torch.tensor([[0, 1], [0, 0]])
    cardinal[:, left, :, :] = torch.tensor([[0, 0], [1, 0]])
    cardinal[:, right, :, :] = torch.tensor([[0, 0], [0, 1]])
    
    # Call the function and get the output
    output = functions.cardinal_to_xy(cardinal)
    
    # Check if the output matches the expected values
    assert output[0,0,0,0] == 0, "Output value mismatch for x-component"#torch.tensor([[0,0],[-1,1]]), "Output value mismatch for x-component"
    assert output[0,0,0,1] == 0, "Output value mismatch for x-component"
    assert output[0,0,1,0] == -1, "Output value mismatch for x-component"
    assert output[0,0,1,1] == 1, "Output value mismatch for x-component"
    assert output[0,1,0,0] == 1, "Output value mismatch for y-component"#torch.tensor([[1,-1],[0,0]]), "Output value mismatch for y-component"
    assert output[0,1,0,1] == -1, "Output value mismatch for y-component"
    assert output[0,1,1,0] == 0, "Output value mismatch for y-component"
    assert output[0,1,1,1] == 0, "Output value mismatch for y-component"

def test_angular_field_loss_input_tensor():
    """
    Test that the function raises an error when either input is not a tensor.
    """
    valid_field_tensor = torch.zeros([1, 2, 2, 2])
    valid_target_tensor = torch.zeros([1, 2])
    with pytest.raises(AssertionError):
        functions.angular_field_loss('Not a tensor', valid_target_tensor)
    with pytest.raises(AssertionError):
        functions.angular_field_loss(valid_field_tensor, 'Also not a tensor')
    functions.angular_field_loss(valid_field_tensor, valid_target_tensor)

def test_angular_field_loss_batch_size_mismatch():
    """
    Test that the function raises an error when there is a batch size mismatch between inputs.
    """
    valid_field_tensor = torch.zeros([1, 2, 2, 2])
    invalid_target_tensor = torch.zeros([2, 2])
    with pytest.raises(AssertionError):
        functions.angular_field_loss(valid_field_tensor, invalid_target_tensor)
    valid_target_tensor = torch.zeros([1, 2])
    functions.angular_field_loss(valid_field_tensor, valid_target_tensor)

def test_angular_field_loss_tensor_shape():
    """
    Test that the function raises an error when either input does not have 4 dimensions.
    """
    valid_field_tensor = torch.zeros([1, 2, 2, 2])
    valid_target_tensor = torch.zeros([1, 2])
    with pytest.raises(AssertionError):
        functions.angular_field_loss(torch.zeros([1, 2, 2]), valid_target_tensor)
    with pytest.raises(AssertionError):
        functions.angular_field_loss(valid_field_tensor, torch.zeros([1, 2, 2]))
    functions.angular_field_loss(valid_field_tensor, valid_target_tensor)

def test_angular_field_loss_channel_shape():
    """
    Test that the function raises an error when either input does not have 2 channels.
    """
    valid_field_tensor = torch.zeros([1, 2, 2, 2])
    valid_target_tensor = torch.zeros([1, 2])
    with pytest.raises(AssertionError):
        functions.angular_field_loss(torch.zeros([1, 3, 2, 2]), valid_target_tensor)
    with pytest.raises(AssertionError):
        functions.angular_field_loss(valid_field_tensor, torch.zeros([1, 3]))
    functions.angular_field_loss(valid_field_tensor, valid_target_tensor)

def test_angular_field_loss_output_shape():
    """
    Test that the function returns a scalar loss.
    """
    valid_field_tensor = torch.zeros([1, 2, 2, 2])
    valid_target_tensor = torch.zeros([1, 2])
    output = functions.angular_field_loss(valid_field_tensor, valid_target_tensor)
    assert output.shape == torch.Size([]), "Output must be a scalar"

def test_angular_field_loss_output_value():
    """"
    Test that the function returns the expected output value.
    """
    valid_field_tensor = torch.ones([1, 2, 2, 2])
    valid_target_tensor = torch.ones([1, 2])
    output = functions.angular_field_loss(valid_field_tensor, valid_target_tensor)
    torch.allclose(output, torch.zeros_like(output), atol=1e-7), "Tensor is not close enough to zero"

