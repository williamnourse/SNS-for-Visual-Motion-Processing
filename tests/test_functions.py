import src.functions as functions
# import pytest
import torch

def test_cardinal_to_xy():
    batch_size = 1
    height = 2
    width = 2
    cardinal = torch.zeros([batch_size,4,height,width])
    up, down, left, right = 0, 1, 2, 3
    cardinal[:,up,:,:] = torch.tensor([[1,0],[0,0]])
    cardinal[:,down,:,:] = torch.tensor([[0,1],[0,0]])
    cardinal[:,left,:,:] = torch.tensor([[0,0],[1,0]])
    cardinal[:,right,:,:] = torch.tensor([[0,0],[0,1]])
    output = functions.cardinal_to_xy(cardinal)
    assert output.shape == (batch_size, 2, height, width), "Output shape mismatch"
    assert output[0,0,0,0] == 0, "Output value mismatch for x-component"#torch.tensor([[0,0],[-1,1]]), "Output value mismatch for x-component"
    assert output[0,0,0,1] == 0, "Output value mismatch for x-component"
    assert output[0,0,1,0] == -1, "Output value mismatch for x-component"
    assert output[0,0,1,1] == 1, "Output value mismatch for x-component"
    assert output[0,1,0,0] == 1, "Output value mismatch for y-component"#torch.tensor([[1,-1],[0,0]]), "Output value mismatch for y-component"
    assert output[0,1,0,1] == -1, "Output value mismatch for y-component"
    assert output[0,1,1,0] == 0, "Output value mismatch for y-component"
    assert output[0,1,1,1] == 0, "Output value mismatch for y-component"