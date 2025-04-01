import torch
import torch.nn as nn
from src.modules import LI, LI2
from src.functions import angular_field_loss
from tqdm import trange
import matplotlib.pyplot as plt


def LI_tester(input_size=3, height=10, width=10, num_steps=50, num_evals=1000):
    # Setup
    inp = torch.rand([num_steps, input_size, height, width], requires_grad=True)
    with torch.no_grad():
        net_target = LI(input_size)
        data_target = torch.zeros([num_steps, input_size, height, width])
        state = torch.zeros(1,input_size, height, width)
        for i in trange(num_steps):
            data_target[i,:,:,:], state = net_target(inp[i, :], state)

    # Training
    net = LI(input_size)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())
    loss_hist = torch.zeros(num_evals)
    loop = trange(num_evals)
    for i in loop:
        pred = torch.zeros([num_steps, input_size, height, width])
        state = torch.zeros(1, input_size, height, width)
        for step in range(num_steps):
            pred[step, :, :, :], state = net(inp[step, :], state)
        loss = loss_fn(pred, data_target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_hist[i] = loss.item()
        loop.set_postfix({'Loss': loss.item()})
    print(net_target.params['tau'])
    print(net.params['tau'])
    return loss_hist, pred, data_target

def LI2_tester(input_size=3, height=10, width=10, num_steps=50, num_evals=1000):
    # Setup
    inp = torch.rand([num_steps, input_size, height, width], requires_grad=True)
    with torch.no_grad():
        net_target = LI2(input_size)
        data_target = torch.zeros([num_steps, input_size, height, width])
        state0 = torch.zeros(1,input_size, height, width)
        state1 = torch.zeros(1,input_size, height, width)
        for i in trange(num_steps):
            data_target[i,:,:,:], state0, state1 = net_target(inp[i, :], state0, state1)

    # Training
    net = LI2(input_size)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())
    loss_hist = torch.zeros(num_evals)
    loop = trange(num_evals)
    for i in loop:
        pred = torch.zeros([num_steps, input_size, height, width])
        state0 = torch.zeros(1, input_size, height, width)
        state1 = torch.zeros(1, input_size, height, width)
        for step in range(num_steps):
            pred[step, :, :, :], state0, state1 = net(inp[step, :], state0, state1)
        loss = loss_fn(pred, data_target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_hist[i] = loss.item()
        loop.set_postfix({'Loss': loss.item()})
    for key in net_target.params.keys():
        print(key)
        print(net_target.params[key])
        print(net.params[key])
    return loss_hist, pred, data_target

def angular_loss_tester(direction, field_size=10, num_evals=5000):
    pred = torch.randn([1,4,field_size,field_size], requires_grad=True)
    optimizer = torch.optim.Adam([pred])
    loss_hist = torch.zeros(num_evals)
    pred_hist = torch.zeros(num_evals,4,field_size,field_size, requires_grad=False)
    loop = trange(num_evals)
    for i in loop:
        pred_hist[i,:,:,:] = pred
        loss = angular_field_loss(pred, direction)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_hist[i] = loss.item()
        loop.set_postfix({'Loss': loss.item()})
    return pred_hist, loss_hist


if __name__ == "__main__":
    test_LI = False
    test_LI2 = False
    test_angular_loss = True

    if test_LI:
        print('Testing LI:')
        loss_hist, pred, target = LI_tester()

        plt.figure()
        plt.plot(loss_hist)

    if test_LI2:
        print('Testing LI2:')
        loss_hist, pred, target = LI2_tester()

        plt.figure()
        plt.plot(loss_hist)

    if test_angular_loss:
        print('Testing angular loss:')
        direction = torch.tensor([[-1,-1]])
        pred_hist, loss_hist = angular_loss_tester(direction)
        flo_hist = cartesian_to_flo(pred_hist)
        plt.figure()
        plt.plot(loss_hist)
        plt.figure()
        plt.subplot(1,2,1)
        plt.quiver(flo_hist[0,0,:,:].detach(),flo_hist[0,1,:,:].detach())
        plt.subplot(1,2,2)
        plt.quiver(flo_hist[-1,0,:,:].detach(),flo_hist[-1,1,:,:].detach())

    plt.show()
