from motion_vision_net import VisionNetNoField, VisionNet
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
import argparse
from motion_data import ClipDataset
from datetime import datetime

def run_sample(sample, net: nn.Module):
    """
    Process a training sequence of frames through the network
    :param sample: sequence of 30 frames
    :param net: network to be trained
    :return: the average value over the sample of the ccw neuron, net
    """
    (state_input, state_bo_input, state_bo_fast, state_bo_slow, state_bo_output, state_lowpass,
    state_bf_input, state_bf_fast, state_bf_slow, state_bf_output, state_enhance_on, state_direct_on,
    state_suppress_on, state_enhance_off, state_direct_off, state_suppress_off, state_ccw_on, state_cw_on,
    state_ccw_off, state_cw_off, state_hc) = net.init()
    net.setup()
    # net.zero_grad()
    num_sub_steps = 13
    warmup = 200
    niter = num_sub_steps*(sample.shape[0])+warmup
    step = 0

    inp = torch.zeros([state_input.shape[0]*state_input.shape[1],niter])
    bo_input = torch.zeros([state_direct_on.shape[0] * state_direct_on.shape[1], niter])
    bo_fast = torch.zeros([state_direct_on.shape[0] * state_direct_on.shape[1], niter])
    bo_slow = torch.zeros([state_direct_on.shape[0] * state_direct_on.shape[1], niter])
    bo_out = torch.zeros([state_direct_on.shape[0] * state_direct_on.shape[1], niter])
    low = torch.zeros([state_direct_on.shape[0] * state_direct_on.shape[1], niter])
    enhance = torch.zeros([state_direct_on.shape[0] * state_direct_on.shape[1], niter])
    direct = torch.zeros([state_direct_on.shape[0] * state_direct_on.shape[1], niter])
    suppress = torch.zeros([state_direct_on.shape[0] * state_direct_on.shape[1], niter])
    cw = torch.zeros([state_cw_on.shape[0] * state_cw_on.shape[1], niter])
    ccw = torch.zeros([state_ccw_on.shape[0] * state_ccw_on.shape[1], niter])
    hc = torch.zeros([niter,2])
    for i in range(warmup):  # warmup
        (state_input, state_bo_input, state_bo_fast, state_bo_slow, state_bo_output, state_lowpass,
                state_bf_input, state_bf_fast, state_bf_slow, state_bf_output, state_enhance_on, state_direct_on,
                state_suppress_on, state_enhance_off, state_direct_off, state_suppress_off, state_ccw_on, state_cw_on,
                state_ccw_off, state_cw_off, state_hc) = net(
            sample[0,:,:], state_input, state_bo_input, state_bo_fast, state_bo_slow, state_bo_output, state_lowpass,
                state_bf_input, state_bf_fast, state_bf_slow, state_bf_output, state_enhance_on, state_direct_on,
                state_suppress_on, state_enhance_off, state_direct_off, state_suppress_off, state_ccw_on, state_cw_on,
                state_ccw_off, state_cw_off, state_hc)

        inp[:, step] = state_input.flatten()
        bo_input[:, step] = state_bf_input.flatten()
        bo_fast[:, step] = state_bf_fast.flatten()
        bo_slow[:, step] = state_bf_slow.flatten()
        bo_out[:, step] = state_bf_output.flatten()
        low[:, step] = state_lowpass.flatten()
        enhance[:, step] = state_enhance_off.flatten()
        direct[:, step] = state_direct_off.flatten()
        suppress[:, step] = state_suppress_off.flatten()
        cw[:, step] = state_cw_off.flatten()
        ccw[:, step] = state_ccw_off.flatten()
        hc[step,:] = state_hc
        step += 1

    for i in range(sample.shape[0]):
        for j in range(num_sub_steps):
            # print(torch.max(sample))

            # print('Sample %i Step %i'%(i,step))
            (state_input, state_bo_input, state_bo_fast, state_bo_slow, state_bo_output, state_lowpass,
             state_bf_input, state_bf_fast, state_bf_slow, state_bf_output, state_enhance_on, state_direct_on,
             state_suppress_on, state_enhance_off, state_direct_off, state_suppress_off, state_ccw_on, state_cw_on,
             state_ccw_off, state_cw_off, state_hc) = net(
                sample[i, :, :], state_input, state_bo_input, state_bo_fast, state_bo_slow, state_bo_output,
                state_lowpass,
                state_bf_input, state_bf_fast, state_bf_slow, state_bf_output, state_enhance_on, state_direct_on,
                state_suppress_on, state_enhance_off, state_direct_off, state_suppress_off, state_ccw_on, state_cw_on,
                state_ccw_off, state_cw_off, state_hc)

            inp[:, step] = state_input.flatten()
            bo_input[:, step] = state_bf_input.flatten()
            bo_fast[:, step] = state_bf_fast.flatten()
            bo_slow[:, step] = state_bf_slow.flatten()
            bo_out[:, step] = state_bf_output.flatten()
            low[:, step] = state_lowpass.flatten()
            enhance[:, step] = state_enhance_off.flatten()
            direct[:, step] = state_direct_off.flatten()
            suppress[:, step] = state_suppress_off.flatten()
            cw[:, step] = state_cw_off.flatten()
            ccw[:, step] = state_ccw_off.flatten()
            hc[step, :] = state_hc

            state_check = state_cw_on
            if torch.any(torch.isnan(state_check)).item():
                print(torch.min(state_check).item(), torch.max(state_check).item())
            # if torch.any(state_check>0):
            #     print(state_check)

            step += 1

    return inp, bo_input, bo_fast, bo_slow, bo_out, low, enhance, direct, suppress, ccw, cw, hc

def plot_state(state, target):
    plt.subplot(2, 2, i + 1)
    plt.imshow(state, cmap='gray', vmin=0, vmax=1)
    plt.title(str(target.item()))

params = {'dt': 1/(30*13)*1000, 'device': 'cpu'}
plot = True
data_test = ClipDataset('FlyWheelTest3s')
loader_testing = DataLoader(data_test, shuffle=True)

loss_fn = nn.MSELoss()

net = VisionNetNoField(params['dt'], [24, 64], device=params['device'])
net = VisionNet(params['dt'], [24,64], 5, device=params['device'])
if plot:
    # plt.figure(1)
    # plt.suptitle('Input')
    # plt.figure(2)
    # plt.suptitle('B In')
    # plt.figure(3)
    # plt.suptitle('B Fast')
    # plt.figure(4)
    # plt.suptitle('B Slow')
    # plt.figure(5)
    # plt.suptitle('B Out')
    # plt.figure(6)
    # plt.suptitle('L')
    # plt.figure(7)
    # plt.suptitle('E')
    # plt.figure(8)
    # plt.suptitle('D')
    # plt.figure(9)
    # plt.suptitle('S')
    # plt.figure(10)
    # plt.suptitle('CCW')
    # plt.figure(11)
    # plt.suptitle('CW')
    plt.figure(12)
    plt.suptitle('HC')

with torch.no_grad():
    loss_history = torch.zeros(len(loader_testing))
    for i, data in enumerate(loader_testing):
        print('%i/%i'%(i+1,len(loader_testing)))
        # Get data
        frames, target = data
        frames = frames.to(params['device'])
        frames = torch.squeeze(frames)

        # Simulate the network
        state_input, state_bo_input, state_bo_fast, state_bo_slow, state_bo_output, state_lowpass, state_enhance_on, state_direct_on, state_suppress_on, state_ccw_on, state_cw_on, state_hc = run_sample(frames, net)

        if plot:
            # plt.figure(1)
            # plot_state(state_input.to('cpu'), target)
            # plt.figure(2)
            # plot_state(state_bo_input.to('cpu'), target)
            # plt.figure(3)
            # plot_state(state_bo_fast.to('cpu'), target)
            # plt.figure(4)
            # plot_state(state_bo_slow.to('cpu'), target)
            # plt.figure(5)
            # plot_state(state_bo_output.to('cpu'), target)
            # plt.figure(6)
            # plot_state(state_lowpass.to('cpu'), target)
            # plt.figure(7)
            # plot_state(state_enhance_on.to('cpu'), target)
            # plt.figure(8)
            # plot_state(state_direct_on.to('cpu'), target)
            # plt.figure(9)
            # plot_state(state_suppress_on.to('cpu'), target)
            # plt.figure(10)
            # plt.subplot(2, 2, i + 1)
            # plt.imshow(state_ccw_on, cmap='bwr', vmin=-1, vmax=1)
            # plt.title(str(target.item()))
            # plt.figure(11)
            # plt.subplot(2, 2, i + 1)
            # plt.imshow(state_cw_on, cmap='bwr', vmin=-1, vmax=1)
            # plt.title(str(target.item()))
            plt.figure(12)
            plt.subplot(2,2,i+1)
            plt.title(str(target.item()))
            plt.plot(state_hc.transpose(0,1)[0,:], label='CW')
            plt.plot(state_hc.transpose(0,1)[1,:], label='CCW')
            plt.legend()
            plt.figure()
            # plt.plot(state_input[200,:], label='Input')
            # plt.plot(state_bo_input[200,:], label='BO In')
            # plt.plot(state_bo_fast[200,:], label='BO Fast')
            # plt.plot(state_bo_slow[200,:], label='BO Slow')
            # plt.plot(state_bo_output[200,:], label='BO Output')
            # plt.plot(state_lowpass[200,:], label='Lowpass')
            plt.plot(state_enhance_on[199,:], label='Enhance')
            plt.plot(state_direct_on[200,:], label='Direct')
            plt.plot(state_suppress_on[201,:], label='Suppress')
            plt.plot(state_ccw_on[200,:], label='CCW')
            plt.plot(state_cw_on[200,:], label='CW')
            plt.title(str(target.item()))
            # plt.ylim([-2,2])
            plt.legend()

            if i == 3:
                break
plt.show()