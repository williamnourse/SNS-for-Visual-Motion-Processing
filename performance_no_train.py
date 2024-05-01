from motion_vision_net import VisionNetNoField, VisionNetNoTrain
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from motion_data import ClipDataset
import pickle

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
    warmup = 400
    niter = num_sub_steps*(sample.shape[0])
    step = 0

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

            hc[step, :] = state_hc

            step += 1
    cw_mean = torch.mean(hc[:, 0])
    ccw_mean = torch.mean(hc[:, 1])

    return cw_mean, ccw_mean

params = {'dt': 1/(30*13)*1000, 'device': 'cuda'}
data_test = ClipDataset('/home/will/flywheel-rotation-dataset/FlyWheelTest3s')
loader_testing = DataLoader(data_test, shuffle=False)


net = VisionNetNoField(params['dt'], [24, 64], device=params['device'])
net = VisionNetNoTrain(params['dt'], [24,64], 5, device=params['device'])

data_cw = torch.zeros([len(loader_testing)], device=params['device'])
data_ccw = torch.zeros([len(loader_testing)], device=params['device'])
targets = torch.zeros([len(loader_testing)])

with torch.no_grad():
    loss_history = torch.zeros(len(loader_testing))
    for i, data in enumerate(loader_testing):
        print('%i/%i'%(i+1,len(loader_testing)))
        # Get data
        frames, target = data
        frames = frames.to(params['device'])
        frames = torch.squeeze(frames)

        # Simulate the network
        cw_mean, ccw_mean = run_sample(frames, net)
        data_cw[i] = cw_mean
        data_ccw[i] = ccw_mean
        targets[i] = target

data = {'cw': data_cw.to('cpu'), 'ccw': data_ccw.to('cpu'), 'targets': targets}
pickle.dump(data, open('field_no_train_mean.p', 'wb'))