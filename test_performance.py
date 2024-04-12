from motion_vision_net import VisionNetNoField, VisionNet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from motion_data import ClipDataset
import pickle
import numpy as np

def condition_inputs(x):
    params = nn.ParameterDict({
        'stdCenBO': nn.Parameter(torch.tensor(10**x[0]), requires_grad=False),
        'ampRelBO': nn.Parameter(torch.tensor(x[1]), requires_grad=False),
        'stdSurBO': nn.Parameter(torch.tensor(10**x[2]), requires_grad=False),
        'ratioTauBO': nn.Parameter(torch.tensor(x[3]), requires_grad=False),
        'stdCenL': nn.Parameter(torch.tensor(10**x[4]), requires_grad=False),
        'ampRelL': nn.Parameter(torch.tensor(x[5]), requires_grad=False),
        'stdSurL': nn.Parameter(torch.tensor(10**x[6]), requires_grad=False),
        'stdCenBF': nn.Parameter(torch.tensor(10**x[7]), requires_grad=False),
        'ampRelBF': nn.Parameter(torch.tensor(x[8]), requires_grad=False),
        'stdSurBF': nn.Parameter(torch.tensor(10**x[9]), requires_grad=False),
        'ratioTauBF': nn.Parameter(torch.tensor(x[10]), requires_grad=False),
        'conductanceLEO': nn.Parameter(torch.tensor(x[11]), requires_grad=False),
        'ratioTauEO': nn.Parameter(torch.tensor(x[12]), requires_grad=False),
        'conductanceBODO': nn.Parameter(torch.tensor(x[13]), requires_grad=False),
        'ratioTauDO': nn.Parameter(torch.tensor(x[14]), requires_grad=False),
        'conductanceDOSO': nn.Parameter(torch.tensor(x[15]), requires_grad=False),
        'ratioTauSO': nn.Parameter(torch.tensor(x[16]), requires_grad=False),
        'conductanceLEF': nn.Parameter(torch.tensor(x[17]), requires_grad=False),
        'ratioTauEF': nn.Parameter(torch.tensor(x[18]), requires_grad=False),
        'conductanceBFDF': nn.Parameter(torch.tensor(x[19]), requires_grad=False),
        'ratioTauDF': nn.Parameter(torch.tensor(x[20]), requires_grad=False),
        'conductanceDFSF': nn.Parameter(torch.tensor(x[21]), requires_grad=False),
        'ratioTauSF': nn.Parameter(torch.tensor(x[22]), requires_grad=False),
        'conductanceEOOn': nn.Parameter(torch.tensor(10*x[23]), requires_grad=False),
        'conductanceDOOn': nn.Parameter(torch.tensor(x[24]), requires_grad=False),
        'conductanceEFOff': nn.Parameter(torch.tensor(x[25]), requires_grad=False),
        'conductanceDFOff': nn.Parameter(torch.tensor(x[26]), requires_grad=False),
        'biasEO': nn.Parameter(torch.tensor(x[27]), requires_grad=False),
        'biasDO': nn.Parameter(torch.tensor(x[28]), requires_grad=False),
        'biasSO': nn.Parameter(torch.tensor(x[29]), requires_grad=False),
        'biasEF': nn.Parameter(torch.tensor(x[30]), requires_grad=False),
        'biasDF': nn.Parameter(torch.tensor(x[31]), requires_grad=False),
        'biasSF': nn.Parameter(torch.tensor(x[32]), requires_grad=False),
        'gainHorizontal': nn.Parameter(torch.tensor(x[33]), requires_grad=False)
    })
    return params

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
data_test = ClipDataset('FlyWheelTest3s')
loader_testing = DataLoader(data_test, shuffle=False)
trial = '2024-04-05-1712341062.2805214'
best_history = pickle.load(open('Runs/'+trial+'-Best-History.p', 'rb'))
fit_history = pickle.load(open('Runs/'+trial+'-Fit-History.p', 'rb'))
pop_history = pickle.load(open('Runs/'+trial+'-Pop-History.p','rb'))
num_gen = len(best_history)
pop_size = len(fit_history)/num_gen
best_index = np.where(np.array(fit_history) <= 0.0001)[0][0]
individual = pop_history[best_index]

# net = VisionNetNoField(params['dt'], [24, 64], device=params['device'])
net_params = condition_inputs(individual)
net = VisionNet(params['dt'], [24,64], 5, device=params['device'], params=net_params)

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
pickle.dump(data, open(trial+'_best.p', 'wb'))
