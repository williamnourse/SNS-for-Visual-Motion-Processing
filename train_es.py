import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import os
import numpy as np
import pandas as pd
from motion_vision_net import VisionNet
from motion_data import ClipDataset
import time
import cma
import pickle
from datetime import date

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
    :param sample: sequence of frames
    :param net: network to be trained
    :return: the average value over the sample of the ccw neuron, net
    """
    (state_input, state_bo_input, state_bo_fast, state_bo_slow, state_bo_output, state_lowpass,
    state_bf_input, state_bf_fast, state_bf_slow, state_bf_output, state_enhance_on, state_direct_on,
    state_suppress_on, state_enhance_off, state_direct_off, state_suppress_off, state_ccw_on, state_cw_on,
    state_ccw_off, state_cw_off, state_hc) = net.init()
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
            if torch.any(torch.isnan(state_hc)):
                return None, None

            step += 1
    cw_mean = torch.mean(hc[:, 0])
    ccw_mean = torch.mean(hc[:, 1])

    return cw_mean, ccw_mean

def run_batch(batch, targets, net):
    results = torch.zeros([len(targets),3])
    for i in range(len(targets)):
        # print(i)
        # Get data
        frames = batch[i,:,:,:]
        frames = torch.squeeze(frames)

        # Simulate the network
        cw_mean, ccw_mean = run_sample(frames, net)
        if cw_mean is None or ccw_mean is None:
            return None
        results[i,0] = cw_mean
        results[i,1] = ccw_mean
        results[i,2] = targets[i]
    dataframe = pd.DataFrame(results, columns=['cw', 'ccw', 'target'])
    return dataframe

def calc_fitness(results):
    if results is None:
        return sys.float_info.max
    ccw_mean = results.groupby('target', as_index=False)['ccw'].mean()['ccw'].to_numpy()
    cw_mean = results.groupby('target', as_index=False)['cw'].mean()['cw'].to_numpy()

    # monotonic
    diff_ccw = np.diff(ccw_mean)
    num_pos = np.count_nonzero(diff_ccw>0)
    num_zero = np.count_nonzero(diff_ccw==0)
    num_neg = len(diff_ccw)-num_pos-num_zero
    minority = min(num_neg,num_pos)
    fit_monotonic = minority+num_zero

    # greater than cw
    fit_comparison = np.count_nonzero(ccw_mean<=cw_mean)

    fitness = fit_monotonic + fit_comparison
    if len(diff_ccw)>1:
        sec_diff_ccw = np.diff(diff_ccw)
        fitness += np.sum(np.abs(sec_diff_ccw))

    return fitness

def individual(x, batch, targets):#, pop_history=None):
    # if pop_history is None:
    #     any_match = False
    # else:
    #     # See if individual has been evaluated before
    #     matching_individuals = (pop_history.iloc[:,:-1] == x).all(axis=1)
    #     any_match = matching_individuals.any()
    # if any_match:
    #     index = pop_history.index[matching_individuals][0]
    #     fitness = pop_history['fitness'][index]
    # else:
    params = condition_inputs(x)
    # Net properties
    dt = 1 / (30 * 13) * 1000
    shape_input = [24, 64]
    shape_field = 5
    net = VisionNet(dt, shape_input, shape_field)
    # torch._dynamo.reset()
    with torch.no_grad():
        net.params.update(params)
        net.setup()
        # model = torch.compile(net, dynamic=True)
        results = run_batch(batch, targets, net)
        fitness = calc_fitness(results)
    return fitness

def shape_fitness(fitness, pop_size):
    rank = np.argsort(fitness)
    fit_range = np.linspace(-1,1,num=pop_size)
    new_fit = np.zeros(pop_size)
    for i in range(pop_size):
        index = rank[i]
        new_fit[index] = fit_range[i]
    return new_fit

if __name__ == '__main__':
    # Bounds
    bounds = pd.read_csv('bounds_20240326.csv')
    x0 = bounds['Start'].to_numpy()
    bounds_lower = bounds['Lower Bound'].to_numpy()
    bounds_upper = bounds['Upper Bound'].to_numpy()
    labels = bounds['Parameter'].to_list()
    labels.append('fitness')

    # CMA-ES
    # Optimization Properties
    seed = 1000
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    num_workers = os.cpu_count()
    pop_size = num_workers
    # pop_size = 9
    tol = 1e-4
    max_gen = 10000
    batch_size = 16
    # batch_size = [2,4,8,16,32,64,128,256]
    data_train = ClipDataset('FlyWheelTrain3s')
    loader_training = DataLoader(data_train, shuffle=True, batch_size=batch_size)
    sigma0 = 0.3
    options = {'bounds': [bounds_lower, bounds_upper],
               'popsize': pop_size,
               'seed': seed}
    optim = cma.CMAEvolutionStrategy(x0, sigma0, options)

    # File management
    today = str(date.today())
    prefix = today+'-'+str(time.time())+'-'

    # Optimization Loop
    stop = False
    gen = 0
    time_start_optim = time.time()
    pop_history = []
    fit_history = []
    pop_best_history = []
    best_history = []
    while not stop:
        time_start_gen = time.time()

        # get data
        batch, targets = next(iter(loader_training))
        batch.share_memory_()
        targets.share_memory_()

        # get candidates
        pop = optim.ask()

        # calculate fitness
        with mp.Pool(num_workers) as pool:
            fitness = pool.starmap(individual, [(x, batch, targets) for x in pop])
        # fitness = []
        # for j in range(pop_size):
        #     fitness.append(individual(pop[j], batch, targets))
        # print(fitness)

        # update population
        rank_fit = shape_fitness(fitness, pop_size)
        optim.tell(pop, rank_fit)

        # update history
        pop_history.extend(pop)
        fit_history.extend(fitness)
        pop_best_fit = fitness[np.argsort(fitness)[0]]
        all_best_fit = fit_history[np.argsort(fit_history)[0]]
        pop_best_history.append(pop_best_fit)
        best_history.append(all_best_fit)

        # Log Performance
        time_end = time.time()
        time_gen = time_end-time_start_gen
        time_all = time_end-time_start_optim
        print('Generation: %i - Generation Time: %i sec - Total Time: %i sec - Population Best: %.4f - All-Time Best: %.4f'%(gen+1, time_gen, time_all, pop_best_fit, all_best_fit))

        # Save to disk
        pickle.dump(pop_history, open('Runs/'+prefix+'Pop-History.p','wb'))
        pickle.dump(fit_history, open('Runs/'+prefix+'Fit-History.p','wb'))
        pickle.dump(pop_best_history, open('Runs/'+prefix+'Pop-Best-History.p','wb'))
        pickle.dump(best_history, open('Runs/'+prefix+'Best-History.p','wb'))

        # Update Iteration
        gen += 1
        # Check for termination
        if gen >= max_gen:
            print('Terminating from Max Number of Generations')
            stop = True
        elif all_best_fit <= tol:
            print('Terminating from Target Fitness')
            stop = True
        else:
            stop = False
