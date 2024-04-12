import pickle
import time
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
# import pandas as pd
import argparse
from motion_vision_net import SNSMotionVisionOn
from motion_data import ClipDataset
from datetime import datetime
import matplotlib.pyplot as plt


def process_args(args):
    """
    Given command-line arguments, fill a parameter dictionary
    :param args: parser arguments
    :return: return a parameter dictionary
    """
    params = {}
    params['dt'] = float(args.dt)
    if args.dtype == 'float64':
        params['dtype'] = torch.float64
    elif args.dtype == 'float32':
        params['dtype'] = torch.float32
    else:
        params['dtype'] = torch.float16
    params['numEpochs'] = int(args.num_epochs)
    params['lr'] = float(args.lr)
    params['shapeField'] = int(args.shape_field)
    params['logInterval'] = int(args.log_interval)
    if args.network == 'On':
        params['network'] = SNSMotionVisionOn
    params['device'] = str(args.device)

    return params


def vel_to_state(vel):
    """
    Convert velocity labels in the dataset to their corresponding neural voltage
    :param vel: Velocity label
    :return: State
    """
    return 1 - ((vel-.1)/0.4) * (1-0.1)


def state_to_vel(state):
    """
    Inverse of vel_to_state, convert a neural state to its corresponding velocity
    :param state: Neural state
    :return: Velocity
    """
    vel = ((1 - state) / (1 - 0.1)) * 0.4 + 0.1
    return vel


def run_sample(sample, net: nn.Module):
    """
    Process a training sequence of frames through the network
    :param sample: sequence of 30 frames
    :param net: network to be trained
    :return: the average value over the sample of the ccw neuron, net
    """
    (state_input, state_bo_input, state_bo_fast, state_bo_slow, state_bo_output, state_lowpass, state_enhance_on,
     state_direct_on, state_suppress_on, state_cw_on, state_ccw_on, state_horizontal, avg) = net.init()
    net.setup()
    net.zero_grad()
    num_sub_steps = 13
    niter = num_sub_steps*sample.shape[0]
    step = 0
    for i in range(sample.shape[0]):
        for j in range(num_sub_steps):
            # print(torch.max(sample))

            # print('Sample %i Step %i'%(i,step))
            (state_input, state_bo_input, state_bo_fast, state_bo_slow, state_bo_output, state_lowpass, state_enhance_on,
             state_direct_on, state_suppress_on, state_cw_on, state_ccw_on, state_horizontal, avg) = net(sample[i, :, :],
                                                                                                         state_input,
                                                                                                         state_bo_input,
                                                                                                         state_bo_fast,
                                                                                                         state_bo_slow,
                                                                                                         state_bo_output,
                                                                                                         state_lowpass,
                                                                                                         state_enhance_on,
                                                                                                         state_direct_on,
                                                                                                         state_suppress_on,
                                                                                                         state_cw_on,
                                                                                                         state_ccw_on,
                                                                                                         state_horizontal,
                                                                                                         avg, niter)


            # print('Avg')
            # print(avg)
            # print('Input')
            # print(torch.min(state_input).item(), torch.max(state_input).item())
            # print('BO Input')
            # print(torch.min(state_bo_input).item(), torch.max(state_bo_input).item())
            # print('BO Fast')
            # print(torch.min(state_bo_fast).item(), torch.max(state_bo_fast).item())
            # print('BO Slow')
            # print(torch.min(state_bo_slow).item(), torch.max(state_bo_slow).item())
            # print('BO Out')
            # print(torch.min(state_bo_output).item(), torch.max(state_bo_output).item())
            # print('LO')
            # print(torch.min(state_lowpass).item(), torch.max(state_lowpass).item())
            # print('EO')
            # print(torch.min(state_enhance_on).item(), torch.max(state_enhance_on).item())
            # print('DO')
            # print(torch.min(state_direct_on).item(), torch.max(state_direct_on).item())
            # print('SO')
            # print(torch.min(state_suppress_on).item(), torch.max(state_suppress_on).item())
            # print('CW')
            # print(torch.min(state_cw_on).item(), torch.max(state_cw_on).item())
            # print('CCW')
            # print(torch.min(state_ccw_on).item(), torch.max(state_ccw_on).item())
            # print('Horizontal')
            # print(state_horizontal)
            if torch.any(torch.isnan(state_lowpass)).item() or torch.any(torch.isnan(state_bo_input)).item() or torch.any(torch.isnan(avg)).item():
                print('Avg')
                print(avg)
                print('Input')
                print(torch.min(state_input).item(), torch.max(state_input).item())
                print('BO Input')
                print(torch.min(state_bo_input).item(), torch.max(state_bo_input).item())
                print('BO Fast')
                print(torch.min(state_bo_fast).item(), torch.max(state_bo_fast).item())
                print('BO Slow')
                print(torch.min(state_bo_slow).item(), torch.max(state_bo_slow).item())
                print('BO Out')
                print(torch.min(state_bo_output).item(), torch.max(state_bo_output).item())
                print('LO')
                print(torch.min(state_lowpass).item(), torch.max(state_lowpass).item())
                print('EO')
                print(torch.min(state_enhance_on).item(), torch.max(state_enhance_on).item())
                print('DO')
                print(torch.min(state_direct_on).item(), torch.max(state_direct_on).item())
                print('SO')
                print(torch.min(state_suppress_on).item(), torch.max(state_suppress_on).item())
                print('CW')
                print(torch.min(state_cw_on).item(), torch.max(state_cw_on).item())
                print('CCW')
                print(torch.min(state_ccw_on).item(), torch.max(state_ccw_on).item())
                print('Horizontal')
                print(state_horizontal)
                quit()

            step += 1

    return avg, net


def validate(net, loss_fn, testing_loader, params):
    """
    Iterate over the testing set and measure the performance
    :param net: Network to evaluate
    :param loss_fn: Loss function to base evaluation on
    :param testing_loader: Dataloader for testing data
    :return: Average loss over the test set
    """
    with torch.no_grad():
        loss_history = torch.zeros(len(testing_loader))
        for i, (frames, target) in enumerate(testing_loader):
            # if i == 200:
            #     break
            # Get data
            frames, target = frames.to(params['device']), target.to(params['device'])
            frames = torch.squeeze(frames)
            target = vel_to_state(target)

            # Simulate the network
            avg, net = run_sample(frames, net)

            # Compute loss and gradients
            loss = loss_fn(avg, target)
            loss_history[i] = loss.item()
        mean_loss = torch.mean(loss_history)
    return mean_loss


def run_epoch(index, net, loss_fn, optimizer, training_loader, testing_loader, params):
    """
    Train the network over every sample in the training dataset, evaluate its average testing loss, and save a checkpoint.
    :param index: Current epoch number
    :param net: Network to train
    :param loss_fn: Loss function to evaluate the network
    :param optimizer: Optimizer to train the network
    :param training_loader: Dataloader for training data
    :param testing_loader: Dataloader for testing data
    :param params: Optimization parameters
    :return: The raw losses
    """
    start = time.time()
    loss_history = torch.zeros(len(training_loader))
    for i, (frames, target) in enumerate(training_loader):
        # if i == 190:
        #     break
        # print(i)
        # Get data
        frames, target = frames.to(params['device']), target.to(params['device'])
        frames = torch.squeeze(frames)
        target = vel_to_state(target)
        target = torch.as_tensor(target, dtype=params['dtype'])

        # Reset gradients
        net.zero_grad()

        # Simulate the network
        avg, net = run_sample(frames, net)

        # Compute loss and gradients
        loss = loss_fn(avg, target)
        loss.backward()

        # Adjust parameters
        optimizer.step()
        if i % params['logInterval'] == 0:
            print('Epoch: %i Sample %i/%i Epoch Time: %.4f Loss: %.4f'%(index, i+1, len(training_loader), time.time()-start, loss.item()))
        loss_history[i] = loss.item()
    epoch_mean_loss = torch.mean(loss_history)
    test_loss = validate(net, loss_fn, testing_loader, params)
    print('Epoch: %i Mean Loss: %.4f'%(index, epoch_mean_loss.item()))
    print('Validation Loss: %.4f'%test_loss.item())

    torch.save({'epoch': index,
                'netStateDict': net.state_dict(),
                'optimizerStateDict': optimizer.state_dict(),
                }, 'checkpoints/model.pt')
    return loss_history, epoch_mean_loss, test_loss, net, optimizer


def train(params):
    """
    Train the network
    :param params: Optimizer parameters
    :return:
    """
    now = datetime.now()
    t = now.strftime('%Y-%m-%d_%H-%M-%S')
    data_training = ClipDataset('FlyWheelTrain')
    loader_training = DataLoader(data_training, shuffle=True)
    data_test = ClipDataset('FlyWheelTest')
    loader_testing = DataLoader(data_test, shuffle=True)

    loss_fn = nn.MSELoss()

    net = params['network'](params['dt'], [24, 64], params['shapeField'], dtype=params['dtype'],
                            device=params['device'])

    optimizer = torch.optim.Adam(net.parameters(), lr=params['lr'])

    epoch_loss_history = torch.zeros(params['numEpochs'])
    test_loss_history = torch.zeros(params['numEpochs'])

    for i in range(params['numEpochs']):
        epoch_loss, epoch_mean_loss, test_loss, net, optimizer = run_epoch(i, net, loss_fn, optimizer, loader_training,
                                                                           loader_testing, params)
        if i == 0:
            loss_history = epoch_loss
        else:
            loss_history = torch.cat((loss_history, epoch_loss))
        epoch_loss_history[i] = epoch_mean_loss
        test_loss_history[i] = test_loss
        if i > 0 and test_loss > test_loss_history[i-1]:
            break

        filename = 'checkpoints/' + t + '_' + str(i) + '.pt'
        torch.save({'epoch': i,
                    'netStateDict': net.state_dict(),
                    'optimizerStateDict': optimizer.state_dict(),
                    'lossHistory': loss_history,
                    'epochLossHistory': epoch_loss_history,
                    'testLossHistory': test_loss_history
                    }, filename)
    return loss_history, epoch_loss_history, test_loss_history


def main(args):
    """
    Main section
    :param args: Command Line Arguments
    :return: None
    """
    params = process_args(args)

    loss_history, epoch_loss_history, test_loss_history = train(params)

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(loss_history)
    plt.title('Raw Loss')
    plt.subplot(3, 1, 2)
    plt.plot(epoch_loss_history)
    plt.title('Mean Loss Per Epoch')
    plt.subplot(3, 1, 3)
    plt.plot(test_loss_history)
    plt.title('Validation Loss')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the Network")
    parser.add_argument('--dt', nargs='?', default='2.56')
    parser.add_argument('--dtype', nargs='?', choices=['float64', 'float32', 'float16'], default='float32')
    parser.add_argument('--bound_file', nargs='?', default='bounds_on_20240215.csv')
    parser.add_argument('--num_epochs', nargs='?', default='10')
    parser.add_argument('--lr', nargs='?', default='0.001')
    parser.add_argument('--shape_field', nargs='?', default='5')
    parser.add_argument('--log_interval', nargs='?', default='1')
    parser.add_argument('--compile', nargs='?', default='True')
    parser.add_argument('--network', nargs='?', default='On', choices=['On'])
    parser.add_argument('--device', nargs='?', default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()

    main(args)
