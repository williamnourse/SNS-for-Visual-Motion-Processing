import pickle
import time
import torch
import torch.nn as nn
from torch import optim
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
# import pandas as pd
import argparse
from motion_vision_net import VisionNet, NetHandler
from motion_data import ClipDataset
from datetime import datetime
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

start = time.time()
N_STEPS = 90
BATCH_SIZE = 24
N_EPOCHS = 50
DT = ((1/30)/13)*1000
IMG_SIZE = [24,64]
FIELD_SIZE = 5
N_SEEDS = 1
LOG_INTERVAL = 1

def vel_to_state(vel, device):
    """
    Convert velocity labels in the dataset to their corresponding neural voltage
    :param vel: Velocity label
    :return: State
    """
    state = torch.zeros([len(vel), 2], device=device)
    state[:,0] = -2*vel
    state[:,1] = 2*vel
    return torch.clamp(state, min=0)

# vel = torch.tensor([-0.5, -0.45, -0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
# out = vel_to_state(vel)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train = ClipDataset('/home/will/flywheel-rotation-dataset/FlyWheelTrain')
test = ClipDataset('/home/will/flywheel-rotation-dataset/FlyWheelTest')
train_dataloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

for r in range(N_SEEDS):
    model = NetHandler(VisionNet, DT, IMG_SIZE, FIELD_SIZE, device=device).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    test_loss = 0.0
    test_loss_last = 0.0
    loss_history = []
    loss_test_history = []
    # for epoch in range(N_EPOCHS):  # loop over the dataset multiple times
    epoch = 0
    while test_loss < test_loss_last or epoch <= N_EPOCHS:
        train_running_loss = 0.0
        train_acc = 0.0
        with torch.enable_grad():
            model.train()
            # zero the parameter gradients
            optimizer.zero_grad()

            # TRAINING ROUND
            for i, data in enumerate(tqdm(train_dataloader)):
                # print(i)
                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                targets = vel_to_state(labels, device).type(model.net.dtype)
                # outputs = torch.zeros_like(targets)
                # inputs = inputs.view(-1, 28, 28)

                model.setup()
                # for j in range(BATCH_SIZE):
                states = model.init(inputs.shape[0])
                outputs = model(inputs, states)
                # reset hidden states
                # states = model.init(BATCH_SIZE)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_running_loss += loss.detach().item()

            # if i%LOG_INTERVAL==0:
            #     print('Run: %i | Epoch:  %d | Batch: %i | Loss: %.4f | Time: %i secs'
            #   % (r, epoch, i, train_running_loss / (i+1), time.time()-start))

        model.eval()
        print('Run: %i | Epoch:  %d | Loss: %.4f | Time: %i secs' % (r, epoch, train_running_loss / (i+1), time.time()-start))
        loss_history.extend([train_running_loss/(i+1)])
        test_loss_last = test_loss
        test_loss = 0.0
        with torch.no_grad():
            model.setup()
            # optimizer.zero_grad()
            for i, data in enumerate(tqdm(test_dataloader)):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                targets = vel_to_state(labels, device).type(model.net.dtype)
                # outputs = torch.zeros_like(targets)
                # inputs = inputs.view(-1, 28, 28)
                states = model.init(inputs.shape[0])

                outputs = model(inputs, states)
                placeholder = criterion(outputs, targets)
                test_loss += placeholder.detach().item()
            loss_test_history.extend([test_loss/(i+1)])
        epoch += 1

        print('Test Loss: %.4f | Time: %i secs' % (test_loss / (i+1), time.time()-start))
        torch.save(model.state_dict(),'1-CHECKPT-'+datetime.now().strftime('%Y-%m-%d-%H-%M-%S')+'.pt')

        save_data = {'loss': loss_history, 'lossTest': loss_test_history}
        pickle.dump(save_data, open('1-LOSS-'+datetime.now().strftime('%Y-%m-%d-%H-%M-%S')+'.p', 'wb'))

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.subplot(2,1,2)
    plt.plot(loss_test_history)
    plt.title('Test Loss')
    plt.xlabel('Epoch')
    plt.legend()
plt.show()