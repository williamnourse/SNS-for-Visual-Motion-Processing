import pickle
import time
import torch
import torch.nn as nn
from torch import optim
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
# import pandas as pd
import argparse
from motion_vision_net import VisionNet_1F, NetHandler, VisionNet_1F_FB, VisionNet_3F, NetHandlerWt
from motion_data import ClipDataset
from datetime import datetime
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

start = time.time()
N_STEPS = 90
BATCH_SIZE = 18
N_EPOCHS = 20
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
    state = torch.zeros(len(vel), device=device, dtype=torch.int64)
    for i in range(len(vel)):
        if vel[i] < 0:
            state[i] = 1
    return state

def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()


# vel = torch.tensor([-0.5, -0.45, -0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
# out = vel_to_state(vel)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train = ClipDataset('/home/will/flywheel-rotation-dataset/FlyWheelTrain')
test = ClipDataset('/home/will/flywheel-rotation-dataset/FlyWheelTest')
train_dataloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

for r in range(N_SEEDS):
    model = NetHandlerWt(VisionNet_3F, DT, IMG_SIZE, FIELD_SIZE, device=device).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    test_loss = 0.0
    test_loss_last = 0.0
    loss_history = []
    loss_test_history = []
    accuracy_history = []
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
                targets = vel_to_state(labels, device)
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
                train_acc += get_accuracy(outputs, targets, len(labels))

            # if i%LOG_INTERVAL==0:
            #     print('Run: %i | Epoch:  %d | Batch: %i | Loss: %.4f | Time: %i secs'
            #   % (r, epoch, i, train_running_loss / (i+1), time.time()-start))

        model.eval()
        print('Run: %i | Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f%% Time: %i secs' % (r, epoch, train_running_loss / (i+1), train_acc/(i+1), time.time()-start))
        loss_history.extend([train_running_loss/(i+1)])
        test_loss_last = test_loss
        test_loss = 0.0
        num_correct = 0
        total = 0
        test_acc = 0
        with torch.no_grad():
            model.setup()
            # optimizer.zero_grad()
            for i, data in enumerate(tqdm(test_dataloader)):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                targets = vel_to_state(labels, device)
                # outputs = torch.zeros_like(targets)
                # inputs = inputs.view(-1, 28, 28)
                states = model.init(inputs.shape[0])

                outputs = model(inputs, states)
                placeholder = criterion(outputs, targets)
                test_loss += placeholder.detach().item()
                test_acc += get_accuracy(outputs, targets, len(labels))
            loss_test_history.extend([test_loss/(i+1)])
            accuracy_history.extend([test_acc])
        epoch += 1

        print('Test Loss: %.4f | Test Accuracy: %.2f%% | Time: %i secs' % (test_loss / (i+1), test_acc/(i+1), time.time()-start))
        torch.save(model.state_dict(),'B-1-CHECKPT-'+datetime.now().strftime('%Y-%m-%d-%H-%M-%S')+'.pt')

        save_data = {'loss': loss_history, 'lossTest': loss_test_history, 'accuracy': accuracy_history}
        pickle.dump(save_data, open('B-1-LOSS-'+datetime.now().strftime('%Y-%m-%d-%H-%M-%S')+'.p', 'wb'))

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