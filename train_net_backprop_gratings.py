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
# from motion_vision_net_v2 import NetHandler, VisionNetv2
from motion_data import ClipDataset
from datetime import datetime
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

start = time.time()
N_STEPS = 30
BATCH_SIZE = 1
STEP_INTERVAL = 1
N_EPOCHS = 2
N_WARMUP = 15 #*13
DT = ((1/30)/13)*1000
IMG_SIZE = [24,64]
N_LAMINA = 5
N_MEDULLA = 10
N_LOBULA = 4
FIELD_SIZE = 5
N_SEEDS = 5
LOG_INTERVAL = 1

def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()


# vel = torch.tensor([-0.5, -0.45, -0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
# out = vel_to_state(vel)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = ClipDataset('/home/will/flywheel-rotation-dataset/Gratings')
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

for r in range(N_SEEDS):
    model = NetHandler(VisionNet_1F, DT, IMG_SIZE, FIELD_SIZE, device=device).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    test_loss = 0.0
    test_loss_last = 0.0
    loss_history = []
    loss_test_history = []
    accuracy_history = []
    # for epoch in range(N_EPOCHS):  # loop over the dataset multiple times
    epoch = 0
    # while test_loss < test_loss_last or epoch <= N_EPOCHS:
    while epoch <= N_EPOCHS:
        train_running_loss = 0.0
        train_acc = 0.0
        # with torch.enable_grad():
        model.train()
        sample = 0
        eval = 0
        # TRAINING ROUND
        for i, data in enumerate(tqdm(dataloader)):
            if sample%STEP_INTERVAL == 0:
                # print(i)
                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.type(torch.int64)
                # targets = vel_to_state(labels, device)
                # outputs = torch.zeros_like(targets)
                # inputs = inputs.view(-1, 28, 28)
                with torch.no_grad():
                    model.setup()

                    # Warmup
                    warmup = torch.zeros([inputs.shape[0], 15, 24, 64], device=device) + 0.5
                    states = model.init(inputs.shape[0])
                    _, states = model(warmup, states)
                with torch.enable_grad():
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    model.setup()
                    outputs, _ = model(inputs, states)
                    # reset hidden states
                    # states = model.init(BATCH_SIZE)

                    loss = criterion(outputs, labels)
                    # if torch.isnan(loss):
                        # print(loss)
                    loss.backward()
                    optimizer.step()

                train_running_loss += loss.detach().item()
                loss_history.extend([loss.detach().item()])
                # print(train_running_loss)
                acc = get_accuracy(outputs, labels, len(labels))
                train_acc += acc
                accuracy_history.extend([acc])
                eval += 1
            sample += 1
        # if i%LOG_INTERVAL==0:
        #     print('Run: %i | Epoch:  %d | Batch: %i | Loss: %.4f | Time: %i secs'
        #   % (r, epoch, i, train_running_loss / (i+1), time.time()-start))

        print('Run: %i | Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f%% Time: %i secs' % (r, epoch, train_running_loss / (eval), train_acc/(eval), time.time()-start))


        epoch += 1
        torch.save(model.state_dict(),str(r)+'GRATE-CHECKPT-'+datetime.now().strftime('%Y-%m-%d-%H-%M-%S')+'.pt')

        save_data = {'loss': loss_history, 'accuracy': accuracy_history}
        pickle.dump(save_data, open(str(r)+'GRATE-LOSS-'+datetime.now().strftime('%Y-%m-%d-%H-%M-%S')+'.p', 'wb'))

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.subplot(2,1,2)
    plt.plot(accuracy_history)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    # plt.legend()
plt.show()