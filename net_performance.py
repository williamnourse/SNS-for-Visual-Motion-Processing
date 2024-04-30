from motion_vision_net import VisionNet_1F, NetHandler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from motion_data import ClipDataset
import pickle
from tqdm import tqdm

def vel_to_state(vel):
    """
    Convert velocity labels in the dataset to their corresponding neural voltage
    :param vel: Velocity label
    :return: State
    """
    return 1 - ((vel-.1)/0.4) * (1-0.1)


params = {'dt': 1/(30*13)*1000, 'device': 'cuda'}
data_test = ClipDataset('/home/will/flywheel-rotation-dataset/FlyWheelTest3s')
loader_testing = DataLoader(data_test, shuffle=False)


net = NetHandler(VisionNet_1F, params['dt'], [24, 64], 5, device=params['device'])
net.load_state_dict(torch.load('Runs/20240419/CHECKPT-19-04-2024-16-03-18.pt'))
net.setup()
criterion = nn.MSELoss()

data_cw = torch.zeros([len(loader_testing)], device=params['device'])
data_ccw = torch.zeros([len(loader_testing)], device=params['device'])
targets = torch.zeros([len(loader_testing)], device=params['device'])

with torch.no_grad():

    # optimizer.zero_grad()
    for i, data in enumerate(tqdm(loader_testing)):
        inputs, labels = data
        inputs, labels = inputs.to(params['device']), labels.to(params['device'])
        # outputs = torch.zeros_like(targets)
        # inputs = inputs.view(-1, 28, 28)
        states = net.init(1)

        ccw_mean, cw_mean = net(inputs, states)
        # cw_mean, ccw_mean = outputs[-1][0], outputs[-1][1]
        data_cw[i] = cw_mean
        data_ccw[i] = ccw_mean
        targets[i] = labels

placeholder = criterion(data_ccw, vel_to_state(targets))
test_loss = placeholder.detach().item()
print(test_loss)

data = {'cw': data_cw.to('cpu'), 'ccw': data_ccw.to('cpu'), 'targets': targets}
pickle.dump(data, open('20240419-Results.p', 'wb'))