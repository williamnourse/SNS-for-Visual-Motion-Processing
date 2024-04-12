import pickle
import time
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
import pandas as pd
import argparse
from evotorch import Problem, Solution
import evotorch.algorithms as alg
from evotorch.logging import PandasLogger
from motion_vision_net import SNSMotionVisionMerged
from motion_data import ClipDataset
from datetime import datetime
import matplotlib.pyplot as plt

optimization_file = 'Runs/Full/log_2024-02-14_11-34-42_CMA-ES.p'
parameter_file = 'Runs/Full/params_2024-02-14_11-34-42_CMA-ES.p'

data = pickle.load(open(optimization_file,'rb'))
plt.figure()
plt.plot(data['pop_best_eval'])
# plt.title('Error: ' + str(test_error.item()))
plt.xlabel('Trial')
plt.ylabel('Error')

plt.show()