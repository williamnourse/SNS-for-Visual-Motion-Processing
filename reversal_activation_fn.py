import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def act(x, lo=torch.tensor([-2]), hi=torch.tensor([5])):
    return torch.abs(lo)*nn.functional.sigmoid(6*(x+1))+lo + hi*nn.functional.sigmoid(6*(x-1))

data = torch.linspace(-3,3, 50)

plt.figure()
plt.plot(data, act(data))
# plt.axhline(y=0, color='black')
plt.xlabel('Input')
plt.ylabel('Reversal Potential')
plt.show()