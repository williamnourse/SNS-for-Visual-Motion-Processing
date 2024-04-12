import numpy as np
import seaborn
import pickle
import pandas as pd
import matplotlib.pyplot as plt

bounds = pd.read_csv('bounds_20240326.csv')
labels = bounds['Parameter'].to_list()
labels.append('Fitness')

pop = np.array(pickle.load(open('Runs/2024-03-29-1711743092.2438712-Pop-History.p', 'rb')))
fit = np.array([pickle.load(open('Runs/2024-03-29-1711743092.2438712-Fit-History.p', 'rb'))])
data = np.hstack((pop,fit.transpose()))
frame = pd.DataFrame(data, columns=labels)

plt.figure()
seaborn.pairplot(frame, hue='Fitness', palette='rocket')
plt.show()
