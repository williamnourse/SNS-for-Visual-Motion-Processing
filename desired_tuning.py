import numpy as np
import pickle

sample_pts = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
desired_tuning = np.linspace(1,0.1,num=len(sample_pts))
desired_performance = {'samplePts': sample_pts,
                       'magnitude': desired_tuning}
pickle.dump(desired_performance, open('Figures/desiredPerformance.p', 'wb'))
