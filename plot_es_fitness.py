import pickle
import matplotlib.pyplot as plt
import numpy as np

def plot_fitness(fit, best, seed, color):
    num_gen = len(best)
    gens = list(range(num_gen))
    pop_size = 24
    fit_reshape = np.reshape(fit,[num_gen, pop_size])
    fit_reshape = np.clip(fit_reshape,0,20)
    fit_median = np.median(fit_reshape, axis=1)
    fit_5 = np.percentile(fit_reshape, 5, axis=1)
    fit_95 = np.percentile(fit_reshape, 95, axis=1)
    plt.subplot(2,1,1)
    plt.fill_between(gens, fit_5, fit_95, color=color, alpha=0.5)
    plt.plot(gens, fit_median, color=color, label='Seed: %i'%(seed))
    plt.subplot(2,1,2)
    plt.plot(gens, best, color=color, label='Seed: %i'%(seed))
    # plt.title('Seed: %i'%(seed))
    # plt.yscale('log', base=2)
    # plt.ylim([0,100])
    # plt.legend()

seed_1_fit = pickle.load(open('Runs/2024-03-29-1711743092.2438712-Fit-History.p', 'rb'))
seed_1_fit_best = pickle.load(open('Runs/2024-03-29-1711743092.2438712-Best-History.p', 'rb'))
seed_10_fit = pickle.load(open('Runs/2024-03-31-1711915538.720788-Fit-History.p', 'rb'))
seed_10_fit_best = pickle.load(open('Runs/2024-03-31-1711915538.720788-Best-History.p', 'rb'))
seed_100_fit = pickle.load(open('Runs/2024-04-01-1711983050.3523383-Fit-History.p', 'rb'))
seed_100_fit_best = pickle.load(open('Runs/2024-04-01-1711983050.3523383-Best-History.p', 'rb'))
seed_1000_fit = pickle.load(open('Runs/2024-04-05-1712341062.2805214-Fit-History.p', 'rb'))
seed_1000_fit_best = pickle.load(open('Runs/2024-04-05-1712341062.2805214-Best-History.p', 'rb'))
test = pickle.load(open('Runs/2024-03-29-1711743092.2438712-Pop-History.p','rb'))

plt.figure()
plot_fitness(seed_1_fit, seed_1_fit_best, 1, 'C0')
plot_fitness(seed_10_fit, seed_10_fit_best, 10, 'C1')
plot_fitness(seed_100_fit, seed_100_fit_best, 100, 'C2')
plot_fitness(seed_1000_fit, seed_1000_fit_best, 1000, 'C3')
plt.subplot(2,1,1)
plt.title('Population Fitness')
plt.legend()
plt.subplot(2,1,2)
plt.title('Best Fitness')
plt.legend()

plt.show()