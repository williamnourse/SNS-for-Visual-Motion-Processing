from evotorch import Problem
import evotorch.algorithms as alg
from evotorch.logging import PandasLogger
import torch
import matplotlib.pyplot as plt
import time
import pickle
import pandas as pd
from evotorch.decorators import vectorized

@vectorized
def sphere(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x.pow(2.0), dim=-1) + torch.rand(1)*0.01

@vectorized
def rastrigin(x: torch.Tensor) -> torch.Tensor:
    n = len(x)
    return 10*n + torch.sum(x.pow(2.0)-10*torch.cos(2*torch.pi*x), dim=-1)

@vectorized
def styblinski(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x.pow(4.0)-16*(x.pow(2.0))+5*x, dim=-1)/2

num_generations = 1000
dimension = 26
popsize = 14
stdev_init = 5
lr = 0.01
tolerance = 1e-3
num_actors = 'max'
# Format as Problem for evotorch
problems = [Problem("min", sphere, solution_length=dimension, initial_bounds=(-1,1), num_actors=num_actors)]#,
            # Problem("min", rastrigin, solution_length=dimension, initial_bounds=(-5.12,5.12), num_actors=num_actors),
            # Problem("min", styblinski, solution_length=dimension, initial_bounds=(-5,5), num_actors=num_actors)]
functions = ['Sphere', 'Rastrigin', 'Syblinski-Tang']

answers = [0.0, 0.0, -39.1662*dimension]
plt.figure()
for k in range(len(problems)):
    problem = problems[k]
    func = functions[k]
    # Algorithm
    solvers = [alg.CMAES(problem, stdev_init=stdev_init, popsize=popsize),
               alg.CEM(problem, popsize=popsize, parenthood_ratio=0.1, stdev_init=stdev_init),
               alg.PGPE(problem, popsize=popsize, center_learning_rate=lr, stdev_learning_rate=lr, stdev_init=stdev_init),
               alg.SNES(problem, stdev_init=stdev_init, popsize=popsize),
               alg.XNES(problem, stdev_init=stdev_init, popsize=popsize)]
    algorithms = ['CMAES', 'CEM', 'PGPE', 'SNES', 'XNES']

    plt.subplot(len(functions),1,k+1)
    plt.title(func)
    for j in range(len(solvers)):
        solver = solvers[j]
        # Logging utility
        logger = PandasLogger(solver)

        directory = 'Runs/'
        algorithm = algorithms[j]
        suffix = str(time.time())
        filename = directory+'log_'+func+'_'+algorithm+'_'+suffix+'.p'

        i = 0
        best = 10000000
        while i < num_generations and best > (answers[k]+tolerance):
            print(func + ' ' + algorithm + ' ' + str(i))
            solver.step()
            data = logger.to_dataframe()
            data.to_pickle(filename)
            best = data['pop_best_eval'].iloc[-1]
            i += 1
        data = pd.read_pickle(filename)
        problem.kill_actors()
        plt.plot(data['pop_best_eval'], label=algorithm)

    plt.axhline(y=answers[k], linestyle='--', color='black', label='Solution')
    solution = problem.status['best'].values
    pickle.dump(solution,open(directory+'sol_'+func+'_'+suffix+'.p', 'wb'))

    plt.legend()
plt.show()