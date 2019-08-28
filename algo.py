import numpy.random as nprand
import numpy as np
import argparse
import math

class Bandit(object) :

    def __init__(self, n_bandits, p_bandits, rand_seed):
        self.n_bandits = n_bandits
        self._p_bandits = p_bandits
        self.rgen = nprand.RandomState(rand_seed)

    def pull(self, i) :
        if self.rgen.rand() < self._p_bandits[i]:
            return 1
        return 0
    
    def get_max_reward(self, horizon):
        return horizon*(max(self._p_bandits))

def ln(x):
    if x == 0:
        print x, "Hello"
    try :
        return np.log(x)
    except:
        print x
        return np.log(x)

def KL(x, y):
    eps = 1e-10
    x = min(max(x,eps), 1 - eps)
    y = min(max(y,eps), 1 - eps)
    return x*ln(x/y) + (1-x)*ln((1-x)/(1-y))

def max_q(p_emp, upper, conf_bound, precision=1e-6):

    lower = p_emp
    while upper - lower > precision:
        q = (lower + upper) / 2
        if KL(p_emp, q) > conf_bound :
            upper = q
        else:
            lower = q

    return (lower + upper)/2


def round_robin(bandit, horizon, *args):
    nb = bandit.n_bandits
    rew = 0
    for i in range(0,horizon):
        rew += bandit.pull(i%nb)
    return rew

def epsilon_greedy(bandit, horizon, seed, epsilon, *args):
    rgen = nprand.RandomState(seed)
    nb = bandit.n_bandits
    rew = 0
    p_emp = [0.0]*nb
    n_pulls = [0]*nb
    for _ in range(0, horizon):
        if rgen.rand() > epsilon :
            arm = p_emp.index(max(p_emp))
        else:
            arm = rgen.randint(0,nb)
        win = bandit.pull(arm)
        rew += win
        p_emp[arm] = (p_emp[arm]*n_pulls[arm] + win)/(n_pulls[arm] +1)
        n_pulls[arm] += 1

    return rew

def ucb(bandit, horizon, *args):
    nb = bandit.n_bandits
    rew = 0
    p_emp = [0.0]*nb
    n_pulls = [0]*nb
    ucb_a = [0.0]*nb
    for i in range(0, horizon):

        if i<nb :
            arm = i
        else :
            ucb_a = [p_emp[arm] + math.sqrt(2*ln(i)/n_pulls[arm]) for arm in range(0,nb)]
            arm = ucb_a.index(max(ucb_a))
        
        win = bandit.pull(arm)
        rew += win
        p_emp[arm] = (p_emp[arm]*n_pulls[arm] + win)/(n_pulls[arm] +1)
        n_pulls[arm] += 1

    return rew

def kl_ucb(bandit, horizon, *args):
    nb = bandit.n_bandits
    rew = 0
    p_emp = [0.0]*nb
    n_pulls = [0]*nb
    ucb_a = [0.0]*nb
    for i in range(0, horizon):

        if i<nb :
            arm = i
        else :
            d = ln(i) + 3*ln(ln(i))
            ucb_a = [max_q(p_emp[arm], 1.0, d/n_pulls[arm]) for arm in range(0,nb)]
            arm = ucb_a.index(max(ucb_a))
        
        win = bandit.pull(arm)
        rew += win
        p_emp[arm] = (p_emp[arm]*n_pulls[arm] + win)/(n_pulls[arm] +1)
        n_pulls[arm] += 1

    return rew

def thompson_sampling(bandit, horizon, seed, *args):
    rgen = nprand.RandomState(seed)
    nb = bandit.n_bandits
    rew = 0
    s = [0]*nb
    f = [0]*nb
    for _ in range(0, horizon):
        beta = [rgen.beta(s[i]+1,f[i]+1) for i in range(0,nb)]
        arm = beta.index(max(beta))
        win = bandit.pull(arm)
        rew += win
        s[arm] += win
        f[arm] += (1-win)
    
    return rew



algorithms = [round_robin, epsilon_greedy, ucb, kl_ucb, thompson_sampling]
algorithms = {algo.__name__.replace('_','-'): algo for algo in algorithms}

parser = argparse.ArgumentParser()
parser.add_argument('--instance',required=True)
parser.add_argument('--algorithm', choices=list(algorithms.keys()), required=True)
parser.add_argument('--randomSeed', type=int, required=True)
parser.add_argument('--epsilon', type=float, required=True)
parser.add_argument('--horizon', type=int, required=True)
args = parser.parse_args()
bandit_file = open(args.instance, 'r')
lines = bandit_file.readlines()
algorithm = args.algorithm.replace('-','_')

bandit_instance = Bandit(len(lines), [float(p) for p in lines], args.randomSeed)
algo_randSeed = args.randomSeed * 2
# rew = round_robin(bandit_instance, args.horizon)
# rew = epsilon_greedy(bandit_instance, args.horizon, args.epsilon, algo_randSeed)
#rew = ucb(bandit_instance, args.horizon)
rew = algorithms[args.algorithm](bandit_instance, args.horizon, algo_randSeed, args.epsilon)
reg = bandit_instance.get_max_reward(args.horizon) - rew
print ("%s, %s, %d, %s, %d, %s" %(args.instance, args.algorithm, args.randomSeed, args.epsilon, args.horizon, reg ))