import numpy.random as nprand
import numpy as np
import argparse
import math

pho_s = 100.0
pho_l = 200.0


class Bandit(object) :

    def __init__(self, n_bandits, p_bandits, rand_seed, price):
        self.n_bandits = n_bandits
        self._p_bandits = p_bandits
        self.rgen = nprand.RandomState(rand_seed)
        self.price = price

    def pull(self, i) :
        if self.rgen.rand() < self._p_bandits[i]:
            return 1
        return 0
    
    def get_max_reward(self, horizon):
        return horizon*(max(self._p_bandits))
    
    def get_price(self, i):
        return self.price[i]

def ln(x):
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
    reg = 0
    wins,earnings = 0,0
    avg_l_price = 0.0
    p_emp = [0.0]*nb
    n_pulls = [0]*nb
    ucb_a = [0.0]*nb
    for i in range(0, horizon):

        if i<nb :
            arm = i
        else :
            ucb_a = [abs(p_emp[arm] - 0.5) + math.sqrt(2*ln(i)/n_pulls[arm]) for arm in range(0,nb)]
            arm = ucb_a.index(max(ucb_a))
        
        win = bandit.pull(arm)
        price = bandit.get_price(arm)
        wins += win
        reg += (win*(pho_l - price)/(pho_l - pho_s)) + (1-win)*1
        earnings += win*price + (1 - win)*pho_s
        if (win):
            avg_l_price = (avg_l_price*(wins-1) + (price))/wins
        p_emp[arm] = (p_emp[arm]*n_pulls[arm] + win)/(n_pulls[arm] +1)
        n_pulls[arm] += 1
        print (" N = %d, Offered_price = %.2f, result=%d, Small=%d, Regret=%.2f"%(i, price, win, 100-(i+1-wins), reg))

    print ("Total Earnings = ", earnings)
    print ("Avg Large Price = ", avg_l_price) 
    return reg

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

bandit_instance = Bandit(len(lines), [float(p) for p in lines], args.randomSeed, [pho_s + (pho_l-pho_s)*i/19.0 for i in range(len(lines))])
algo_randSeed = args.randomSeed * 2
rew = algorithms[args.algorithm](bandit_instance, args.horizon, algo_randSeed, args.epsilon)
# reg = bandit_instance.get_max_reward(args.horizon) - rew
# print ("%s, %s, %d, %s, %d, %s\n" %(args.instance, args.algorithm, args.randomSeed, args.epsilon, args.horizon, reg ))
