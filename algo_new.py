import numpy.random as nprand
import numpy as np
import argparse
from math import sqrt

pho_s = 100.0
pho_l = 105.0
cost_s = 50.0
cost_l = 60.0


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
            ucb_a = [abs(p_emp[arm] - 0.5) + sqrt(2*ln(i)/n_pulls[arm]) for arm in range(0,nb)]
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

def ucb_bv1(bandit, horizon, *args):
    print("Small Cup : %d // %d"%(pho_s,cost_s))
    print("Large Cup : %d // %d"%(pho_l,cost_l))
    norm_min = min(-cost_l + pho_s, pho_l - cost_l, pho_s - cost_s)
    norm_max = max(-cost_l + pho_s, pho_l - cost_l, pho_s - cost_s)
    nb = bandit.n_bandits
    reg = 0
    B = 100
    wins,earnings,i = 0,0,0
    avg_l_price = 0.0
    n_pulls = [0]*nb
    ucb_a = [0.0]*nb
    cost, rew = [0.0]*nb, [0.0]*nb 
    lmbda = 1e-6
    D = lambda i,r,c,n : (r/c) + ((1+1/lmbda)*sqrt(ln(i)/n))/(lmbda - sqrt(ln(i)/n)) 
    print("Starting out with Budget = ",B)
    for _ in range(2):
        while sum(cost) < B:
            if i<nb :
                arm = i
            else :
                ucb_a = [D(i,rew[arm], max(cost[arm],1e-6), n_pulls[arm]) for arm in range(nb)]
                arm = ucb_a.index(max(ucb_a))
            i += 1
            win = bandit.pull(arm)
            price = bandit.get_price(arm)
            wins += win
            cost[arm] += win*0 + (1 - win)*1.0
            rew[arm] += (win*(price-cost_l) + (1 - win)*(pho_s-cost_s) - norm_min)/(norm_max - norm_min )
            n_pulls[arm] += 1

            if (win):
                avg_l_price = (avg_l_price*(wins-1) + (price))/wins
            
            print (" N = %d, Offered_price = %.2f, result=%d, Small=%d, "%(i, price, win, 100 - sum(cost)))
        print("BOOST BY 100 SMALL CUPS")
        B = 200

    earnings = sum(rew)*(norm_max - norm_min ) + i*(norm_min)
    print ("Total Earnings = ", earnings)
    print ("Avg Large Price = ", avg_l_price) 
    return reg



algorithms = [ucb, ucb_bv1, thompson_sampling]
algorithms = {algo.__name__.replace('_','-'): algo for algo in algorithms}

parser = argparse.ArgumentParser()
parser.add_argument('--instance',required=True)
parser.add_argument('--algorithm', choices=list(algorithms.keys()), required=True)
parser.add_argument('--epsilon', type=float, required=True)
parser.add_argument('--horizon', type=int, required=True)
args = parser.parse_args()
bandit_file = open(args.instance, 'r')
lines = bandit_file.readlines()
algorithm = args.algorithm.replace('-','_')

bandit_instance = Bandit(len(lines), [float(p) for p in lines], 20, [pho_s + (pho_l-pho_s)*i/19.0 for i in range(len(lines))])

rew = algorithms[args.algorithm](bandit_instance, args.horizon, 40, args.epsilon)
# reg = bandit_instance.get_max_reward(args.horizon) - rew
# print ("%s, %s, %d, %s, %d, %s\n" %(args.instance, args.algorithm, args.randomSeed, args.epsilon, args.horizon, reg ))
