import numpy.random as nprand
import numpy as np
import argparse
from math import sqrt

pho_s = 100.0
pho_l = 120.0
cost_s = 50.0
cost_l = 60.0
norm_min = min(-cost_l + pho_s, pho_l - cost_l, pho_s - cost_s)
norm_max = max(-cost_l + pho_s, pho_l - cost_l, pho_s - cost_s)

def cost_arm(win,price,cost):
    if(cost == 'loss_in_profit'):
        out = win*(pho_l - price)/(pho_l - pho_s)
    elif(cost == 'cups_gone'):
        out = win*0 + (1 - win)*1.0
    return out

def revenue_arm(win,price,revenue):
    if (revenue == 'revenue'):
        out = (win*(price) + (1 - win)*(pho_s) - pho_s)/(pho_l - pho_s )
    elif (revenue == 'profit'):
        out = (win*(price-cost_l) + (1 - win)*(pho_s-cost_s) - norm_min)/(norm_max - norm_min )
    return out

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
# 
def ln(x):
    return np.log(x)

def ucb_bv1(c_type,r_type,bandit, horizon, *args):
    print("Small Cup : %d // %d"%(pho_s,cost_s))
    print("Large Cup : %d // %d"%(pho_l,cost_l))
    print("norm_min = min(-cost_l + pho_s, pho_l - cost_l, pho_s - cost_s)")
    print("norm_max = max(-cost_l + pho_s, pho_l - cost_l, pho_s - cost_s)")

    nb = bandit.n_bandits
    reg = 0
    B = 500
    earnings,i = 0,0
    n_pulls = [0]*nb
    ucb_a = [0.0]*nb
    cost, rew = [0.0]*nb, [0.0]*nb 
    lmbda = 1e-6
    D = lambda i,r,c,n : (r/c) + ((1+1/lmbda)*sqrt(ln(i)/n))/(lmbda - sqrt(ln(i)/n)) 
    print("Starting out with Budget = ",B)
    # for _ in range(2):hhttps://github.com/ydidwania/Coffee-Conundrumttps://github.com/ydidwania/Coffee-Conundrum
    wins,i = 0,0
    avg_l_price = 0.0
    while (B -(i-wins)) >0:
        if i<nb :
            arm = i
        else :
            ucb_a = [D(i,rew[arm], max(cost[arm],1e-6), n_pulls[arm]) for arm in range(nb)]
            arm = ucb_a.index(max(ucb_a))
        i += 1
        win = bandit.pull(arm)
        price = bandit.get_price(arm)
        wins += win
        cost[arm] += cost_arm(win,price,c_type)
        rew[arm]  += revenue_arm(win,price,r_type)
        n_pulls[arm] += 1
        if (win):
            avg_l_price = (avg_l_price*(wins-1) + (price))/wins
        print (" N = %d, Offered_price = %.2f, result=%d, Small=%d, "%(i, price, win, 500 -(i-wins)))
        # print ("Total Earnings = ", earnings)
        # print ("Avg Large Price = ", avg_l_price) 
        # print("BOOST BY 100 SMALL CUPS")
        # B = 200

    # earnings = sum(rew)*(pho_l - pho_s ) + i*(pho_s)
    earnings = sum(rew)*(norm_max - norm_min ) + 200*(norm_min)
    print ("Total Earnings = ", earnings)
    print ("Avg Large Price = ", avg_l_price) 
    return reg

algorithms = [ucb_bv1]
algorithms = {algo.__name__.replace('_','-'): algo for algo in algorithms}
cost_type       = ['loss_in_profit','cups_gone']
revenue_type    = ['profit','revenue']

parser = argparse.ArgumentParser()
parser.add_argument('--instance',required=True)
parser.add_argument('--algorithm', choices=list(algorithms.keys()), required=True)
parser.add_argument('--epsilon', type=float, required=True)
parser.add_argument('--horizon', type=int, required=True)
parser.add_argument('--cost',choices=cost_type, required=True)
parser.add_argument('--revenue',choices=revenue_type, required=True)

args = parser.parse_args()
bandit_file = open(args.instance, 'r')
lines = bandit_file.readlines()
algorithm = args.algorithm.replace('-','_')
print('cost_type: ', args.cost )
print('revenue_type: ',args.revenue)
bandit_instance = Bandit(len(lines), [float(p) for p in lines], 20, [pho_s + (pho_l-pho_s)*i/19.0 for i in range(len(lines))])

rew = algorithms[args.algorithm](args.cost,args.revenue,bandit_instance, args.horizon, 40, args.epsilon)
# reg = bandit_instance.get_max_reward(args.horizon) - rew
# print ("%s, %s, %d, %s, %d, %s\n" %(args.instance, args.algorithm, args.randomSeed, args.epsilon, args.horizon, reg ))
