import numpy.random as nprand
import numpy as np
import argparse
from math import sqrt
import matplotlib.pyplot as plt


pho_s = 100.0 
pho_l = 120.0
cost_s = 50.0
cost_l = 60.0
norm_min = min(-cost_l + pho_s, pho_l - cost_l, pho_s - cost_s)
norm_max = max(-cost_l + pho_s, pho_l - cost_l, pho_s - cost_s)


def cost_arm(win,price,cost):
    if(cost == 'loss_in_profit'):
        # out = win*(pho_l - price)/(pho_l - pho_s)
        if(price==pho_s):
            out = 1
        else:
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

def ucb_bv1(B,B_1,c_type,r_type,bandit, *args):
    # print("Small Cup : %d // %d"%(pho_s,cost_s))
    # print("Large Cup : %d // %d"%(pho_l,cost_l))
    # print("norm_min = min(-cost_l + pho_s, pho_l - cost_l, pho_s - cost_s)")
    # print("norm_max = max(-cost_l + pho_s, pho_l - cost_l, pho_s - cost_s)")

    nb = bandit.n_bandits
    reg = 0
    # B = 100
    earnings,i = 0,0
    passive_n_pulls = [0]*nb
    active_n_pulls = [0]*nb
    active_n_success = [0]*nb
    ucb_a = [0.0]*nb
    passive_cost, passive_rew = [0.0]*nb, [0.0]*nb 
    active_cost, active_rew = [0.0]*nb, [0.0]*nb 
    lmbda = 1e-6
    D = lambda i,r,c,n : (r/c) + ((1+1/lmbda)*sqrt(ln(i)/n))/(lmbda - sqrt(ln(i)/n)) 
    # print("Starting out with Budget = ",B)
    # for _ in range(2):hhttps://github.com/ydidwania/Coffee-Conundrumttps://github.com/ydidwania/Coffee-Conundrum
    wins,i = 0,0
    avg_l_price = 0.0
    avg_off_price = 0.0
    x = []
    y = []
    n_rejects  = 0
    # while (cost[0]<B):
    while (sum(active_cost[1:])<=B_1-1+n_rejects and active_cost[0]<B):
        if i<nb :
            arm = i
        else :
            ucb_a = [D(i,passive_rew[arm], max(passive_cost[arm],1e-6), passive_n_pulls[arm]) for arm in range(nb)]
            arm = ucb_a.index(max(ucb_a))
        i += 1
        win = bandit.pull(arm)
        price = bandit.get_price(arm)
        avg_off_price  += (price)

        if(price == pho_s):
            passive_cost[0] += 1
            active_cost[0] += 1
            passive_rew[0]  += 0
            passive_n_pulls[0] +=1
        else:
            if(win==1):
                update = cost_arm(win,price,c_type)
                active_cost[arm] += update
                # passive_cost[arm] += update
                for j in range(1,arm+1):
                    passive_cost[j] = passive_cost[j] + cost_arm(win,bandit.get_price(j),c_type)
                    passive_rew[j] += revenue_arm(win,bandit.get_price(j),r_type)
                    passive_n_pulls[j] += 1 
            else:
                active_cost[0] = active_cost[0] + 1
                passive_cost[0] = passive_cost[0] + 1
                n_rejects  += 1
                passive_n_pulls[0] += 1
                active_cost[arm] = active_cost[arm] + 1
                for k in range(arm,nb):
                    passive_cost[k] = passive_cost[k] + 1
                    passive_rew[k] += revenue_arm(win,bandit.get_price(k),r_type)
                    passive_n_pulls[k] += 1
               
        active_rew[arm]  += revenue_arm(win,price,r_type)

        active_n_pulls[arm] += 1
        active_n_success[arm] += win
        if ((win) and (price>pho_s)):
            wins += win
            avg_l_price = (avg_l_price*(wins-1) + (price))/wins

        # print (" N = %d, Offered_price = %.2f, result=%d, Small=%d, "%(i, price, win, B -(active_cost[0])))
        # sum_of_cost ==> budget till now
        
        # x.append(i)/
        # plt.pause(0.05)
    
    avg_off_price = avg_off_price/i
        # print ("Total Earnings = ", earnings)
        # print ("Avg Large Price = ", avg_l_price) 
        
        # B = 200
    # plt.title('Regret vs budget')
    # plt.ylabel('No of small cups_gone')
    # plt.xlabel('Time')

    # plt.plot(x,y)
    # plt.show()  
    earnings = sum(active_rew)*(pho_l - pho_s ) + i*(pho_s)
    # earnings = sum(rew)*(norm_max - norm_min ) + 200*(norm_min)
    # print ("Total Earnings = ", earnings)
    # print ("Avg Large Price = ", avg_l_price) 
    # print ("Avg offered Price = ", avg_off_price) 
    # print("Residual B_1 = ",B_1-sum(active_cost[1:])) 
    # print('Remaining Small Cups = ',B-active_cost[0])
    # print("No of large cup sold = ",i-active_cost[0])
    # # reward_total = sum(active_rew)
    n_probs = [active_n_success[i]/active_n_pulls[i] for i in range(nb)]
    n_probs = [str(i) for i in n_probs]
    print(" ".join(n_probs))
    print(" ".join([str(i) for i in active_n_pulls]))
    # print(B, B_1, earnings, avg_l_price, avg_off_price, B_1+B-sum(active_cost[1:]), B-active_cost[0], i-active_cost[0],active_cost[0],i)
    return earnings

algorithms = [ucb_bv1]
algorithms = {algo.__name__.replace('_','-'): algo for algo in algorithms}
cost_type       = ['loss_in_profit','cups_gone']
revenue_type    = ['profit','revenue']

parser = argparse.ArgumentParser()
parser.add_argument('--instance',required=True)
parser.add_argument('--algorithm', choices=list(algorithms.keys()), required=True)
parser.add_argument('--cost',choices=cost_type, required=True)
parser.add_argument('--revenue',choices=revenue_type, required=True)
parser.add_argument('--budget',type=int, required=True)
parser.add_argument('--cups',type=int, required=True)
parser.add_argument('--seed',type=int, required=True)

args = parser.parse_args()
bandit_file = open(args.instance, 'r')
lines = bandit_file.readlines()
algorithm = args.algorithm.replace('-','_')
bandit_instance = Bandit(len(lines), [float(p) for p in lines], args.seed, [pho_s + (pho_l-pho_s)*i/19.0 for i in range(len(lines))])

reward = algorithms[args.algorithm](args.cups,args.budget,args.cost,args.revenue,bandit_instance)

# reg = bandit_instance.get_max_reward(args.horizon) - rew
# print ("%s, %s, %d, %s, %d, %s\n" %(args.instance, args.algorithm, args.randomSeed, args.epsilon, args.horizon, reg ))
