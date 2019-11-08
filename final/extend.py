import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import math

def nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# D_opt_avg[i][2]+100*D_opt_avg[i][6]

# (B, B_1, earnings, avg_l_price, avg_off_price, B+B_1-sum(cost[1:]), B-cost[0], i-cost[0],cost[0],i)
runs_opt = np.loadtxt("opt_avg.txt", usecols=range(0,10), dtype=np.float64)
runs_new = np.loadtxt("new_avg.txt", usecols=range(0,10), dtype=np.float64)

prob_opt = np.loadtxt("opt_prob_avg.txt", usecols=range(0,20), dtype=np.float64)
prob_new = np.loadtxt("new_prob_avg.txt", usecols=range(0,20), dtype=np.float64)

final_opt = np.zeros(shape=(64,5))
final_new = np.zeros(shape=(64,5))

a = 1 #alpha
pho_s = 100
pho_l = 120
x,y,z =[],[],[]
x_ticks = []
for i in range(64):
    Ns = runs_opt[i][0]
    B = runs_opt[i][1]
    N_opt = runs_opt[i][-1]
    N_new = runs_new[i][-1]
    rev_opt  = runs_opt[i][2]+100*runs_opt[i][6] + 100*(int(runs_opt[i][5]))
    rev_new = runs_new[i][2]+100*runs_new[i][6] + 100*(int(runs_new[i][5]))
    cus_opt = runs_opt[i][-1] + runs_opt[i][6] + int(runs_opt[i][5])
    cus_new = runs_new[i][-1] + runs_new[i][6] + int(runs_new[i][5])
    

    if(Ns == 20000):
        x.append(B)
        x_ticks.append(math.log(B))
        y.append(rev_opt)
        z.append(runs_opt[i][2])
    # C = max(N_new, N_opt)
    # if(C == N_new):
    #     K_opt = (a*C - N_opt)/(C - N_opt)
    #     p_opt = pho_s + (nearest_idx(prob_opt[i],K_opt)/19)*(pho_l - pho_s)
    #     rev_opt += p_opt*(a*C - N_opt) 
    # else:
    #     K_new = (a*C - N_new)/(C - N_new)
    #     p_new = pho_s + (nearest_idx(prob_new[i],K_new)/19)*(pho_l - pho_s)
    #     rev_new += p_new*(a*C - N_new)
    # pc_diff = (rev_new - rev_opt)/rev_opt
    # #     # if(N_opt<C and N_new < C):
    # print(B, Ns, rev_opt, rev_new, rev_new>rev_opt, 100*pc_diff, sep='\t' )
    # else:
        # print(B, Ns, sep='\t' )
print(y)
# print(z)

fig1, ax1 = plt.subplots()

# fig1.figure(1)
plt.title('Ns = 20000', fontsize='22')
plt.ylabel('Revenue', fontsize='20')
plt.xlabel('\u0394Profit', fontsize='20')
ax1.plot(x,z, 'b')
ax1.plot(x,y, 'r')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.legend(['Stopping time', 'Exhausting Residuals'])
# ax1.set_xticks(x_ticks)
# ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.show()  


# plt.figure(1)
# plt.title('Ns = 500', fontsize=22)
# plt.ylabel('Revenue')
# plt.xlabel('log(\u0394Profit)')
# plt.plot(x,y, 'b')
# plt.plot(x,z, 'r')

# plt.show()  
    



    
    

    

    



