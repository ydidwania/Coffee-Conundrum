import numpy as np
import matplotlib.pyplot as plt
import math

D_opt = np.loadtxt("graph_opt.txt", usecols=range(0,10), dtype=np.float64)

D_new = np.loadtxt("graph_new.txt", usecols=range(0,10), dtype=np.float64)

# D_opt_avg = np.zeros(shape=(81,10))
# D_new_avg = np.zeros(shape=(81,10))

D_opt_avg = np.zeros(shape=(72,10))
D_new_avg = np.zeros(shape=(72,10))

for i in range(72):
	for j in range(20):
		D_opt_avg[i] = np.add(D_opt_avg[i],D_opt[j+20*i])
		D_new_avg[i] = np.add(D_new_avg[i],D_new[j+20*i])

	
	D_opt_avg[i] = D_opt_avg[i]/20
	D_new_avg[i] = D_new_avg[i]/20
# print(D_opt_avg)
np.savetxt("opt_avg.txt", D_opt_avg, fmt="%4d", delimiter=",", newline="\n")
np.savetxt("new_avg.txt", D_new_avg, fmt="%4d", delimiter=",", newline="\n")
# np.savetxt('graph_opt_avg.txt',D_opt_avg)

# N_s = 1000
# B_1 = []
# earning = [] 
# no_of_customers = []
# # no_of_small_cups_sold = []
# for i in range(len(D_opt_avg)):
# 	if(int(D_opt_avg[i][0]) == 1000):
# 		B_1.append(D_opt_avg[i][1])
# 		earning.append(D_opt_avg[i][2]+100*D_opt_avg[i][6]) #earning + no_of small cup left*Ps
# 		no_of_customers.append(D_opt_avg[i][-1]+D_opt_avg[i][6])#no_of customers served + no_of small cup left
		
# # print(B_1,earning)

# B_1 = [math.log10(i) for i in B_1]
# print(B_1)
# plt.figure(1)
# plt.title('Ns = 1000')
# plt.ylabel('Revenue')
# plt.xlabel('log(dProfit)')
# plt.plot(B_1,earning)
# plt.show()  

# plt.figure(2)
# plt.title('Ns = 1000')
# plt.ylabel('No of customers served')
# plt.xlabel('log(dProfit)')
# plt.plot(B_1,no_of_customers)
# plt.show()  