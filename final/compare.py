import numpy as np
import matplotlib.pyplot as plt
import math

D_opt = np.loadtxt("prob_opt.txt", usecols=range(0,20), dtype=np.float64)

D_new = np.loadtxt("prob_new.txt", usecols=range(0,20), dtype=np.float64)

# D_opt_avg = np.zeros(shape=(81,10))
# D_new_avg = np.zeros(shape=(81,10))

D_opt_avg = np.zeros(shape=(64,20))
D_new_avg = np.zeros(shape=(64,20))

for i in range(64):
	for j in range(20):
		D_opt_avg[i] = np.add(D_opt_avg[i],D_opt[2*j+40*i]) #Every alternate
		D_new_avg[i] = np.add(D_new_avg[i],D_new[2*j+40*i]) #Every alternate

	
	D_opt_avg[i] = D_opt_avg[i]/20
	D_new_avg[i] = D_new_avg[i]/20
# print(D_opt_avg)
np.savetxt("opt_prob_avg.txt", D_opt_avg, fmt="%.4f", delimiter=",", newline="\n")
np.savetxt("new_prob_avg.txt", D_new_avg, fmt="%.4f", delimiter=",", newline="\n")
# np.savetxt('graph_opt_avg.txt',D_opt_avg)

# N_s = 1000
# B_1 = []
# B_1_new = []
# earning = [] 
# earning_new = []
# no_of_customers = []
# no_of_customers_new = []
# # no_of_small_cups_sold = []
# for i in range(len(D_opt_avg)):
# 	if(int(D_opt_avg[i][0]) == 1000):
# 		B_1.append(D_opt_avg[i][1])
# 		earning.append(D_opt_avg[i][2]+100*D_opt_avg[i][6]) #earning + no_of small cup left*Ps
# 		no_of_customers.append(D_opt_avg[i][-1]+D_opt_avg[i][6])#no_of customers served + no_of small cup left
	
# 	if(int(D_new_avg[i][0]) == 1000):
# 		B_1_new.append(D_new_avg[i][1])
# 		earning_new.append(D_new_avg[i][2]+100*D_new_avg[i][6]) #earning + no_of small cup left*Ps
# 		no_of_customers_new.append(D_new_avg[i][-1]+D_new_avg[i][6])#no_of customers served + no_of small cup left
	

# # print(B_1,earning)

# B_1 = [math.log10(i) for i in B_1]

# B_1_new = [math.log10(i) for i in B_1_new]

# # print(B_1,B_1_new)

# plt.figure(1)
# plt.title('Ns = 1000')
# plt.ylabel('Revenue')
# plt.xlabel('log(delProfit)')
# plt.plot(B_1,earning)
# plt.show()  

# plt.figure(2)
# plt.title('Ns = 1000')
# plt.ylabel('No of customers served')
# plt.xlabel('log(delProfit)')
# plt.plot(B_1,no_of_customers)
# plt.show()  


# B_2 = 1000
# N_s2 = []
# earning2 = [] 
# no_of_customers2 = []
# for i in range(len(D_opt_avg)):
# 	if(int(D_opt_avg[i][1]) == 1000):
# 		N_s2.append(D_opt_avg[i][0])
# 		earning2.append(D_opt_avg[i][2]+100*D_opt_avg[i][6]) #earning + no_of small cup left*Ps
# 		no_of_customers2.append(D_opt_avg[i][-1]+D_opt_avg[i][6])#no_of customers served + no_of small cup left

# print(N_s2)
# plt.figure(3)
# plt.title('B = 1000')
# plt.ylabel('Revenue')
# plt.xlabel('Ns')
# plt.plot(N_s2,earning2)
# plt.show()  

# plt.figure(4)
# plt.title('B = 1000')
# plt.ylabel('No of customers served')
# plt.xlabel('Ns')
# plt.plot(N_s2,no_of_customers2)
# plt.show()  


# plt.figure(5)
# plt.title('Ns = 1000')
# plt.ylabel('Revenue')
# plt.xlabel('log(delProfit)')
# plt.plot(B_1,earning,'r')
# plt.plot(B_1_new,earning_new,'c')
# plt.show()  
