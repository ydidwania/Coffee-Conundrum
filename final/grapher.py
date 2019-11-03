import os
import matplotlib.pyplot as plt


B = [100,200,500,1000,2000,5000,10000,20000] # cups
B_1 = [100,200,500,1000,2000,5000,10000,20000]
rand_seed = [str(i) for i in range(20)]

# B = [100]
# B_1 = [1000]
for i in range(len(B_1)):
	for j in range(len(B)):
		for r in rand_seed:
			os.system('python3 graph.py --instance linear_20.txt --algorithm ucb-bv1 --cost loss_in_profit --revenue revenue --budget '+str(B_1[i])+ ' --cups '+str(B[j]) +' --seed '+ r +' >> prob_opt.txt')	
			os.system('python3 graph_new.py --instance linear_20.txt --algorithm ucb-bv1 --cost loss_in_profit --revenue revenue --budget '+str(B_1[i])+ ' --cups '+str(B[j]) +' --seed '+ r +' >> prob_new.txt')

# y = []:
# for i in range(len(d)):
# 	if(i!=len(d)-1):
# 		y.append(float(d[i][:len(d[i])-2]))
# 	else:
# 		y.append(float(d[i][:len(d[i])-1]))
	

# plt.plot(B_1,y)
# plt.show()