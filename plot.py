from matplotlib import pyplot as plt

f = open('output.txt','r')

res = [x.rstrip().split(', ') for x in f.readlines()]

x = [50, 200, 800, 3200, 12800, 51200, 204800]

ins1 = [ r for r in res if r[0]=='../instances/i-1.txt']
algo_dict = {}

# algo_dict = { round-robin : {50: [], 200: [] ,...}, ucb:{..},..}
for i in ins1:
    if i[1] == 'epsilon-greedy':
        algo_dict.setdefault(i[1]+'-'+i[3],{}).setdefault(i[4],[]).append(i)
    else:
        algo_dict.setdefault(i[1],{}).setdefault(i[4],[]).append(i)

y = {}
for key,item in algo_dict.iteritems():
    for k,i in item.iteritems():
        y.setdefault(key,[0]*7)[x.index(int(k))] = sum(map(float,zip(*i)[5]))/len(i)

#for k,i in algo_dict['epsilon-greedy'].iteritems():
#    y.set



#for key in algo_dict.Keys():
#    for l in algo_dict[key]:
#        y[x.index(int(l[4]))] += float(l[5])

#y = [i/50 for i in y]

plt.xscale('log')
plt.yscale('log')
plt.title('Bandit Instance i-1')
plt.ylabel('Regret (log)')
plt.xlabel('Horizon (log)')
plt.plot(x,y['round-robin'],	  'C0', label='round-robin')
plt.plot(x,y['epsilon-greedy-0.002'],	  'C1', label='epsilon-greedy-0.002')
plt.plot(x,y['epsilon-greedy-0.02'],	  'C2', label='epsilon-greedy-0.02')
plt.plot(x,y['epsilon-greedy-0.2'],	  'C3', label='epsilon-greedy-0.2')
plt.plot(x,y['ucb'],	    	  'C4',label='ucb')
plt.plot(x,y['kl-ucb'],	    	  'C5',label='kl-ucb')
plt.plot(x,y['thompson-sampling'],'C6',label='thompson-sampling')
plt.legend(loc='upper left')
plt.show()    
#print algo_dict






