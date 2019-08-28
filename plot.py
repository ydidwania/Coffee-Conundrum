from matplotlib import pyplot as plt

f = open('output.txt','r')

res = [x.rstrip().split(', ') for x in f.readlines()]

x = [50, 200, 800, 3200, 12800, 51200, 204800]

ins1 = [ r for r in res if r[0]=='../instances/i-1.txt']
algo_dict = {}

for i in ins1:
    algo_dict.setdefault(i[1],{}).setdefault(i[4],[]).append(float(i[5]))

y = {}
for key,item in algo_dict.iteritems():
    for k,i in item.iteritems():
        y.setdefault(key,[0]*7)[x.index(int(k))] = sum(i)/len(i)

     
   

#for key in algo_dict.Keys():
#    for l in algo_dict[key]:
#        y[x.index(int(l[4]))] += float(l[5])

#y = [i/50 for i in y]

plt.xscale('log')
plt.yscale('log')
plt.plot(x,y['round-robin'],'b')
print y['ucb']
plt.plot(x,y['ucb'],'r')
plt.plot(x,y['kl-ucb'],'g')
plt.plot(x,y['thompson-sampling'],'y')
plt.show()    
#print algo_dict






