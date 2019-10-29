import math

n_arms = 20
for  i in range(n_arms):
    print((math.exp(-i/19)-math.exp(-1))/(1-math.exp(-1)))