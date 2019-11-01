<<<<<<< HEAD
n_arms = 20
for  i in range(n_arms):
    print(1 - i/19.0)
=======
import math

n_arms = 20
for  i in range(n_arms):
    print((math.exp(-i/19)-math.exp(-1))/(1-math.exp(-1)))
>>>>>>> 29122570c47411ac63f6b3bab533a501bf82d06e
