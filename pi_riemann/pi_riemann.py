#Aproximacion de PI sin parallelizacion
from math import sqrt
import cProfile

def fun(x):
  return sqrt(1-x**2)

def pi_riemann(n):
  dx = 1/n
  sr = 0
  sr = sum(fun(i * dx) * dx for i in range(n))
  print(4*sr)

#pi_riemann(100000)

# Run the profiling
cProfile.run('pi_riemann(100000)')