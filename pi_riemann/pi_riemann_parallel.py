#Aproximacion de PI CON parallelizacion
from math import sqrt
import multiprocessing
import cProfile

def fun(x):
  return sqrt(1 - x**2)

def partial_sum(args):
  #xi = i * dx
  n, start, end, dx = args
  return sum(fun(i * dx) * dx for i in range(start, end))

def pi_riemann_parallel(n, num_processes=4):
  pool = multiprocessing.Pool()
  dx = 1 / n
  ranges = [(n, int(i * n / num_processes), int((i + 1) * n / num_processes), dx) for i in range(num_processes)]
  
  results = pool.map(partial_sum, range)
  total_sum = sum(results)
  print(4 * total_sum)

# Ejemplo de uso:
#pi_riemann_parallel(100_000)

# Run the profiling
#cProfile.run('pi_riemann_parallel(100000)')
cProfile.run('pi_riemann_parallel(100000)', sort='cumtime')