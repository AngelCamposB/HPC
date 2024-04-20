###Usando mpi4py
# Escribir el código en un archivo Python
codigo_mpi = """
from math import sqrt
from mpi4py import MPI
import cProfile

def fun(x):
    return sqrt(1 - x**2)

def pi_riemann_mpi(n): 
    comm = MPI.COMM_WORLD #Comunicador
    rank = comm.Get_rank()  #ID
    size = comm.Get_size()  #No.Procesos

    dx = 1.0 / n
    local_n = n // size
    local_start = rank * local_n
    local_end = (rank + 1) * local_n if rank != size - 1 else n

    local_sum = sum(fun(i * dx) * dx for i in range(local_start, local_end))

    total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

    if rank == 0:
        pi = 4 * total_sum
        print(f"Approximation of PI is {pi}")

if __name__ == "__main__":
    
    n=1000
    pi_riemann_mpi(n)
    cProfile.run('pi_riemann(1000)')

"""

# Escribir el código en un archivo en el entorno de Colab
with open("mpi_example2.py", "w") as file:
    file.write(codigo_mpi)

# Ejecutar el programa MPI
# Ejecutar el programa MPI permitiendo la sobresuscripción de procesos
!mpiexec --allow-run-as-root --oversubscribe -n 2 python mpi_example.py