import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import random as sparse_random, csr_matrix
from time import perf_counter

# Parameters to vary
sizes = [100, 500, 1000, 2000, 4000]        # Matrix sizes (square)
sparsity_levels = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99]

results = {}

for sparsity in sparsity_levels:
    times = []
    for size in sizes:
        # Generate two random sparse matrices A and B
        A = sparse_random(size, size, density=sparsity, format='csr', dtype=np.float32)
        B = sparse_random(size, size, density=sparsity, format='csr', dtype=np.float32)

        # Time the multiplication
        start = perf_counter()
        C = A.dot(B)
        end = perf_counter()

        elapsed_time = end - start
        times.append(elapsed_time)
        print(f"Size: {size}x{size}, Sparsity: {sparsity:.2f}, Time: {elapsed_time:.6f} sec")

    results[sparsity] = times

# Plotting
plt.figure(figsize=(10, 6))
for sparsity, times in results.items():
    plt.plot(sizes, times, marker='o', label=f"Sparsity {sparsity:.2f}")

plt.title("CSR Matrix Multiplication Time vs Size")
plt.xlabel("Matrix Size (N x N)")
plt.ylabel("Time (seconds)")
plt.legend(title="Sparsity")
plt.grid(True)
plt.tight_layout()
plt.savefig('csr_matrix_multiplication.png')
