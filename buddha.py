from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64
from PIL import Image
import time
import numpy as np


@cuda.jit(device=True)
def check_point(a: float, b: float, iterations: int) -> bool:
    zr,zi = a, b
    zr2, zi2 = a*a, b*b
    for _ in range(iterations):
        zi, zr = 2 * zr * zi + b, zr2 - zi2 + a
        zr2, zi2 = zr * zr, zi * zi
        if zr2 + zi2 > 1e10:
            return True
    return False

@cuda.jit(device=True)
def get_trajectory(a: float, b: float, iterations: int, out):
    width, height = out.shape
    zr,zi = a, b
    zr2, zi2 = a*a, b*b
    for _ in range(iterations):
        zi, zr = 2 * zr * zi + b, zr2 - zi2 + a
        zr2, zi2 = zr * zr, zi * zi
        if zr2 + zi2 > 1e10: return
        else:
            x = int((zr + 2) * width / 3)
            y = int((zi + 1.5) * height / 3)
            if 0 < x < width and 0 < y < height:
                cuda.atomic.add(out, (x, y), 70)

@cuda.jit
def generate_buddha(rng_states, iterations, out):
    thread_id = cuda.grid(1)
    for i in range(1e5):
        x = xoroshiro128p_uniform_float64(rng_states, thread_id)
        y = xoroshiro128p_uniform_float64(rng_states, thread_id)
        a, b = 3.0*x - 2.0, 3.0*y - 1.5
        if check_point(a, b, iterations):
            get_trajectory(a, b, iterations, out)

threads_per_block = 64
blocks = 256
iterations = 10000
dim = 10000

start = time.time()

rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=1)
out = np.zeros((dim, dim), dtype=np.int32)
g_out = cuda.to_device(out)
generate_buddha[blocks, threads_per_block](rng_states, iterations, g_out)
Image.fromarray(g_out.copy_to_host()).save("buddha10k.png")
print(f"--- {time.time() - start} seconds ---")
    