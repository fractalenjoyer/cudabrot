from numba import cuda
from PIL import Image
import numpy as np
import time

@cuda.jit(device=True)
def calc(a, b, iterations):
    zr = a
    zi = b 
    zr2 = a*a
    zi2 = b*b
    for i in range(iterations):
        zi = 2 * zr * zi + b
        zr = zr2 - zi2 + a
        zr2 = zr * zr
        zi2 = zi * zi
        if zr2 + zi2 > 1e10:
            if i < 256: return i
            else: return 255
    return 0

@cuda.jit
def calculate_mandel(iterations, out):
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    
    width = out.shape[0]
    height = out.shape[1]
    
    for x in range(startx, width, stridex):
        for y in range(starty, height, stridey):
            a = 3.*x/width - 2.
            b = 3.*y/height - 1.5
            out[x,y] = calc(a, b, iterations)
            
blocks = (512, 512)
threads_per_block = (32, 32)
iterations = 10000

imgarr = np.zeros((1000, 2000), dtype=np.uint8)
start = time.time()
device_arr = cuda.to_device(imgarr)
calculate_mandel[blocks, threads_per_block](iterations, device_arr)
print(f"Completed in {time.time() - start} seconds")
Image.fromarray(device_arr.copy_to_host()).save("mandeltest.png")