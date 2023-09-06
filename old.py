import numpy as np; from numba import cuda; from PIL import Image; import time
Iterations = 10000
Height, Width = 1000,2000
XZoom,YZoom,Xb,Yb = 4, 2.2, 2.5, 1.1

@cuda.jit(device=True)
def calculation(a,b):
    zi = 0
    zr = 0
    for i in range(Iterations):                            
        zi, zr = 2 * zr * zi + b, zr**2 - zi**2 + a        
        if (zr**2 + zi**2) > 4:
            if i < 256: return i
            else: return 255
    return 0

@cuda.jit()
def main(g_array):
    startX, startY = cuda.grid(2)
    gridX, gridY = cuda.gridsize(2)
    for h in range(startY, Height, gridY):
        for w in range(startX, Width, gridX):
            a,b = w * XZoom / Width  - Xb, h * YZoom / Height - Yb
            g_array[h,w,0] = calculation(a,b)
 

array = np.zeros((Height,Width,3), dtype=np.uint8)
g_array = cuda.to_device(array)
start_time = time.time()
main[(32,16),(32,8)](g_array)
o_array = g_array.copy_to_host()
im = Image.fromarray(o_array, mode="RGB")
im.save("Mandelbrot_%s.png" % round(time.time()))
print("--- %s seconds ---" % (time.time() - start_time))