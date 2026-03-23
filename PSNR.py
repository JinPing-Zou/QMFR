import numpy as np

def psnr(img1, img2, peakval):
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10((peakval ** 2) / mse)
    return psnr
