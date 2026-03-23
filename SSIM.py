
import numpy as np
from scipy.ndimage import gaussian_filter


def ssim(img1, img2, max_val=1, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    mu1 = gaussian_filter(img1, sigma=filter_sigma, mode='reflect')
    mu2 = gaussian_filter(img2, sigma=filter_sigma, mode='reflect')

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(img1 ** 2, sigma=filter_sigma, mode='reflect') - mu1_sq
    sigma2_sq = gaussian_filter(img2 ** 2, sigma=filter_sigma, mode='reflect') - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sigma=filter_sigma, mode='reflect') - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()