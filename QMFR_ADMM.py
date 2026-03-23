
import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
import time 

from TV_Projection import tv_projection
from Qutils import reshape_UVW, hamilton_product, relative_error_omega, load_masked_image
from Model import UVDecomposition
from PSNR import psnr
from SSIM import ssim

dtype = torch.cuda.FloatTensor
torch.manual_seed(5)
np.random.seed(5)

w_decay = 1
lr_real = 0.0001
max_iter = 20001
down = [1.1,1.1,1]
omega = 0.6

image, mask_image, mask = load_masked_image('girl.bmp', mask_rate=0.9)

[n_1,n_2,n_3] = image.shape

hidden = int(n_2)
r_1 = int(n_1/down[0]) 
r_2 = int(n_2/down[1])

mask = torch.ones(image.shape).type(dtype)
mask[mask_image == 0] = 0 
mask_image[mask == 0] = 0

U_input = torch.from_numpy(np.array(range(1, n_1*3 + 1))).reshape(n_1*3, 1).type(torch.cuda.FloatTensor)
V_input = torch.from_numpy(np.array(range(1, n_2*3 + 1))).reshape(n_2*3, 1).type(torch.cuda.FloatTensor)

model = UVDecomposition(hidden, r_1, r_2).type(torch.cuda.FloatTensor)
optimizier = optim.Adam(model.parameters(), lr=lr_real, weight_decay=w_decay)

rho1 = rho2 = 1
tau = 0.01
lambda_UV = 0.005
epsilon = 0.025

Y1 = torch.zeros_like(image)
Y2 = torch.zeros_like(image)
Z = image.clone()  
X_prev = Z

start_time = time.time() 

for iter in range(max_iter):
    iter_start_time = time.time()
    optimizier.zero_grad()
    U = model.U_net(U_input)  
    U = reshape_UVW(U)   
    
    V = model.V_net(V_input)     
    V = reshape_UVW(V)
    
    X_Out = hamilton_product(U, V)[:, :, 1:]
    
    R1 = X_Out - Y1 / rho1
    A = R1
    B = Z - Y2 / rho2

    C = mask * ((image + rho1 * A + rho2 * B)/(1 + rho1 + rho2))
    D = (1 - mask) * ((rho1 * A + rho2 * B)/(rho1 + rho2))
    X = C + D
    
    W = X + Y2 / rho2
    Z = tv_projection(W.detach(), tau / rho2, iter_max=50)

    loss_UV = (lambda_UV * (torch.norm(U)**2 + torch.norm(V)**2) +
               rho1 * torch.norm(X - X_Out + Y1 / rho1)**2)
    loss_UV.backward(retain_graph=True)
    optimizier.step()
    
    Y1 = Y1 + rho1 * (X.detach() - X_Out.detach())
    Y2 = Y2 + rho2 * (X.detach() - Z.detach())
    
    image_recovery = X_Out*(1-mask) + mask_image*mask
    
    if iter % 10 == 0:
        if iter > 0:
            relerr = relative_error_omega(X_Out, X_prev, mask)
            print(f"Iteration {iter}, RelErr_Ω: {relerr.item():.6f}")
            if relerr < epsilon:
                print(f"Early stopping at epoch {iter} due to RelErr_Ω < {epsilon}")
                break
        X_prev = X_Out

    if iter % 10 == 0:
        ps = peak_signal_noise_ratio(image.cpu().detach().numpy(),
                                     np.clip(image_recovery.cpu().detach().numpy(), 0, 1))
        print('iteration:', iter, 'PSNR', ps)
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.imshow(image.cpu().detach().numpy())
        plt.title('gt')
        
        plt.subplot(132)
        plt.imshow(mask_image.cpu().detach().numpy())
        plt.title('mask')
        
        plt.subplot(133)
        plt.imshow(np.clip(image_recovery.cpu().detach().numpy(), 0, 1))
        plt.title('out')
        
        plt.show()

total_time = time.time() - start_time
print(f"\nTotal time for {max_iter} iterations: {total_time:.2f} seconds.")

psnr_value = psnr(image.cpu().detach().numpy(), 
                  np.clip(image_recovery.cpu().detach().numpy(), 0, 1), 1)
print("PSNR: ", psnr_value)

ssim_value = ssim(image.cpu().detach().numpy(), 
                  np.clip(image_recovery.cpu().detach().numpy(), 0, 1))
print("SSIM: ", ssim_value)



