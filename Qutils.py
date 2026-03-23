from PIL import Image
import torch
import numpy as np

def load_masked_image(image_path, mask_rate=0.9, size=(512, 512), dtype=torch.float32, device='cuda'):
    image = Image.open(image_path).resize(size)
    image = np.array(image) / 255.0  # (H, W, 3)

    mask_indices = np.random.choice([0, 1], size=image.shape[:2], p=[mask_rate, 1-mask_rate])
    mask = np.expand_dims(mask_indices, axis=-1)  # (H, W, 1)
    masked_image = mask * image  # (H, W, 3)

    image = torch.from_numpy(image).type(dtype).to(device)        # (H, W, 3)
    mask_image = torch.from_numpy(masked_image).type(dtype).to(device)  # (H, W, 3)

    return image, mask_image, mask



def reshape_UVW(tensor):
    if tensor.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got shape {tensor.shape}")

    n1, n2 = tensor.shape
    if n1 % 3 != 0:
        raise ValueError(f"n1={n1} is not divisible by 3")

    return tensor.reshape(n1 // 3, n2, 3)

def quaternion_multiply(r1_t, i1_t, j1_t, k1_t, r2, i2, j2, k2):
    r = r2 @ r1_t - i2 @ i1_t - j2 @ j1_t - k2 @ k1_t
    i = i2 @ r1_t + r2 @ i1_t + k2 @ j1_t - j2 @ k1_t
    j = j2 @ r1_t - k2 @ i1_t + r2 @ j1_t + i2 @ k1_t
    k = k2 @ r1_t + j2 @ i1_t - i2 @ j1_t + r2 @ k1_t
    return r, i, j, k


def hamilton_product(X, Y):
    zeros_X = torch.zeros((X.shape[0], X.shape[1], 1), dtype=X.dtype, device=X.device)
    X_padded = torch.cat([zeros_X, X], dim=2) 
    
    zeros_Y = torch.zeros((Y.shape[0], Y.shape[1], 1), dtype=Y.dtype, device=Y.device)
    Y_padded = torch.cat([zeros_Y, Y], dim=2) 

    r1, i1, j1, k1 = X_padded.unbind(dim=2)  
    r2, i2, j2, k2 = Y_padded.unbind(dim=2) 

    r1_t, i1_t, j1_t, k1_t = r1.t(), i1.t(), j1.t(), k1.t()
    
    r, i, j, k = quaternion_multiply(r1_t, i1_t, j1_t, k1_t, r2, i2, j2, k2)
    output = torch.stack((r, i, j, k), dim=-1)
    return output

def relative_error_omega(T_current, T_previous, mask):
    numerator = torch.norm(T_current * mask - T_previous * mask, p=2)
    denominator = torch.norm(T_previous * mask, p=2)
    
    if denominator > 0:
        return numerator / denominator
    else:
        return torch.full_like(numerator, float('inf'))
    
    
    