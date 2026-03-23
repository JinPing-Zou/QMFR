
import torch

def tv_projection(x, weight, iter_max=50):
    original_shape = x.shape

    if x.dim() == 3 and x.shape[0] in [1, 3]:  
        x = x.unsqueeze(0)  
    elif x.dim() == 3 and x.shape[2] in [1, 3]:  
        x = x.permute(2, 0, 1).unsqueeze(0)  
    elif x.dim() == 4 and x.shape[1] in [1, 3]:  
        pass
    else:
        raise ValueError(f"Unsupported input shape: {original_shape}")
    
    B, C, H, W = x.shape
    device = x.device
    p = torch.zeros((B, 2, C, H, W), device=device)

    # tau = 0.125
    tau = 0.4

    for _ in range(iter_max):
        div_p = torch.zeros_like(x)
        p1 = p[:, 0]   #x
        p2 = p[:, 1]   #y

        div_p[:, :, :-1, :] += p1[:, :, :-1, :]
        div_p[:, :, 1:, :]  -= p1[:, :, :-1, :]
        div_p[:, :, :, :-1] += p2[:, :, :, :-1]
        div_p[:, :, :, 1:]  -= p2[:, :, :, :-1]

        u = x - weight * div_p

        grad_u_x = torch.zeros_like(x)
        grad_u_y = torch.zeros_like(x)

        grad_u_x[:, :, :, :-1] = u[:, :, :, :-1] - u[:, :, :, 1:]
        grad_u_y[:, :, :-1, :] = u[:, :, :-1, :] - u[:, :, 1:, :]

        norm = torch.sqrt(grad_u_x**2 + grad_u_y**2 + 1e-6)
        # norm = torch.sqrt(grad_u_x**2 + grad_u_y**2 + 1e-6).sum(dim=1, keepdim=True).sqrt()
        
        p[:, 0] = (p[:, 0] + tau * grad_u_y) / (1 + tau * norm)
        p[:, 1] = (p[:, 1] + tau * grad_u_x) / (1 + tau * norm)

    div_p = torch.zeros_like(x)
    p1 = p[:, 0]
    p2 = p[:, 1]
    div_p[:, :, :-1, :] += p1[:, :, :-1, :]
    div_p[:, :, 1:, :]  -= p1[:, :, :-1, :]
    div_p[:, :, :, :-1] += p2[:, :, :, :-1]
    div_p[:, :, :, 1:]  -= p2[:, :, :, :-1]

    out = x - weight * div_p

    if original_shape == out.shape[1:]:  
        return out[0]
    elif len(original_shape) == 3 and original_shape[2] in [1, 3]:   
        return out[0].permute(1, 2, 0).contiguous()
    else: 
        return out
