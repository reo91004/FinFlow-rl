import torch
def clip_grad_norm_(params, max_norm: float):
    torch.nn.utils.clip_grad_norm_(params, max_norm)
