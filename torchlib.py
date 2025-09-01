import torch

def default_device(cuda_preferred=True):
    if not cuda_preferred:
        return torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)
    return device

