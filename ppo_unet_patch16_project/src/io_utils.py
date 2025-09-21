import numpy as np
import torch
import random

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_np01(t: torch.Tensor) -> np.ndarray:
    """
    Converte tensor Torch para numpy [0,1].
    Espera tensor float [B,1,H,W] ou [1,H,W].
    """
    arr = t.detach().cpu().numpy()
    if arr.ndim == 4:   # [B,1,H,W] -> pega primeiro canal
        arr = arr[:,0]
    elif arr.ndim == 3: # [1,H,W] -> [H,W]
        arr = arr[0]
    arr = np.clip(arr, 0.0, 1.0)
    return arr.astype(np.float32)

def from_np01(arr: np.ndarray, device="cpu") -> torch.Tensor:
    """
    Converte numpy [0,1] para tensor Torch [1,H,W].
    """
    if arr.ndim == 2:   # [H,W]
        arr = np.expand_dims(arr, axis=0)  # [1,H,W]
    t = torch.from_numpy(arr).float().to(device)
    return t
