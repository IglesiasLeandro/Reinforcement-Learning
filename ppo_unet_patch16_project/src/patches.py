import torch
from .config import PATCH_SIZE

def img_to_patches(img: torch.Tensor, ps=PATCH_SIZE) -> torch.Tensor:
    """[B,1,H,W] -> [B, n_patches, ps, ps]"""
    B, C, H, W = img.shape
    assert C == 1, "Esperado canal único para máscara."
    assert H % ps == 0 and W % ps == 0, "H e W devem ser múltiplos do patch."
    patches = img.unfold(2, ps, ps).unfold(3, ps, ps)
    # [B,1,H//ps, W//ps, ps, ps]
    patches = patches.contiguous().view(B, -1, ps, ps)
    return patches

def patches_to_img(patches: torch.Tensor, H: int, W: int, ps=PATCH_SIZE) -> torch.Tensor:
    """[B, n_patches, ps, ps] -> [B,1,H,W]"""
    B, N, _, _ = patches.shape
    h_blocks = H // ps
    w_blocks = W // ps
    patches = patches.view(B, h_blocks, w_blocks, ps, ps)
    patches = patches.permute(0, 1, 3, 2, 4).contiguous()  # [B, hb, ps, wb, ps]
    img = patches.view(B, 1, H, W)
    return img