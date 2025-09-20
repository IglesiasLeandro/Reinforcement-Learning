
import cv2
import numpy as np
import torch
from .patches import img_to_patches, patches_to_img
from .io_utils import to_np01, from_np01
from .config import PATCH_SIZE

def apply_action_patch(patch_np: np.ndarray, action: int) -> np.ndarray:
    """
    patch_np: np.uint8 [ps,ps] em {0..255}, binário (0/255)
    action: 0=keep, 1=grow, 2=shrink, 3=smooth, 4=clean
    """
    kernel = np.ones((3,3), np.uint8)
    if action == 0:   # KEEP
        return patch_np
    elif action == 1: # GROW
        return cv2.dilate(patch_np, kernel, iterations=1)
    elif action == 2: # SHRINK
        return cv2.erode(patch_np, kernel, iterations=1)
    elif action == 3: # SMOOTH
        return cv2.medianBlur(patch_np, 3)
    elif action == 4: # CLEAN
        num_labels, labels = cv2.connectedComponents((patch_np > 127).astype(np.uint8))
        out = np.zeros_like(patch_np)
        for lbl in range(1, num_labels):
            comp = (labels == lbl).astype(np.uint8)
            if comp.sum() >= 8:
                out = np.maximum(out, comp * 255)
        return out
    return patch_np

def apply_actions_patchwise(mask_t: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """
    Aplica ações por patch em toda a máscara.
    mask_t: [B,1,H,W] float {0,1}
    actions: [B, n_patches] long
    return: M_{t+1} [B,1,H,W] float {0,1}
    """
    device = mask_t.device
    B, _, H, W = mask_t.shape
    ps = PATCH_SIZE

    patches = img_to_patches(mask_t, ps)  # [B,N,ps,ps]
    B, N, _, _ = patches.shape

    out_patches = []
    for b in range(B):
        arrs = []
        for i in range(N):
            patch_np = to_np01(patches[b, i:i+1, ...])  # [ps,ps] em 0..255
            action = int(actions[b, i].item())
            new_patch = apply_action_patch(patch_np, action)  # np.uint8
            arrs.append(from_np01(new_patch, device).squeeze(0))  # [1,ps,ps] -> [ps,ps]
        arrs = torch.stack(arrs, dim=0)  # [N, ps, ps]
        out_patches.append(arrs)
    out_patches = torch.stack(out_patches, dim=0)  # [B,N,ps,ps]
    mask_tp1 = patches_to_img(out_patches, H, W, ps)
    return mask_tp1