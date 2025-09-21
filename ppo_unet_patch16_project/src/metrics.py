import torch
def dice_coef(bin_pred: torch.Tensor, bin_gt: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """
    bin_pred/bin_gt: [B,1,H,W] em {0,1}
    retorna: [B] dice por amostra
    """
    inter = (bin_pred * bin_gt).sum(dim=(1,2,3))
    union = bin_pred.sum(dim=(1,2,3)) + bin_gt.sum(dim=(1,2,3))
    return (2*inter + eps) / (union + eps)