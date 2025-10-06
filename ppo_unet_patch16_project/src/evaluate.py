import torch
import torch.nn.functional as F
from train import apply_actions_pixelwise

@torch.no_grad()
def evaluate_segmentation(model, dataloader, device="cuda"):
    """
    Avalia o modelo em termos de IoU, Dice e Accuracy pixel-wise.
    O modelo PPO segmenta iterativamente refinando a máscara.
    """
    model.eval()
    total_iou, total_dice, total_acc = 0.0, 0.0, 0.0
    n = 0

    for img, gt in dataloader:
        img, gt = img.to(device), gt.to(device)
        mask = torch.zeros_like(gt)  # máscara inicial (vazia)

        # --- refinamento iterativo (usando a política aprendida) ---
        for _ in range(4):  # número de passos PPO (mesmo T usado no treino)
            state = torch.cat([img, mask], dim=1)
            logits, _ = model(state)
            probs = F.softmax(logits, dim=1)
            actions = torch.argmax(probs, dim=1)
            mask = apply_actions_pixelwise(mask, actions)  # função do PPO

        # --- calcula métricas ---
        pred = (mask > 0.5).float()

        inter = (pred * gt).sum(dim=(1, 2, 3))
        union = (pred + gt - pred * gt).sum(dim=(1, 2, 3))
        iou = (inter + 1e-6) / (union + 1e-6)

        dice = (2 * inter + 1e-6) / (pred.sum(dim=(1,2,3)) + gt.sum(dim=(1,2,3)) + 1e-6)
        acc  = (pred == gt).float().mean(dim=(1,2,3))

        total_iou += iou.sum().item()
        total_dice += dice.sum().item()
        total_acc  += acc.sum().item()
        n += img.size(0)

    model.train()
    return total_iou/n, total_dice/n, total_acc/n
