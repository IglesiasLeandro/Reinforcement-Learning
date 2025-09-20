import torch
import torch.nn as nn
from .config import PATCH_SIZE, N_ACTIONS

class ActorCriticUNet(nn.Module):
    """
    Encoder leve + cabeça de ator (logits por patch) + cabeça de crítico (valor escalar).
    """
    def __init__(self, in_ch=2, base_ch=32, n_actions=N_ACTIONS, patch_size=PATCH_SIZE):
        super().__init__()
        self.patch_size = patch_size
        self.n_actions  = n_actions

        # Encoder (troque por sua U-Net encoder+skips se quiser)
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Ator: logits densos -> agregamos por patch
        self.actor_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, self.n_actions, 1)
        )

        # Crítico: valor escalar
        self.critic_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_ch, base_ch),
            nn.ReLU(inplace=True),
            nn.Linear(base_ch, 1)
        )

    def forward(self, state: torch.Tensor):
        """
        state: [B,2,H,W] = concat(img, mask_t)
        retorna:
          logits_patch: [B, n_patches, n_actions]
          value:        [B, 1]
        """
        feats = self.enc(state)                # [B,C,H,W]
        logits_map = self.actor_head(feats)    # [B,n_actions,H,W]

        B, A, H, W = logits_map.shape
        ps = self.patch_size
        # agregação de logits por patch (média)
        logits_map = logits_map.view(B, A, H//ps, ps, W//ps, ps)
        logits_map = logits_map.permute(0,2,4,1,3,5)          # [B,hb,wb,A,ps,ps]
        logits_patch = logits_map.mean(dim=(4,5)).contiguous()# [B,hb,wb,A]
        logits_patch = logits_patch.view(B, -1, A)            # [B,N,A]

        value = self.critic_head(feats)        # [B,1]
        return logits_patch, value