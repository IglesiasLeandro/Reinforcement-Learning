import torch
from .model import ActorCriticUNet
from .actions import apply_actions_patchwise
import torch.nn as nn

@torch.no_grad()
def env_step(model: nn.Module,
             imgs: torch.Tensor, mask_t: torch.Tensor, gts: torch.Tensor):
    """
    Um passo do ambiente para TODO o batch.
    returns: mask_tp1, actions, logp_sum_per_sample, value
    """
    state = torch.cat([imgs, mask_t], dim=1)          # [B,2,H,W]
    logits_patch, value = model(state)                # [B,N,A], [B,1]
    B, NP, A = logits_patch.shape

    dist = torch.distributions.Categorical(logits=logits_patch.view(-1, A))
    actions = dist.sample().view(B, NP)               # [B, N_patches]
    logp = dist.log_prob(actions.view(-1)).view(B, NP).sum(dim=1)  # [B]

    mask_tp1 = apply_actions_patchwise(mask_t, actions)

    return mask_tp1, actions, logp, value