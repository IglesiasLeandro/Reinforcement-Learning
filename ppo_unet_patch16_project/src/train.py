# ppo_segmentation.py
import math, random, time
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical




# ============================================================
# 2) Utilidades do ambiente
# ============================================================

def softmax2probs(logits):  # [B,A,H,W] -> [B,A,H,W]
    return F.softmax(logits, dim=1)

def choose_pixel_actions(probs):  # [B,A,H,W] -> [B,H,W] (long)
    # amostra categórica por pixel
    B, A, H, W = probs.shape
    # reshape para amostrar por “pixels como batch”
    p = probs.permute(0,2,3,1).reshape(-1, A)  # [(B*H*W), A]
    dist = Categorical(p)
    a = dist.sample()                           # [(B*H*W)]
    return a.view(B, H, W)

def logprob_actions_from_logits(logits, actions):  # logits [B,A,H,W], actions [B,H,W]
    B, A, H, W = logits.shape
    logp = F.log_softmax(logits, dim=1)          # [B,A,H,W]
    # gather por ação escolhida em cada pixel
    idx = actions.unsqueeze(1)                   # [B,1,H,W]
    logp_pix = logp.gather(1, idx).squeeze(1)    # [B,H,W]
    # log prob conjunta da imagem = soma dos logs dos pixels
    logp_joint = logp_pix.flatten(1).sum(dim=1)  # [B]
    return logp_pix, logp_joint

def entropy_from_logits(logits):  # média de entropia por pixel para logging/bonus
    p = F.softmax(logits, dim=1)
    logp = F.log_softmax(logits, dim=1)
    ent_pix = -(p * logp).sum(dim=1)           # [B,H,W]
    ent_img = ent_pix.flatten(1).mean(dim=1)   # [B]
    return ent_img

# --- ambiente: aplica ações pixel-wise à máscara (grow/erode/keep) ---
import numpy as np, cv2
def apply_actions_pixelwise(mask_t: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """
    mask_t: [B,1,H,W] float {0,1}
    actions: [B,H,W] long  (0=KEEP, 1=GROW, 2=SHRINK)
    return: mask_{t+1} [B,1,H,W]
    """
    device = mask_t.device
    B, _, H, W = mask_t.shape
    out = torch.zeros_like(mask_t)
    kernel = np.ones((3,3), np.uint8)

    for b in range(B):
        m = (mask_t[b,0].detach().cpu().numpy()*255).astype(np.uint8)
        a = actions[b].detach().cpu().numpy()

        new_m = m.copy()
        grow_sel   = (a == 1)
        shrink_sel = (a == 2)

        if grow_sel.any():
            dil = cv2.dilate(m, kernel, 1)
            new_m[grow_sel] = dil[grow_sel]

        if shrink_sel.any():
            ero = cv2.erode(m, kernel, 1)
            new_m[shrink_sel] = ero[shrink_sel]

        out[b,0] = torch.from_numpy(new_m/255.0).to(device=device, dtype=torch.float32)
    return out

# IoU util
def iou(mask, gt):  # ambos [B,1,H,W] float {0,1}
    inter = (mask*gt).sum(dim=(1,2,3))
    union = (mask+gt-(mask*gt)).sum(dim=(1,2,3))
    return (inter + 1e-6) / (union + 1e-6)

# recompensa = ΔIoU por imagem (escála para [-1,1] opcional)
def reward_delta_iou(mask_prev, mask_new, gt):
    iou_prev = iou(mask_prev, gt)  # [B]
    iou_new  = iou(mask_new,  gt)  # [B]
    r = (iou_new - iou_prev).clamp(-1.0, 1.0)    # [B]
    return r


# ============================================================
# 3) Buffer de rollout (on-policy)
# ============================================================
@dataclass
class Rollout:
    states:   list
    actions:  list
    logp_old: list
    values:   list
    rewards:  list
    dones:    list

    def to_tensors(self, device):
        S = torch.cat(self.states,   dim=0)    # [T*B, 2, H, W]
        A = torch.cat(self.actions,  dim=0)    # [T*B, H, W]
        LP= torch.cat(self.logp_old, dim=0)    # [T*B]
        V = torch.cat(self.values,   dim=0)    # [T*B, 1]
        R = torch.cat(self.rewards,  dim=0)    # [T*B]
        D = torch.cat(self.dones,    dim=0)    # [T*B]
        return S.to(device), A.to(device), LP.to(device), V.to(device), R.to(device), D.to(device)


# ============================================================
# 4) GAE e retornos
# ============================================================
def compute_gae(rews, vals, dones, gamma=0.99, lam=0.95):
    """
    rews:  [T, B]
    vals:  [T+1, B] ou [T+1, B, 1]
    dones: [T, B]
    """
    # garante formato [T+1, B, 1]
    if vals.ndim == 2:
        vals = vals.unsqueeze(-1)

    T, B = rews.shape
    advs = torch.zeros(T, B, device=rews.device)
    last_adv = torch.zeros(B, 1, device=rews.device)

    for t in reversed(range(T)):
        mask = 1.0 - dones[t].unsqueeze(1)
        delta = rews[t].unsqueeze(1) + gamma * vals[t+1] * mask - vals[t]
        last_adv = delta + gamma * lam * mask * last_adv
        advs[t] = last_adv.squeeze(1)

    rets = advs + vals[:-1].squeeze(2)
    return advs, rets


# ============================================================
# 5) PPO Update
# ============================================================
def ppo_update(
    model, optimizer,
    states, actions, logp_old, values_old, returns, advantages,
    clip_eps=0.2, vf_coef=0.5, ent_coef=0.01,
    epochs=4, minibatch_size=8
):
    N = states.size(0)
    # normalizar vantagens
    advantages = (advantages - advantages.mean())/(advantages.std()+1e-8)

    idx = torch.randperm(N)
    for _ in range(epochs):
        for start in range(0, N, minibatch_size):
            mb = idx[start:start+minibatch_size]

            s = states[mb]            # [m,2,H,W]
            a = actions[mb]           # [m,H,W]
            old_lp = logp_old[mb]     # [m]
            ret = returns[mb]         # [m]
            val_old = values_old[mb]  # [m,1]
            adv = advantages[mb]      # [m]

            logits, value = model(s)  # [m,A,H,W], [m,1]
            # log-prob conjunta atual (soma dos pixels)
            _, logp_joint = logprob_actions_from_logits(logits, a)  # [m]
            ratio = torch.exp(logp_joint - old_lp)                 # [m]

            # policy loss (clipped)
            unclipped = ratio * adv
            clipped   = torch.clamp(ratio, 1.0-clip_eps, 1.0+clip_eps) * adv
            L_clip = torch.mean(torch.min(unclipped, clipped))

            # value loss (MSE) — opcionalmente clipar valor
            L_v = F.mse_loss(value.squeeze(1), ret)

            # entropy (média por imagem)
            ent = entropy_from_logits(logits).mean()

            loss = -L_clip + vf_coef*L_v - ent_coef*ent

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()


# ============================================================
# 6) Loop de treino (coleta on-policy + update)
# ============================================================
def train_ppo_segmentation(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader,             # deve fornecer (img [B,1,H,W], gt [B,1,H,W])
    device="cuda",
    T=4,                    # steps por rollout (refinamentos por imagem)
    updates=1000,
    gamma=0.99, lam=0.95,
    clip_eps=0.2, vf_coef=0.5, ent_coef=0.01,
    epochs=4, minibatch_size=8
):
    model.to(device).train()

    for update in range(1, updates+1):
        # === COLETA ON-POLICY ===
        buffers = Rollout(states=[], actions=[], logp_old=[], values=[], rewards=[], dones=[])
        # pega um batch do dataset (on-policy)
        try:
            img, gt = next(train_ppo_segmentation._dl_it)
        except:
            train_ppo_segmentation._dl_it = iter(dataloader)
            img, gt = next(train_ppo_segmentation._dl_it)

        img = img.to(device)                   # [B,1,H,W]  (normalize fora)
        gt  = gt.to(device)                    # [B,1,H,W]
        B, _, H, W = img.shape

        # inicializa máscara (vazia ou heurística)
        mask = torch.zeros_like(gt)            # [B,1,H,W]

        # para GAE: guardamos valores com bootstrap (T+1)
        vals_track = []

        # rollout de T passos de refinamento
        for t in range(T):
            state = torch.cat([img, mask], dim=1)  # [B,2,H,W]
            logits, value = model(state)           # [B,A,H,W], [B,1]
            vals_track.append(value.detach())      # para GAE

            probs = softmax2probs(logits)
            actions = choose_pixel_actions(probs)  # [B,H,W] (long)

            # logp conjunta atual (para "old")
            _, logp_joint = logprob_actions_from_logits(logits.detach(), actions)

            # aplica ações no ambiente
            actions_up = F.interpolate(actions.unsqueeze(1).float(), size=mask.shape[-2:], mode="nearest").long().squeeze(1)
            mask_next = apply_actions_pixelwise(mask, actions_up)

            # recompensa ΔIoU (escala por imagem)
            rew = reward_delta_iou(mask, mask_next, gt)  # [B]

            # done? (critério simples: IoU alto ou passo final)
            done = ((iou(mask_next, gt) > 0.95) | torch.tensor([t == T - 1], device=device, dtype=torch.bool)).float() # [B]

            # guarda no buffer
            buffers.states.append(state.detach())
            buffers.actions.append(actions.detach())
            buffers.logp_old.append(logp_joint.detach().view(B))
            buffers.values.append(value.detach())
            buffers.rewards.append(rew.detach())
            buffers.dones.append(done.detach())

            # avança
            mask = mask_next

        # bootstrap do valor no último estado
        with torch.no_grad():
            final_state = torch.cat([img, mask], dim=1)
            _, v_last = model(final_state)   # [B,1]
        vals_track.append(v_last.detach())   # agora temos [T+1,B,1]

        # === PREPARA GAE ===
        # empilha para [T,B], [T+1,B]
        rews  = torch.stack(buffers.rewards, dim=0)          # [T,B]
        dones = torch.stack(buffers.dones,   dim=0)          # [T,B]
        vals  = torch.stack(vals_track,      dim=0).squeeze(2)  # [T+1,B,1]->[T+1,B]

        advs, rets = compute_gae(rews, vals, dones, gamma=gamma, lam=lam)  # [T,B],[T,B]

        # === ACHATA e ATUALIZA PPO ===
        S, A, LP, V, R, D = buffers.to_tensors(device)  # [T*B,...]
        advs_tr  = advs.transpose(0,1).reshape(-1)      # [T*B]
        rets_tr  = rets.transpose(0,1).reshape(-1)      # [T*B]

        ppo_update(
            model, optimizer,
            states=S, actions=A, logp_old=LP, values_old=V,
            returns=rets_tr, advantages=advs_tr,
            clip_eps=clip_eps, vf_coef=vf_coef, ent_coef=ent_coef,
            epochs=epochs, minibatch_size=minibatch_size
        )

        if update % 10 == 0:
            with torch.no_grad():
                iou_final = iou(mask, gt).mean().item()
            print(f"[upd {update}] mean IoU(final)={iou_final:.4f}")
