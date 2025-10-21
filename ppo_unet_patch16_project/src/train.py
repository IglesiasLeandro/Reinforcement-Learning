import math, random, time
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import cv2



# ============================================================
# Utilidades do ambiente
# ============================================================

def softmax2probs(logits):
    return F.softmax(logits, dim=1)

def choose_pixel_actions(probs):
    B, A, H, W = probs.shape
    p = probs.permute(0,2,3,1).reshape(-1, A)
    dist = Categorical(p)
    a = dist.sample()
    return a.view(B, H, W)

def logprob_actions_from_logits(logits, actions):
    B, A, H, W = logits.shape
    logp = F.log_softmax(logits, dim=1)
    idx = actions.unsqueeze(1)
    logp_pix = logp.gather(1, idx).squeeze(1)
    logp_joint = logp_pix.flatten(1).sum(dim=1)
    return logp_pix, logp_joint

def entropy_from_logits(logits):
    p = F.softmax(logits, dim=1)
    logp = F.log_softmax(logits, dim=1)
    ent_pix = -(p * logp).sum(dim=1)
    ent_img = ent_pix.flatten(1).mean(dim=1)
    return ent_img


# ============================================================
# MODIFICADO: Ações Simplificadas (Paper: apenas 2 ações)
# ============================================================
def apply_actions_binary(mask_t: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """
    Paper approach: Apenas 2 ações
    mask_t: [B,1,H,W] float - estado atual da máscara
    actions: [B,H,W] long
        0 = SET_ZERO (marca como background)
        1 = DO_NOTHING (mantém valor atual - objeto)
    
    DIFERENÇA CHAVE: Não usa erosão/dilatação, apenas define pixels!
    """
    B, _, H, W = mask_t.shape
    mask_new = mask_t.clone()
    
    # Ação 0: Seta pixel para 0 (background)
    # Ação 1: Mantém pixel como está (objeto)
    for b in range(B):
        mask_new[b, 0][actions[b] == 0] = 0.0
        # actions[b] == 1 não faz nada (mantém valor)
    
    return mask_new


# ============================================================
# MODIFICADO: Recompensa baseada em L2 Distance (Paper Eq. 12)
# ============================================================
def reward_l2_distance(mask_prev, mask_new, gt):
    """
    Paper Equation 12:
    r^(t) = ||f^(t-1) - G||² - ||f^(t) - G||²
    
    Retorna recompensa POSITIVA se a distância diminuiu
    """
    # Distância L2 ao quadrado
    dist_prev = ((mask_prev - gt) ** 2).sum(dim=(1,2,3))
    dist_new  = ((mask_new - gt) ** 2).sum(dim=(1,2,3))
    
    # Recompensa = redução da distância (normalizado)
    reward = (dist_prev - dist_new) / 1000.0
    
    return reward.clamp(-1.0, 1.0)


# ============================================================
# MODIFICADO: Inicialização com a Imagem (Paper Sec. 3.4)
# ============================================================
def initialize_mask_from_image(img):
    """
    Paper approach: s^(0)_i = I_i
    Inicializa máscara com a própria imagem!
    O agente aprende a "apagar" pixels de background.
    """
    # Normaliza imagem para [0,1] se necessário
    mask = img.clone()
    
    # Opcional: pode aplicar threshold simples como hint inicial
    # mask = (mask > mask.mean(dim=(2,3), keepdim=True)).float()
    
    return mask


def initialize_mask_otsu(img):
    """Alternativa: Inicialização com Otsu como no código original"""
    B, _, H, W = img.shape
    device = img.device
    mask = torch.zeros(B, 1, H, W, device=device)
    
    for b in range(B):
        img_np = (img[b,0].detach().cpu().numpy() * 255).astype(np.uint8)
        _, thresh = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morfologia para limpar
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        mask[b,0] = torch.from_numpy(thresh / 255.0).to(device)
    
    return mask


# ============================================================
# Métricas
# ============================================================
def segmentation_metrics(pred_mask, gt_mask, threshold=0.5):
    pred = (pred_mask > threshold).float()
    gt   = (gt_mask > threshold).float()
    
    pred_flat = pred.view(-1).cpu().numpy()
    gt_flat   = gt.view(-1).cpu().numpy()

    acc  = accuracy_score(gt_flat, pred_flat)
    prec = precision_score(gt_flat, pred_flat, zero_division=0)
    rec  = recall_score(gt_flat, pred_flat, zero_division=0)
    f1   = f1_score(gt_flat, pred_flat, zero_division=0)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

def iou(mask, gt):
    inter = (mask*gt).sum(dim=(1,2,3))
    union = (mask+gt-(mask*gt)).sum(dim=(1,2,3))
    return (inter + 1e-6) / (union + 1e-6)


# ============================================================
# Buffer de rollout
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
        S = torch.cat(self.states,   dim=0)
        A = torch.cat(self.actions,  dim=0)
        LP= torch.cat(self.logp_old, dim=0)
        V = torch.cat(self.values,   dim=0)
        R = torch.cat(self.rewards,  dim=0)
        D = torch.cat(self.dones,    dim=0)
        return S.to(device), A.to(device), LP.to(device), V.to(device), R.to(device), D.to(device)


# ============================================================
# GAE
# ============================================================
def compute_gae(rews, vals, dones, gamma=0.99, lam=0.95):
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
# PPO Update
# ============================================================
def ppo_update(
    model, optimizer,
    states, actions, logp_old, values_old, returns, advantages,
    clip_eps=0.2, vf_coef=0.5, ent_coef=0.01,
    epochs=4, minibatch_size=8
):
    N = states.size(0)
    advantages = (advantages - advantages.mean())/(advantages.std()+1e-8)

    idx = torch.randperm(N)
    for _ in range(epochs):
        for start in range(0, N, minibatch_size):
            mb = idx[start:start+minibatch_size]

            s = states[mb]
            a = actions[mb]
            old_lp = logp_old[mb]
            ret = returns[mb]
            val_old = values_old[mb]
            adv = advantages[mb]

            logits, value = model(s)
            _, logp_joint = logprob_actions_from_logits(logits, a)
            ratio = torch.exp(logp_joint - old_lp)

            unclipped = ratio * adv
            clipped   = torch.clamp(ratio, 1.0-clip_eps, 1.0+clip_eps) * adv
            L_clip = torch.mean(torch.min(unclipped, clipped))

            L_v = F.mse_loss(value.squeeze(1), ret)
            ent = entropy_from_logits(logits).mean()

            loss = -L_clip + vf_coef*L_v - ent_coef*ent

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()


# ============================================================
# Loop de treino - MODIFICADO para Paper PixelDRL-MG
# ============================================================
def train_ppo_segmentation(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader,
    device="cuda",
    T=6,  # Paper usa t_max=10, você pode começar com 6
    updates=5000,
    gamma=0.95,  # Paper usa 0.95
    lam=0.95,
    clip_eps=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    epochs=4,
    minibatch_size=8,
    init_strategy='image',  # 'image' (paper) ou 'otsu'
    use_l2_reward=True,     # True = L2 distance (paper), False = IoU
    warmup_steps=100,
    log_interval=50
):
    model.to(device).train()
    
    # Métricas de tracking
    iou_history = []
    reward_history = []
    best_iou = 0.0

    for update in range(1, updates+1):
        # Learning rate warmup
        if update <= warmup_steps:
            lr_scale = update / warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * lr_scale

        # === COLETA ON-POLICY ===
        buffers = Rollout(states=[], actions=[], logp_old=[], values=[], rewards=[], dones=[])
        
        try:
            img, gt = next(train_ppo_segmentation._dl_it)
        except:
            train_ppo_segmentation._dl_it = iter(dataloader)
            img, gt = next(train_ppo_segmentation._dl_it)

        img = img.to(device)
        gt  = gt.to(device)
        B, _, H, W = img.shape

        # MODIFICAÇÃO CHAVE: Inicializa máscara com a imagem (Paper Sec. 3.4)
        if init_strategy == 'image':
            mask = initialize_mask_from_image(img)
        elif init_strategy == 'otsu':
            mask = initialize_mask_otsu(img)
        else:
            mask = img.clone()  # Default: usa imagem

        vals_track = []

        # Rollout de T passos
        for t in range(T):
            # State = concatena imagem + máscara atual
            state = torch.cat([img, mask], dim=1)  # [B, 2, H, W]
            
            logits, value = model(state)
            vals_track.append(value.detach())

            probs = softmax2probs(logits)
            actions = choose_pixel_actions(probs)  # [B, H, W]

            _, logp_joint = logprob_actions_from_logits(logits.detach(), actions)

            # MODIFICAÇÃO: Aplica ações binárias (sem erosão/dilatação)
            if actions.shape[-2:] != mask.shape[-2:]:
                actions_resized = F.interpolate(
                    actions.unsqueeze(1).float(), 
                    size=mask.shape[-2:], 
                    mode="nearest"
                ).long().squeeze(1)
            else:
                actions_resized = actions

            mask_next = apply_actions_binary(mask, actions_resized)

            # MODIFICAÇÃO: Recompensa L2 distance (Paper Eq. 12)
            if use_l2_reward:
                rew = reward_l2_distance(mask, mask_next, gt)
                iou_current = iou(mask_next, gt)
            else:
                # Fallback: recompensa IoU original
                iou_prev = iou(mask, gt)
                iou_current = iou(mask_next, gt)
                rew = ((iou_current - iou_prev) * 10.0).clamp(-1.0, 1.0)

            # Done condition
            is_final_step = torch.full((B,), t == T-1, device=device, dtype=torch.bool)
            done = ((iou_current > 0.95) | is_final_step).float()

            buffers.states.append(state.detach())
            buffers.actions.append(actions.detach())
            buffers.logp_old.append(logp_joint.detach().view(B))
            buffers.values.append(value.detach())
            buffers.rewards.append(rew.detach())
            buffers.dones.append(done.detach())

            mask = mask_next

        # Bootstrap
        with torch.no_grad():
            final_state = torch.cat([img, mask], dim=1)
            _, v_last = model(final_state)
        vals_track.append(v_last.detach())

        # === GAE ===
        rews  = torch.stack(buffers.rewards, dim=0)
        dones = torch.stack(buffers.dones,   dim=0)
        vals  = torch.stack(vals_track,      dim=0).squeeze(2)

        advs, rets = compute_gae(rews, vals, dones, gamma=gamma, lam=lam)

        # === PPO UPDATE ===
        S, A, LP, V, R, D = buffers.to_tensors(device)
        advs_tr  = advs.transpose(0,1).reshape(-1)
        rets_tr  = rets.transpose(0,1).reshape(-1)

        ppo_update(
            model, optimizer,
            states=S, actions=A, logp_old=LP, values_old=V,
            returns=rets_tr, advantages=advs_tr,
            clip_eps=clip_eps, vf_coef=vf_coef, ent_coef=ent_coef,
            epochs=epochs, minibatch_size=minibatch_size
        )

        # === LOGGING ===
        if update % log_interval == 0:
            with torch.no_grad():
                iou_final = iou(mask, gt).mean().item()
                mean_reward = rew.mean().item()
                
                iou_history.append(iou_final)
                reward_history.append(mean_reward)
                
                if iou_final > best_iou:
                    best_iou = iou_final
                    torch.save(model.state_dict(), 'best_pixeldrl_model.pth')
                
                print(f"[Update {update}/{updates}]")
                print(f"  IoU: {iou_final:.4f} (best: {best_iou:.4f})")
                print(f"  Reward: {mean_reward:.4f}")
                if len(iou_history) >= 10:
                    print(f"  Avg IoU (last 10): {np.mean(iou_history[-10:]):.4f}")
        
        # Early stopping
        if len(iou_history) >= 100 and np.mean(iou_history[-50:]) > 0.90:
            print(f"\n✓ Convergiu! IoU médio: {np.mean(iou_history[-50:]):.4f}")
            break

    # === AVALIAÇÃO FINAL ===
    print("\n=== Avaliação Final ===")
    
    try:
        model.load_state_dict(torch.load('best_pixeldrl_model.pth'))
        print("✓ Melhor modelo carregado")
    except:
        print("⚠ Usando modelo do último update")
    
    model.eval()
    all_preds, all_gts = [], []

    with torch.no_grad():
        for img, gt in dataloader:
            img, gt = img.to(device), gt.to(device)
            
            # Mesma estratégia de inicialização
            if init_strategy == 'image':
                mask = initialize_mask_from_image(img)
            elif init_strategy == 'otsu':
                mask = initialize_mask_otsu(img)
            else:
                mask = img.clone()

            # Refinamento iterativo
            for _ in range(T):
                state = torch.cat([img, mask], dim=1)
                logits, _ = model(state)
                probs = F.softmax(logits, dim=1)
                
                actions = choose_pixel_actions(probs)
                
                if actions.shape[-2:] != mask.shape[-2:]:
                    actions_resized = F.interpolate(
                        actions.unsqueeze(1).float(), 
                        size=mask.shape[-2:], 
                        mode="nearest"
                    ).long().squeeze(1)
                else:
                    actions_resized = actions
                
                mask = apply_actions_binary(mask, actions_resized)

            all_preds.append(mask)
            all_gts.append(gt)

    preds = torch.cat(all_preds, dim=0)
    gts   = torch.cat(all_gts, dim=0)

    mean_iou = iou(preds, gts).mean().item()
    metrics = segmentation_metrics(preds, gts)

    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")
    
    return {
        'final_metrics': metrics,
        'mean_iou': mean_iou,
        'iou_history': iou_history,
        'reward_history': reward_history,
        'best_iou': best_iou
    }