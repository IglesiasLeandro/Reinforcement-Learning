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
# Utilidades do ambiente (logprob/entropy modificados)
# ============================================================

def softmax2probs(logits):
    return F.softmax(logits, dim=1)

def choose_pixel_actions(probs):
    B, A, H, W = probs.shape
    p = probs.permute(0, 2, 3, 1).reshape(-1, A)
    dist = Categorical(p)
    a = dist.sample()
    return a.view(B, H, W)

def logprob_actions_from_logits(logits, actions):
    """Retorna log-probs em nível de pixel e a soma (joint)"""
    B, A, H, W = logits.shape
    logp = F.log_softmax(logits, dim=1)
    idx = actions.unsqueeze(1)                   # [B,1,H,W]
    logp_pix = logp.gather(1, idx).squeeze(1)    # [B,H,W] <--- O que precisamos para PA3C
    logp_joint = logp_pix.flatten(1).sum(dim=1)  # [B]
    return logp_pix, logp_joint

def entropy_from_logits(logits):
    """MODIFICADO: Retorna o mapa de entropia em nível de pixel"""
    p = F.softmax(logits, dim=1)
    logp = F.log_softmax(logits, dim=1)
    ent_pix = -(p * logp).sum(dim=1)             # [B,H,W]
    return ent_pix # Retorna o mapa espacial

# ============================================================
# Ações Simplificadas (Inalterado)
# ============================================================

def apply_actions_ternary(mask_t: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """
    mask_t: [B,1,H,W] float
    actions: [B,H,W] long
           0 = SET_BACKGROUND (força 0)
           1 = DO_NOTHING (mantém mask_t)
           2 = SET_FOREGROUND (força 1)
    """
    actions_expanded = actions.unsqueeze(1)  # [B,1,H,W]
    
    mask_new = torch.where(
        actions_expanded == 2,
        torch.ones_like(mask_t),      # SET_FOREGROUND
        torch.where(
            actions_expanded == 0,
            torch.zeros_like(mask_t),  # SET_BACKGROUND
            mask_t                     # DO_NOTHING
        )
    )
    
    return mask_new

def apply_actions_binary(mask_t: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """
    Aplica ações binárias à máscara.
    
    mask_t: [B,1,H,W] float - A máscara da etapa anterior.
    actions: [B,H,W] long - O tensor de ações (0 ou 1).
            0 = SET_BACKGROUND (força 0)
            1 = DO_NOTHING (mantém o valor de mask_t)
    """
    
    # 1. Cria uma máscara "keep_mask"
    # Aonde a ação for 1 (DO_NOTHING), keep_mask será 1.0
    # Aonde a ação for 0 (SET_BACKGROUND), keep_mask será 0.0
    keep_mask = (actions == 1).unsqueeze(1).to(mask_t.dtype)  # Shape: [B,1,H,W]
    
    # 2. Multiplica a máscara original pela keep_mask
    # - Onde a ação era 0, multiplica por 0.0 -> força 0 (background)
    # - Onde a ação era 1, multiplica por 1.0 -> mantém o valor (do_nothing)
    return mask_t * keep_mask

# ============================================================
# Recompensa L2 (Inalterado)
# ============================================================

def reward_l2_map(mask_prev: torch.Tensor, mask_new: torch.Tensor, gt: torch.Tensor,
                  scale: float = 1.0, clip: float | None = None):
    """
    Retorna r(t) por pixel: [B,1,H,W]
    r(t) = ||f(t-1)-G||^2 - ||f(t)-G||^2   (sem somar no espaço)
    """
    # garanta que gt e máscaras estão em [0,1]
    # gt = (gt > 0.5).float()  # se necessário, binarize antes na pipeline
    r_map = (mask_prev - gt)**2 - (mask_new - gt)**2  # [B,1,H,W]
    r_map = r_map * scale
    if clip is not None:
        r_map = r_map.clamp(-clip, clip)
    return r_map

# ============================================================
# Inicialização de máscara (Inalterado)
# ============================================================

def initialize_mask_from_image(img):
    return img[:, :1, ...].clone()

def initialize_mask_otsu(img):
    B, _, H, W = img.shape
    device = img.device
    mask = torch.zeros(B, 1, H, W, device=device)
    for b in range(B):
        img_np = (img[b, 0].detach().cpu().numpy() * 255).astype(np.uint8)
        _, thresh = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        mask[b, 0] = torch.from_numpy(thresh / 255.0).to(device)
    return mask

@torch.no_grad()
def compute_returns_advantages_pa3c(
    rews_map: torch.Tensor,     # [T,B,1,H,W]  r(t) por pixel
    vals_map: torch.Tensor,     # [T+1,B,1,H,W] V(s_t) por pixel (inclui V_{T})
    omega_conv: nn.Conv2d,      # ω (conv 3x3) aprendível
    gamma: float = 0.95
):
    T, B, _, H, W = rews_map.shape
    device = rews_map.device
    
    R = torch.zeros_like(rews_map)   # [T,B,1,H,W]
    A = torch.zeros_like(rews_map)   # [T,B,1,H,W]

    for t in reversed(range(T)):
        # Vizinhança no passo t+1: sum_{j in N(i)} ω_{i-j} V_{t+1}(j)
        V_next_nb = omega_conv(vals_map[t+1])              # [B,1,H,W]
        R_t = rews_map[t] + gamma * V_next_nb              # [B,1,H,W]  (Eq. 8)
        R[t] = R_t
        A[t] = R_t - vals_map[t]                           # [B,1,H,W]  (Eq. 10)

    return R, A  # ambos [T,B,1,H,W]

def a3c_update(
    model,
    optimizer,
    states,        # [N,C,H,W]
    actions,       # [N,H,W]
    targets_R,     # [N,1,H,W]
    advantages,    # [N,1,H,W]
    ent_coef: float = 0.01,
    vf_coef: float = 1.0,
    epochs: int = 1,
    minibatch_size: int = 8
):
    N = states.size(0)
    _, _, H, W = states.shape
    
    # Normalizar vantagens por imagem (opcional, não está no paper)
    mean = advantages.mean(dim=(1,2,3), keepdim=True)
    std = advantages.std(dim=(1,2,3), keepdim=True)
    adv = (advantages - mean) / (std + 1e-8)

    idx = torch.randperm(N, device=states.device)
    for _ in range(epochs):
        for start in range(0, N, minibatch_size):
            mb = idx[start:start+minibatch_size]
            s  = states[mb]
            a  = actions[mb]
            Rm = targets_R[mb]
            Am = adv[mb]

            logits, Vmap = model(s)
            logp_pix = torch.log_softmax(logits, dim=1)
            logp_a   = logp_pix.gather(1, a.unsqueeze(1)).squeeze(1)  # [mb,H,W]

            # === Equação (15): soma espacial, média no batch, normaliza por N ===
            Am_2d = Am.squeeze(1).detach()  # [mb,H,W]
            L_policy = -(logp_a * Am_2d).sum(dim=(1,2)).mean() / (H * W)
            
            # === Equação (13) ===
            L_value = F.mse_loss(Vmap, Rm)
            
            # Entropia (regularização)
            ent = (-(logp_pix.exp() * logp_pix).sum(dim=1)).mean()
            
            loss = L_policy + vf_coef * L_value - ent_coef * ent

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
# ============================================================
# Métricas (Inalterado)
# ============================================================

def segmentation_metrics(pred_mask, gt_mask, threshold=0.5):
    pred = (pred_mask > threshold).float()
    gt   = (gt_mask   > threshold).float()
    pred_flat = pred.view(-1).cpu().numpy()
    gt_flat   = gt.view(-1).cpu().numpy()
    acc  = accuracy_score(gt_flat, pred_flat)
    prec = precision_score(gt_flat, pred_flat, zero_division=0)
    rec  = recall_score(gt_flat, pred_flat, zero_division=0)
    f1   = f1_score(gt_flat, pred_flat, zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def iou(mask, gt):
    inter = (mask * gt).sum(dim=(1, 2, 3))
    union = (mask + gt - (mask * gt)).sum(dim=(1, 2, 3))
    return (inter + 1e-6) / (union + 1e-6)

# ============================================================
# Buffer de rollout (MODIFICADO)
# ============================================================

@dataclass
class Rollout:
    states:   list
    actions:  list
    logp_old: list # MODIFICADO: Irá armazenar mapas [B, H, W]
    values:   list # MODIFICADO: Irá armazenar mapas [B, 1, H, W]
    rewards:  list # Armazena recompensas globais [B]
    dones:    list

    def to_tensors(self, device):
        S  = torch.cat(self.states,   dim=0)  # [T*B, C, H, W]
        A  = torch.cat(self.actions,  dim=0)  # [T*B, H, W]
        LP = torch.cat(self.logp_old, dim=0)  # [T*B, H, W]  <-- MODIFICADO
        V  = torch.cat(self.values,   dim=0)  # [T*B, 1, H, W] <-- MODIFICADO
        R  = torch.cat(self.rewards,  dim=0)  # [T*B]
        D  = torch.cat(self.dones,    dim=0)  # [T*B]
        return S.to(device), A.to(device), LP.to(device), V.to(device), R.to(device), D.to(device)




# ============================================================
# Loop de treino (MODIFICADO PARA PIXEL-LEVEL)
# ============================================================

def train_ppo_segmentation(  # vai virar A3C pixel-wise do paper
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader,
    device="cuda",
    T=6,
    updates=5000,
    gamma=0.95,
    vf_coef=1.0,            # paper usa 1.0 no termo de valor
    ent_coef=0.01,
    epochs=1,               # A3C normalmente 1 passada
    minibatch_size=8,
    init_strategy='image',  # 'image' (paper) ou 'otsu'
    warmup_steps=100,
    log_interval=50
):
    model.to(device).train()

    # === ω: vizinhança aprendível (Eq. 8) ===
    omega_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False).to(device)
    with torch.no_grad():
        omega_conv.weight.zero_()
        omega_conv.weight[:, :, 1, 1] = 1.0  # identidade inicial
    
    # Adiciona omega ao otimizador
    optimizer.add_param_group({
        'params': omega_conv.parameters(), 
        'lr': optimizer.param_groups[0]['lr']
    })
    
    # AGORA pega as LRs base (já inclui model + omega)
    base_lrs = [g['lr'] for g in optimizer.param_groups]
    iou_history, reward_history = [], []
    best_iou = 0.0

    for update in range(1, updates + 1):
        # Warmup de LR
        if update <= warmup_steps:
            lr_scale = update / warmup_steps
            for g, base_lr in zip(optimizer.param_groups, base_lrs):
                g['lr'] = base_lr * lr_scale

        # === COLETA ON-POLICY (T passos) ===
        states_seq   = []  # cada item: [B,2,H,W]
        actions_seq  = []  # [B,H,W]
        values_seq   = []  # [B,1,H,W]
        rewards_seq  = []  # [B,1,H,W]

        # lote do dataloader
        try:
            img, gt = next(train_ppo_segmentation._dl_it)
        except Exception:
            train_ppo_segmentation._dl_it = iter(dataloader)
            img, gt = next(train_ppo_segmentation._dl_it)

        img = img.to(device)
        gt  = gt.to(device)
        if img.size(1) > 1: img = img[:, :1, ...]
        if gt.size(1)  > 1: gt  = gt[:, :1, ...]
        B, _, H, W = img.shape

        # inicialização da máscara
        if init_strategy == 'image':
            mask = initialize_mask_from_image(img)          # [B,1,H,W]
        elif init_strategy == 'otsu':
            mask = initialize_mask_otsu(img)                # [B,1,H,W]
        else:
            mask = img[:, :1, ...].clone()

        vals_track = []  # guardará V_t .. V_T (T+1 entradas)

        for t in range(T):
            state = torch.cat([img, mask], dim=1)           # [B,2,H,W]
            logits, value_map = model(state)                 # value_map: [B,1,H,W]
            vals_track.append(value_map.detach())

            # amostra ação por pixel
            probs   = F.softmax(logits, dim=1)               # [B,2,H,W]
            actions = choose_pixel_actions(probs)             # [B,H,W]

            

            # aplica ação binária
            mask_next = apply_actions_binary(mask, actions)  # [B,1,H,W]

            # recompensa por pixel (mapa) — Eq. (12)
            rew_map = reward_l2_map(mask, mask_next, gt)             # [B,1,H,W]

            # guarda no buffer
            states_seq.append(state.detach())
            actions_seq.append(actions.detach())
            values_seq.append(value_map.detach())
            rewards_seq.append(rew_map.detach())

            # avança estado
            mask = mask_next

        # Bootstrap V_{T}
        with torch.no_grad():
            final_state = torch.cat([img, mask], dim=1)
            _, v_last = model(final_state)                  # [B,1,H,W]
        vals_track.append(v_last.detach())

        # === Empilha tensores do rollout ===
        # rews_map: [T,B,1,H,W] ; vals_map: [T+1,B,1,H,W]
        rews_map = torch.stack(rewards_seq, dim=0)
        vals_map = torch.stack(vals_track,  dim=0)

        # === R_t e A_t com vizinhança ω (Eq. 8,10) ===
        R, A = compute_returns_advantages_pa3c(
            rews_map=rews_map,        # [T,B,1,H,W]
            vals_map=vals_map,        # [T+1,B,1,H,W]
            omega_conv=omega_conv,
            gamma=gamma
        )                              # ambos [T,B,1,H,W]

        # === Achatar (T,B) -> N para atualização ===
        Tsteps, Bsz = R.shape[:2]
        N = Tsteps * Bsz
        S  = torch.cat(states_seq,  dim=0)                   # [N,2,H,W]
        Act = torch.cat(actions_seq, dim=0)                  # [N,H,W]
        targets_R  = R.transpose(0,1).reshape(N, 1, H, W)    # [N,1,H,W]
        advantages = A.transpose(0,1).reshape(N, 1, H, W)    # [N,1,H,W]

        # === A3C update (Eqs. 11 e 9) ===
        a3c_update(
            model, optimizer,
            states=S,
            actions=Act,
            targets_R=targets_R,
            advantages=advantages,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            epochs=epochs,
            minibatch_size=minibatch_size,
        )

        # === LOGGING ===
        with torch.no_grad():
            iou_final = iou(mask, gt).mean().item()
            mean_reward = rews_map[-1].mean().item()
            
            # 🆕 CALCULAR PRECISION, RECALL, F1 durante o treino
            pred_binary = (mask > 0.5).float()
            gt_binary = (gt > 0.5).float()
            
            # Métricas por pixel (flatten tudo)
            pred_flat = pred_binary.view(-1).cpu().numpy()
            gt_flat = gt_binary.view(-1).cpu().numpy()
            
            train_precision = precision_score(gt_flat, pred_flat, zero_division=0)
            train_recall = recall_score(gt_flat, pred_flat, zero_division=0)
            train_f1 = f1_score(gt_flat, pred_flat, zero_division=0)
            
            # Guardar histórico
            iou_history.append(iou_final)
            reward_history.append(mean_reward)
            
            # 🆕 Adicionar novos históricos (declarar antes do loop)
            # precision_history.append(train_precision)
            # recall_history.append(train_recall)
            # f1_history.append(train_f1)
            
            if iou_final > best_iou:
                best_iou = iou_final
                torch.save(model.state_dict(), 'best_pixeldrl_model.pth')
            
            if update % log_interval == 0:
                print(f"[Update {update}/{updates}]")
                print(f"  IoU:       {iou_final:.4f} (best: {best_iou:.4f})")
                print(f"  Precision: {train_precision:.4f}")  # 🆕
                print(f"  Recall:    {train_recall:.4f}")     # 🆕
                print(f"  F1-score:  {train_f1:.4f}")         # 🆕
                print(f"  Reward:    {mean_reward:.4f}")
                
                
        # (Opcional) early stop simples
        if len(iou_history) >= 100 and np.mean(iou_history[-50:]) > 0.90:
            print(f"\n✓ Convergiu! IoU médio (últimas 50): {np.mean(iou_history[-50:]):.4f}")
            break

    # === Avaliação final (igual ao que você já tinha) ===
    print("\n=== Avaliação Final ===")
    try:
        model.load_state_dict(torch.load('best_pixeldrl_model.pth'))
        print("✓ Melhor modelo carregado")
    except Exception:
        print("⚠ Usando modelo do último update")

    model.eval()
    all_preds, all_gts = [], []
    with torch.no_grad():
        eval_loader = dataloader
        for img, gt in eval_loader:
            img, gt = img.to(device), gt.to(device)
            if img.size(1) > 1: img = img[:, :1, ...]
            if gt.size(1)  > 1: gt  = gt[:, :1, ...]
            if init_strategy == 'image': mask = initialize_mask_from_image(img)
            elif init_strategy == 'otsu': mask = initialize_mask_otsu(img)
            else: mask = img[:, :1, ...].clone()

            for _ in range(T):
                state = torch.cat([img, mask], dim=1)
                logits, _ = model(state)
                actions = torch.argmax(logits, dim=1)  # [B,H,W]

                mask = apply_actions_binary(mask, actions)

            all_preds.append(mask)
            all_gts.append(gt)

    preds = torch.cat(all_preds, dim=0)
    gts   = torch.cat(all_gts,   dim=0)

    mean_iou = iou(preds, gts).mean().item()
    metrics  = segmentation_metrics(preds, gts)
    print(f"Mean IoU:  {mean_iou:.4f}")
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

# Não se esqueça de definir train_ppo_segmentation._dl_it = iter(dataloader)
# antes de chamar a função pela primeira vez.
# Exemplo:
# train_ppo_segmentation._dl_it = iter(train_loader) 
# results = train_ppo_segmentation(...)