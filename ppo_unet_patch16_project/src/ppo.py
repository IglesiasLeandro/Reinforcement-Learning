import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import PPOConfig

class RolloutBuffer:
    """
    Armazena transições por passo (para todo o batch) e achata em (T*B)
    """
    def __init__(self):
        self.obs      = []
        self.actions  = []
        self.logps    = []
        self.values   = []
        self.rewards  = []
        self.dones    = []

    def add(self, obs, act, logp, val, rew, done):
        self.obs.append(obs.detach().cpu())
        self.actions.append(act.detach().cpu())
        self.logps.append(logp.detach().cpu().unsqueeze(1))
        self.values.append(val.detach().cpu())
        self.rewards.append(rew.detach().cpu().unsqueeze(1))
        self.dones.append(done.detach().cpu().unsqueeze(1))

    def cat(self, device):
        obs   = torch.cat(self.obs, dim=0).to(device)         # [T*B,2,H,W]
        acts  = torch.cat(self.actions, dim=0).to(device)     # [T*B,N]
        logps = torch.cat(self.logps, dim=0).to(device).squeeze(1)  # [T*B]
        vals  = torch.cat(self.values, dim=0).to(device).squeeze(1) # [T*B]
        rews  = torch.cat(self.rewards, dim=0).to(device).squeeze(1)# [T*B]
        dns   = torch.cat(self.dones, dim=0).to(device).squeeze(1)  # [T*B]
        return obs, acts, logps, vals, rews, dns

def compute_returns_advantages(rews, vals, dones, gamma, lam):
    """
    rews, vals, dones: [T, B]
    retorna: returns [T,B], adv [T,B]
    """
    if rews.ndim == 1:  # caso [T]
        T = rews.shape[0]
        B = 1
        rews = rews.unsqueeze(1)
        vals = vals.unsqueeze(1)
        dones = dones.unsqueeze(1)
    else:
        T, B = rews.shape
        adv  = torch.zeros_like(rews)
        lastgaelam = torch.zeros(B, device=rews.device)
        for t in reversed(range(T-1)):
            nonterm = 1.0 - dones[t+1]
            delta   = rews[t] + gamma * vals[t+1] * nonterm - vals[t]
            lastgaelam = delta + gamma * lam * nonterm * lastgaelam
            adv[t] = lastgaelam
        ret = adv + vals[:-1]
        return ret, adv

def ppo_update(model: nn.Module,
               optimizer_pi, optimizer_v,
               obs, acts, logp_old, vals_old, rets, advs,
               cfg: PPOConfig):
    """
    Atualização PPO (minibatches).
    """
    model.train()
    N = obs.size(0)
    idx = torch.randperm(N, device=obs.device)

    for _ in range(cfg.train_epochs):
        for start in range(0, N, cfg.batch_size):
            sl = idx[start:start+cfg.batch_size]
            o  = obs[sl]
            a  = acts[sl]
            logp_o = logp_old[sl]
            v_old  = vals_old[sl]
            ret    = rets[sl]
            adv    = advs[sl]

            logits_patch, value = model(o)            # [B,N,A], [B,1]
            B, NP, A = logits_patch.shape
            dist = torch.distributions.Categorical(logits=logits_patch.view(-1, A))
            logp_per_patch = dist.log_prob(a.view(-1))     # [B*NP]
            logp = logp_per_patch.view(B, NP).sum(dim=1)   # [B]
            entropy = dist.entropy().view(B, NP).mean(dim=1).mean()

            ratio = torch.exp(logp - logp_o)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value = value.squeeze(1)
            value_loss = F.mse_loss(value, ret)

            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

            optimizer_pi.zero_grad()
            optimizer_v.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer_pi.step()
            optimizer_v.step()