import torch
from .io_utils import set_seed
from .config import PPOConfig
from .model import ActorCriticUNet
from .ppo import RolloutBuffer, compute_returns_advantages, ppo_update
from .metrics import dice_coef
from .env import env_step

def train(device="cuda"):
    set_seed(42)
    cfg = PPOConfig()
    model = ActorCriticUNet(in_ch=2, base_ch=32, n_actions=cfg.__dict__.get("n_actions", None) or None).to(device)
    # NOTE: model init will use default N_ACTIONS from model.py if not passed

    optimizer_pi = torch.optim.Adam(model.parameters(), lr=cfg.pi_lr)
    optimizer_v  = torch.optim.Adam(model.parameters(), lr=cfg.vf_lr)

    # >>> Substitua por seu DataLoader BUSI
    def get_busi_batch():
        B, H, W = 8, 256, 256
        imgs = torch.rand(B,1,H,W, device=device)  # normalizada [0,1]
        gts  = (torch.rand(B,1,H,W, device=device) > 0.7).float()
        return imgs, gts

    for epoch in range(1, cfg.max_epochs+1):
        model.eval()
        buffer = RolloutBuffer()

        imgs, gts = get_busi_batch()
        B, _, H, W = imgs.shape
        mask = torch.zeros_like(gts)

        vals_bootstrap = None

        for t in range(cfg.horizon_T):
            mask_next, actions, logp_sum, value = env_step(model, imgs, mask, gts)

            dice_t   = dice_coef((mask>0.5).float(), gts)         # [B]
            dice_tp1 = dice_coef((mask_next>0.5).float(), gts)    # [B]
            rew_vec  = (dice_tp1 - dice_t)                        # [B]

            done_vec = torch.zeros(B, device=device)
            state = torch.cat([imgs, mask], dim=1)
            buffer.add(state, actions, logp_sum, value, rew_vec, done_vec)

            mask = mask_next

            if t == cfg.horizon_T - 1:
                last_state = torch.cat([imgs, mask], dim=1)
                _, v_last = model(last_state)
                vals_bootstrap = v_last.squeeze(1)  # [B]

        obs, acts, logps_old, vals, rews, dones = buffer.cat(device)
        T = cfg.horizon_T
        obs  = obs.view(T, B, *obs.shape[1:])              # [T,B,2,H,W]
        acts = acts.view(T, B, -1)                         # [T,B,Np]
        logp_old = logps_old.view(T, B)                    # [T,B]
        vals = vals.view(T, B)                             # [T,B]
        rews = rews.view(T, B)                             # [T,B]
        dones = dones.view(T, B)                           # [T,B]

        vals_plus = torch.cat([vals, vals_bootstrap.unsqueeze(0)], dim=0)  # [T+1,B]

        returns = []
        advs = []
        for b in range(B):
            ret_b, adv_b = compute_returns_advantages(rews[:, b], vals_plus[:, b], dones[:, b],
                                                      gamma=cfg.gamma, lam=cfg.lam)
            returns.append(ret_b.unsqueeze(1))
            advs.append(adv_b.unsqueeze(1))
        returns = torch.cat(returns, dim=1)
        advs    = torch.cat(advs, dim=1)

        obs_train   = obs[:-1].reshape(-1, *obs.shape[2:])
        acts_train  = acts[:-1].reshape(-1, acts.shape[2])
        logp_train  = logp_old[:-1].reshape(-1)
        vals_old    = vals[:-1].reshape(-1)
        returns_tr  = returns.reshape(-1)
        advs_tr     = advs.reshape(-1)

        advs_tr = (advs_tr - advs_tr.mean()) / (advs_tr.std() + 1e-8)

        ppo_update(model, optimizer_pi, optimizer_v,
                   obs_train, acts_train, logp_train, vals_old,
                   returns_tr, advs_tr, cfg)

        with torch.no_grad():
            d_final = dice_coef((mask>0.5).float(), gts).mean().item()
        if epoch % 5 == 0:
            print(f"[Epoch {epoch:03d}] Dice(final@T={cfg.horizon_T}): {d_final:.4f}")

    print("Treino concluído!")