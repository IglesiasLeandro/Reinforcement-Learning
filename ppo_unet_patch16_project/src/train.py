import torch
from .io_utils import set_seed
from .config import PPOConfig
from .models import ActorCriticUNet
from .ppo import RolloutBuffer, compute_returns_advantages, ppo_update
from .metrics import dice_coef
from .env import env_step

# importa seu dataset
from data.BUSI.DataClass import get_busi_loaders

def train(device="cuda"):
    set_seed(42)
    cfg = PPOConfig()
    model = ActorCriticUNet(
        in_ch=2, base_ch=32,
        n_actions=cfg.__dict__.get("n_actions", None) or None
    ).to(device)

    optimizer_pi = torch.optim.Adam(model.parameters(), lr=cfg.pi_lr)
    optimizer_v  = torch.optim.Adam(model.parameters(), lr=cfg.vf_lr)

    # ------------------------------
    # usa DataLoader BUSI
    # ------------------------------
    root = "./data/BUSI"   # ajuste se precisar
    train_loader, val_loader, test_loader = get_busi_loaders(
        root, size=256, batch_size=cfg.batch_size, num_workers=0
    )
    train_iter = iter(train_loader)

    def get_busi_batch():
        nonlocal train_iter
        try:
            imgs, gts = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            imgs, gts = next(train_iter)
        return imgs.to(device), gts.to(device)

    # ------------------------------
    # loop de treino PPO
    # ------------------------------
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
            buffer.add(
                state.detach(),
                actions.detach(),
                logp_sum.detach(),
                value.detach(),
                rew_vec.detach(),
                done_vec.detach()
            )

            mask = mask_next

            if t == cfg.horizon_T - 1:
                last_state = torch.cat([imgs, mask], dim=1)
                _, v_last = model(last_state)
                vals_bootstrap = v_last.squeeze(1)  # [B]

        # ------------------------------
        # PPO update
        # ------------------------------
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
            ret_b, adv_b = compute_returns_advantages(
                rews[:, b].unsqueeze(1),
                vals_plus[:, b].unsqueeze(1),
                dones[:, b].unsqueeze(1),
                gamma=cfg.gamma, lam=cfg.lam
            )
            returns.append(ret_b.unsqueeze(1))
            advs.append(adv_b.unsqueeze(1))
        returns = torch.cat(returns, dim=1)
        advs    = torch.cat(advs, dim=1)

        obs_train   = obs[:-1].reshape(-1, *obs.shape[2:]).detach()
        acts_train  = acts[:-1].reshape(-1, acts.shape[2]).detach()
        logp_train  = logp_old[:-1].reshape(-1).detach()
        vals_old    = vals[:-1].reshape(-1).detach()
        returns_tr  = returns.reshape(-1).detach()
        advs_tr     = advs.reshape(-1).detach()

        advs_tr = (advs_tr - advs_tr.mean()) / (advs_tr.std() + 1e-8)

        ppo_update(model, optimizer_pi, optimizer_v,
                   obs_train, acts_train, logp_train, vals_old,
                   returns_tr, advs_tr, cfg)

        with torch.no_grad():
            d_final = dice_coef((mask>0.5).float(), gts).mean().item()
        if epoch % 5 == 0:
            print(f"[Epoch {epoch:03d}] Dice(final@T={cfg.horizon_T}): {d_final:.4f}")

    print("Treino concluído!")
