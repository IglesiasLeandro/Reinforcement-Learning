from dataclasses import dataclass

PATCH_SIZE = 16
N_ACTIONS = 5  # 0=KEEP,1=GROW,2=SHRINK,3=SMOOTH,4=CLEAN

@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    pi_lr: float = 3e-4
    vf_lr: float = 1e-3
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    train_epochs: int = 4
    batch_size: int = 64
    horizon_T: int = 6
    max_epochs: int = 200
    grad_clip: float = 1.0