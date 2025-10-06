import torch
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.BUSI.DataClass import get_busi_loaders
from models import RLPixelNet_Pretrained as RLPixelNet
from train import train_ppo_segmentation
import torch.optim as optim
import os, glob

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root_dir = "data/BUSI"
    
    # Caminho raiz da pasta BUSI
    root_busi = os.path.join(os.path.dirname(__file__), "..", "data", "BUSI")
    root_busi = os.path.abspath(root_busi)

    # Cria DataLoaders (train, val, test)
    train_loader, val_loader, test_loader = get_busi_loaders(
        root_dir=root_busi,
        size=128,         # redimensionamento (ajuste conforme seu modelo)
        batch_size=2,
        num_workers=0
    )

    # Instancia o modelo
    model = RLPixelNet()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # Treina com PPO
    train_ppo_segmentation(
        model=model,
        optimizer=optimizer,
        dataloader=train_loader,
        device=device,
        T=4,
        updates=200,
        gamma=0.99, lam=0.95,
        clip_eps=0.2, vf_coef=0.5, ent_coef=0.01,
        epochs=4, minibatch_size=4
    )
    # Avalia no conjunto de teste
    from evaluate import evaluate_segmentation
    iou_mean, dice_mean, acc_mean = evaluate_segmentation(
    model=model,
    dataloader=val_loader,
    device=device
    )

    print(f"\n=== Avaliação no conjunto de validação ===")
    print(f"IoU médio:  {iou_mean:.4f}")
    print(f"Dice médio: {dice_mean:.4f}")
    print(f"Acurácia:   {acc_mean:.4f}")
