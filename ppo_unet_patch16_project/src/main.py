# main.py
"""
Main script para treinar segmentação com PixelDRL-MG style PPO
NÃO precisa pré-treinar! O PPO treina end-to-end diretamente.
"""

import torch
import torch.optim as optim
import sys, os

# Ajusta path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.BUSI.DataClass import get_busi_loaders
from models import PixelDRLSegmentationModel  # Seu modelo
from train import train_ppo_segmentation
from train import apply_actions_binary

CONFIG_V1_CONSERVATIVE = {
    # === DADOS ===
    'batch_size': 4,  # Aumentado de 2 para 4
    'size': 128,      # Mantém 128
    
    # === MODELO ===
    'num_actions': 2,
    'pretrained': True,
    
    # === OPTIMIZER ===
    'lr': 3e-3,  # AUMENTADO! Era 1e-3
    
    # === PPO ===
    'T': 4,  # REDUZIDO! Era 6, agora 4 para convergir mais rápido
    'updates': 10000,  # AUMENTADO! Era 5000
    'gamma': 0.95,
    'lam': 0.95,
    'clip_eps': 0.2,
    'vf_coef': 1.0,  # AUMENTADO! Era 0.5
    'ent_coef': 0.05,  # MUITO AUMENTADO! Era 0.01
    'epochs': 4,
    'minibatch_size': 8,  # Aumentado de 4 para 8
    
    # === ESTRATÉGIAS ===
    'init_strategy': 'otsu',  # MUDADO! Era 'image'
    'use_l2_reward': False,   # MUDADO! Volta para IoU reward
    
    # === LOGGING ===
    'warmup_steps': 200,  # Aumentado
    'log_interval': 25,   # Mais frequente
    
    'description': 'Configuração conservadora - Otsu + IoU reward + mais exploração'
}


def main():
    # ============================================================
    # 1. CONFIGURAÇÃO
    # ============================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Usando device: {device}\n")
    
    # Paths
    root_busi = os.path.join(os.path.dirname(__file__), "..", "data", "BUSI")
    root_busi = os.path.abspath(root_busi)
    
    # ============================================================
    # 2. CARREGA DADOS
    # ============================================================
    print("📊 Carregando dataset BUSI...")
    train_loader, val_loader, test_loader = get_busi_loaders(
        root_dir=root_busi,
        size=128,  # Importante: tamanho menor = mais rápido
        batch_size=2,  # Paper usa batch_size=2 por limitação de GPU
        num_workers=0
    )
    
    print(f"✓ Train: {len(train_loader)} batches")
    print(f"✓ Val:   {len(val_loader)} batches")
    print(f"✓ Test:  {len(test_loader)} batches\n")
    
    # ============================================================
    # 3. CRIA MODELO
    # ============================================================
    print("🏗️  Criando modelo PixelDRL-MG Style...")
    model = PixelDRLSegmentationModel(
        num_actions=2,  # Paper: apenas SET_ZERO e DO_NOTHING
        pretrained=True  # Usa VGG16 pré-treinada
    ).to(device)
    
    # Conta parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total de parâmetros: {total_params/1e6:.2f}M")
    print(f"✓ Parâmetros treináveis: {trainable_params/1e6:.2f}M")
    print(f"  (Paper PixelDRL-MG: 7.14M)\n")
    
    # ============================================================
    # 4. OPTIMIZER
    # ============================================================
    # Paper usa Adam com lr=1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Opcional: Scheduler (decay 0.9 a cada 25 epochs como no paper)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.9)
    
    # ============================================================
    # 5. TREINAMENTO PPO (End-to-End, SEM pré-treino!)
    # ============================================================
    print("="*60)
    print("🚀 INICIANDO TREINAMENTO PPO")
    print("="*60)
    print("CONFIGURAÇÃO:")
    print(f"  - Estratégia de inicialização: 'image' (Paper Sec. 3.4)")
    print(f"  - Recompensa: L2 Distance (Paper Eq. 12)")
    print(f"  - Ações: 2 (SET_ZERO / DO_NOTHING)")
    print(f"  - Steps por episódio (T): 6")
    print(f"  - Gamma: 0.95 (Paper)")
    print("="*60 + "\n")
    
    results = train_ppo_segmentation(
        model=model,
        optimizer=optimizer,
        dataloader=train_loader,
        device=device,
        
        # Hiperparâmetros do Paper
        T=10,  # Paper usa t_max=10, comece com 6 para teste
        updates=5000,  # Comece com menos updates para teste
       
        vf_coef=1,
        ent_coef=0.02,  # Pode aumentar para 0.02 se precisar mais exploração
        epochs=4,
        minibatch_size=4,  # Ajuste conforme memória GPU
        # MODIFICAÇÕES DO PAPER
        init_strategy='otsu',  # 'image' (paper) ou 'otsu' (seu código original)
        # Logging
        warmup_steps=200,
        log_interval=25
    )
    
    # ============================================================
    # 6. RESULTADOS DO TREINAMENTO
    # ============================================================
    print("\n" + "="*60)
    print("📈 RESULTADOS DO TREINAMENTO")
    print("="*60)
    print(f"Best IoU durante treino: {results['best_iou']:.4f}")
    print(f"Mean IoU final: {results['mean_iou']:.4f}")
    print(f"Metrics finais:")
    for metric, value in results['final_metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    # ============================================================
    # 7. AVALIAÇÃO NO CONJUNTO DE VALIDAÇÃO (Opcional)
    # ============================================================
    print("\n" + "="*60)
    print("🔍 AVALIAÇÃO NO CONJUNTO DE VALIDAÇÃO")
    print("="*60)
    
    model.eval()
    val_ious = []
    val_dices = []
    
    with torch.no_grad():
        for img, gt in val_loader:
            img, gt = img.to(device), gt.to(device)
            
            # Inicializa com imagem (como no paper)
            mask = img.clone()
            
            # Refinamento iterativo
            for _ in range(6):  # T=6 steps
                state = torch.cat([img, mask], dim=1)
                logits, _ = model(state)
                
                # Amostra ações
                probs = torch.softmax(logits, dim=1)
                actions = torch.argmax(probs, dim=1)  # Greedy na validação
                
                # Aplica ações binárias
                # garante HxW compatível
                if actions.shape[-2:] != mask.shape[-2:]:
                    actions = F.interpolate(actions.unsqueeze(1).float(), size=mask.shape[-2:], mode="nearest").long().squeeze(1)

                mask = apply_actions_binary(mask, actions)  # lida com o canal
            
            # Calcula métricas
            pred = (mask > 0.5).float()
            gt_bin = (gt > 0.5).float()
            
            # IoU
            inter = (pred * gt_bin).sum(dim=(1,2,3))
            union = (pred + gt_bin - pred * gt_bin).sum(dim=(1,2,3))
            iou = (inter + 1e-6) / (union + 1e-6)
            val_ious.extend(iou.cpu().tolist())
            
            # Dice
            dice = (2 * inter + 1e-6) / (pred.sum(dim=(1,2,3)) + gt_bin.sum(dim=(1,2,3)) + 1e-6)
            val_dices.extend(dice.cpu().tolist())
    
    print(f"✓ IoU médio (validação):  {sum(val_ious)/len(val_ious):.4f}")
    print(f"✓ Dice médio (validação): {sum(val_dices)/len(val_dices):.4f}")
    
    # ============================================================
    # 8. SALVA MODELO FINAL
    # ============================================================
    save_path = "pixeldrl_final.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'results': results,
        'val_iou': sum(val_ious)/len(val_ious),
        'val_dice': sum(val_dices)/len(val_dices),
    }, save_path)
    print(f"\n💾 Modelo salvo em: {save_path}")
    
    # ============================================================
    # 9. COMPARAÇÃO COM EXPECTATIVA DO PAPER
    # ============================================================
    print("\n" + "="*60)
    print("📊 COMPARAÇÃO COM PAPER PixelDRL-MG")
    print("="*60)
    print("Paper (Cardiac dataset):")
    print("  DICE: 0.8346 | IoU: 0.7510 | BIoU: 0.4081")
    print("\nSeu resultado (BUSI dataset):")
    print(f"  DICE: {results['final_metrics'].get('f1', 0.0):.4f} | IoU: {results['mean_iou']:.4f}")
    print("\n⚠️  BUSI é diferente de Cardiac, então métricas podem variar!")
    print("   O importante é ver se as métricas MELHORAM durante o treino.")


if __name__ == "__main__":
    main()