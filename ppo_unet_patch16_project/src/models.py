import torch
import torch.nn as nn
from torchvision.models import vgg16

# ---------------------------------------------------------
# Attention Block (bottleneck)
# ---------------------------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.query = nn.Conv2d(ch, ch // 8, 1)
        self.key   = nn.Conv2d(ch, ch // 8, 1)
        self.value = nn.Conv2d(ch, ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H*W)
        k = self.key(x).view(B, -1, H*W)
        attn = torch.softmax(torch.bmm(q.transpose(1,2), k), dim=-1)
        v = self.value(x).view(B, -1, H*W)
        out = torch.bmm(v, attn.transpose(1,2)).view(B, C, H, W)
        return self.gamma * out + x

# ---------------------------------------------------------
# ConvBlock simples
# ---------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

# ---------------------------------------------------------
# Modelo final: Encoder pré-treinado + Attention + Atores
# ---------------------------------------------------------
class RLPixelNet_Pretrained(nn.Module):
    def __init__(self, in_ch=2, n_actions=3, base_ch=512):
        super().__init__()

        # ------------- Encoder VGG16 -------------
        vgg = vgg16(weights="IMAGENET1K_V1").features
        vgg[0] = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1)
        self.encoder = vgg  # saída: [B, 512, H/32, W/32]

        # ------------- Attention Block -------------
        self.att = SelfAttention(base_ch)

        # ------------- Actor (decoder) -------------
        self.actor = nn.Sequential(
            ConvBlock(base_ch, base_ch//2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(base_ch//2, base_ch//4),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(base_ch//4, base_ch//8),
            nn.Conv2d(base_ch//8, n_actions, 1)
        )

        # ------------- Critic (decoder) -------------
        self.critic = nn.Sequential(
            ConvBlock(base_ch, base_ch//2),
            ConvBlock(base_ch//2, base_ch//4),
            ConvBlock(base_ch//4, base_ch//8),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_ch//8, 1)
        )

    def forward(self, state):
        feats = self.encoder(state)
        feats = self.att(feats)
        logits = self.actor(feats)   # [B, 3, H', W']
        value  = self.critic(feats)  # [B, 1]
        return logits, value
