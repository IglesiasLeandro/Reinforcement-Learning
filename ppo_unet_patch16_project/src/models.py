import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim

# ============================================================
# NOVO: Self-Attention Module (Paper Eq. 1)
# ============================================================
class SelfAttentionModule(nn.Module):
    """
    Self-Attention Module do paper PixelDRL-MG
    M_att = Softmax(W ⊗ f_att + f_att)
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1)
        
    def forward(self, features):
        # Eq. 1: M_att = Softmax(W⊗f + f)
        x = self.conv1x1(features) + features  # Residual connection
        attention_map = F.softmax(x, dim=1)
        return features * attention_map


# ============================================================
# NOVO: Policy e Value Networks com Dilated Convolutions
# ============================================================
class PolicyNetworkDilated(nn.Module):
    """Policy Network com dilated convolutions (Paper Sec. 3.3)"""
    def __init__(self, in_channels, num_actions=2):
        super().__init__()
        # Todas as conv layers usam dilation=2 para campo receptivo maior
        self.conv1 = nn.Conv2d(in_channels, 128, 3, padding=2, dilation=2)
        self.conv2 = nn.Conv2d(128, 64, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(64, num_actions, 3, padding=2, dilation=2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        logits = self.conv3(x)  # [B, num_actions, H, W]
        return logits


class ValueNetworkDilated(nn.Module):
    """Value Network com dilated convolutions (Paper Sec. 3.3)"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, 3, padding=2, dilation=2)
        self.conv2 = nn.Conv2d(128, 64, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(64, 1, 3, padding=2, dilation=2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        value = self.conv3(x)  # [B, 1, H, W]
        return value

class PixelDRLSegmentationModel(nn.Module):
    """
    Arquitetura inspirada no paper PixelDRL-MG:
    Input → VGG16 → SAM → Policy/Value Networks (com Dilated Convs)
    """
    def __init__(self, num_actions=2, pretrained=True):
        super().__init__()
        
        # 1. Feature Extractor: VGG16 (metade dos canais como no paper)
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES if pretrained else None)
        features = list(vgg.features.children())

        first: nn.Conv2d = features[0]  # 3->64
        new_first = nn.Conv2d(
            in_channels=2, out_channels=first.out_channels,
            kernel_size=first.kernel_size, stride=first.stride, padding=first.padding, bias=(first.bias is not None)
        )

        if pretrained:
            # média sobre o eixo de canais e replica para 2 canais
            w = first.weight.data.mean(dim=1, keepdim=True)  # [64,1,3,3]
            new_first.weight.data = w.repeat(1, 2, 1, 1)     # [64,2,3,3]
            if first.bias is not None:
                new_first.bias.data = first.bias.data

        features[0] = new_first
        self.features = nn.Sequential(*features[:23])
        
        # Reduz canais pela metade (512 → 256)
        self.channel_reducer = nn.Conv2d(512, 256, kernel_size=1)
        
        # 2. Self-Attention Module (Paper Sec. 3.2)
        self.self_attention = SelfAttentionModule(channels=256)
        
        # 3. Policy Network com Dilated Convolutions (Paper Sec. 3.3)
        self.policy_net = PolicyNetworkDilated(
            in_channels=256,  # Features do VGG + SAM
            num_actions=num_actions
        )
        
        # 4. Value Network com Dilated Convolutions
        self.value_net = ValueNetworkDilated(in_channels=256)
        
    def forward(self, x):
        """
        x: [B, 2, H, W] (img + mask concatenados)
        Returns: logits [B, 2, H, W], value [B, 1]
        """
        # Extrai features com VGG
        features = self.features(x)  # [B, 512, H/16, W/16]
        
        # Reduz canais
        features = self.channel_reducer(features)  # [B, 256, H/16, W/16]
        
        # Aplica Self-Attention
        features = self.self_attention(features)  # [B, 256, H/16, W/16]
        
        # Upsampling para resolução original (se necessário)
        features_upsampled = F.interpolate(
            features, 
            size=x.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        # Policy e Value
        logits = self.policy_net(features_upsampled)  # [B, 2, H, W]
        value = self.value_net(features_upsampled)     # [B, 1, H, W]
        
        # Value global (média espacial)
        value_global = value.mean(dim=(2, 3), keepdim=True)  # [B,1,1,1]
        value_global = value_global.view(x.size(0), 1)       # [B,1]
        return logits, value_global
