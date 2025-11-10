import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim

# ============================================================
# Self-Attention Module (Inalterado)
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttentionModule, self).__init__()
        
        # Duas convoluções 1x1 para gerar Query e Key
        self.conv_query = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.conv_key = nn.Conv2d(in_channels, in_channels // 2, 1)
        
        # Conv1x1 opcional para a saída
        self.conv_output = nn.Conv2d(in_channels, in_channels, 1)
        
    def forward(self, x):
        # x shape: [B, C, H, W]
        B, C, H, W = x.size()
        
        # Passo 1: Convoluções 1x1
        query = self.conv_query(x)  # [B, C/2, H, W]
        key = self.conv_key(x)      # [B, C/2, H, W]
        
        # Passo 2: Reshape para multiplicação matricial
        query = query.view(B, C//2, H*W)  # [B, C/2, H×W]
        key = key.view(B, C//2, H*W).permute(0, 2, 1)  # [B, H×W, C/2]
        
        # Passo 3: Multiplicação matricial (attention map)
        attention_map = torch.matmul(key, query)  # [B, H×W, H×W]
        
        # Passo 4: Softmax
        attention_weights = F.softmax(attention_map, dim=-1)  # [B, H×W, H×W]
        
        # Passo 5: Aplicar atenção na entrada
        x_reshaped = x.view(B, C, H*W)  # [B, C, H×W]
        attended = torch.matmul(x_reshaped, attention_weights)  # [B, C, H×W]
        
        # Passo 6: Reshape de volta
        attended = attended.view(B, C, H, W)  # [B, C, H, W]
        attended = self.conv_output(attended)  # [B, C, H, W]
        
        # Passo 7: Soma residual
        output = x + attended  # [B, C, H, W]
        
        return output



class DilatedDecoderBlock(nn.Module):
    """
    Bloco de decoder com dilated convolutions para capturar contexto amplo
    sem perder resolução espacial
    """
    def __init__(self, in_channels, skip_channels, out_channels, dilation=2):
        super().__init__()
        
        # Upsample 2x
        self.up = nn.ConvTranspose2d(in_channels, in_channels, 
                                     kernel_size=2, stride=2)
        
        # Após concatenação: in_channels + skip_channels
        concat_channels = in_channels + skip_channels
        
        # Primeira dilated conv
        self.conv1 = nn.Conv2d(concat_channels, out_channels, 
                               kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Segunda dilated conv
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x, skip):
        """
        Args:
            x: features do decoder [B, in_channels, H, W]
            skip: features do encoder [B, skip_channels, H*2, W*2]
        """
        # Upsample
        x = self.up(x)
        
        # Concatena com skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Dilated convolutions
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        
        return x
    
class PixelDRLSegmentationModel(nn.Module):
    """
    Arquitetura inspirada no paper PixelDRL-MG:
    Input → VGG16 (com skip connections) → SAM → Policy/Value Networks
    """
    def __init__(self, num_actions=2, pretrained=True):
        super().__init__()
        
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES if pretrained else None)
        features = list(vgg.features.children())

        # Ajusta a primeira camada para 2 canais (img + mask)
        first: nn.Conv2d = features[0]  # 3->64
        new_first = nn.Conv2d(
            in_channels=2, out_channels=first.out_channels,
            kernel_size=first.kernel_size, stride=first.stride, 
            padding=first.padding, bias=(first.bias is not None)
        )

        if pretrained:
            w = first.weight.data.mean(dim=1, keepdim=True)  # [64,1,3,3]
            new_first.weight.data = w.repeat(1, 2, 1, 1)     # [64,2,3,3]
            if first.bias is not None:
                new_first.bias.data = first.bias.data

        features[0] = new_first
        
        # Divide o VGG em blocos para extrair skip connections
        # Camadas de skip conforme o paper: conv1-2, conv2-2, conv3-3 
        
        # Bloco 1: até conv1-2 (Índice 3)
        self.vgg_block1 = nn.Sequential(*features[0:4])   # [B, 64, H, W]
        self.pool1 = features[4]
        
        # Bloco 2: até conv2-2 (Índice 8)
        self.vgg_block2 = nn.Sequential(*features[5:9])   # [B, 128, H/2, W/2]
        self.pool2 = features[9]
        
        # Bloco 3: até conv3-3 (Índice 15)
        self.vgg_block3 = nn.Sequential(*features[10:16]) # [B, 256, H/4, W/4]
        self.pool3 = features[16]
        
        # Bloco 4: até conv4-3 (Índice 22) - Feature principal
        self.vgg_block4 = nn.Sequential(*features[17:23]) # [B, 512, H/8, W/8]
        
        # Self-Attention Module (Paper Sec. 3.2)
        self.self_attention = SelfAttentionModule(channels=512)
        
        # Policy Network com decoder
        self.policy_net = PolicyNetworkWithDecoder(
            num_actions=num_actions
        )
        
        # Value Network com decoder
        self.value_net = ValueNetworkWithDecoder()
    
    def forward(self, x):
        """
        Args:
            x: [B, 2, H, W] - imagem + máscara
        
        Returns:
            policy: [B, num_actions, H, W] - mapa de probabilidades de ações
            value: [B, 1, H, W] - mapa de valores
        """
        # ============= ENCODER (VGG) =============
        # Extrai features em diferentes escalas para skip connections
        conv1_2 = self.vgg_block1(x)        # [B, 64, H, W]
        x_pool1 = self.pool1(conv1_2)       # [B, 64, H/2, W/2]
        
        conv2_2 = self.vgg_block2(x_pool1)  # [B, 128, H/2, W/2]
        x_pool2 = self.pool2(conv2_2)       # [B, 128, H/4, W/4]
        
        conv3_3 = self.vgg_block3(x_pool2)  # [B, 256, H/4, W/4]
        x_pool3 = self.pool3(conv3_3)       # [B, 256, H/8, W/8]
        
        conv4_3 = self.vgg_block4(x_pool3)  # [B, 512, H/8, W/8]
        
        # ============= SELF-ATTENTION MODULE =============
        sam_features = self.self_attention(conv4_3)  # [B, 512, H/8, W/8]
        
        # ============= DECODERS =============
        # Passa SAM features + skip connections para ambas as redes
        
        # Policy Network
        policy = self.policy_net(
            sam_features=sam_features,
            skip_conv3_3=conv3_3,
            skip_conv2_2=conv2_2,
            skip_conv1_2=conv1_2
        )  # [B, num_actions, H, W]
        
        # Value Network
        value = self.value_net(
            sam_features=sam_features,
            skip_conv3_3=conv3_3,
            skip_conv2_2=conv2_2,
            skip_conv1_2=conv1_2
        )  # [B, 1, H, W]
        
        return policy, value


class PolicyNetworkWithDecoder(nn.Module):
    """Policy Network com Decoder e Dilated Convolutions"""
    def __init__(self, num_actions=2):
        super().__init__()
        
        # Decoder blocks
        self.decoder_block1 = DilatedDecoderBlock(
            in_channels=512,      # do SAM
            skip_channels=256,    # do conv3-3
            out_channels=256,
            dilation=4
        )
        
        self.decoder_block2 = DilatedDecoderBlock(
            in_channels=256,
            skip_channels=128,    # do conv2-2
            out_channels=128,
            dilation=2
        )
        
        self.decoder_block3 = DilatedDecoderBlock(
            in_channels=128,
            skip_channels=64,     # do conv1-2
            out_channels=64,
            dilation=2
        )
        
        # Output head - policy logits
        self.output_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=2, dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=2, dilation=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_actions, 3, padding=1)
        )
    
    def forward(self, sam_features, skip_conv3_3, skip_conv2_2, skip_conv1_2):
        """
        Args:
            sam_features: [B, 512, H/8, W/8]
            skip_conv3_3: [B, 256, H/4, W/4]
            skip_conv2_2: [B, 128, H/2, W/2]
            skip_conv1_2: [B, 64, H, W]
        
        Returns:
            policy: [B, num_actions, H, W]
        """
        # Decoder com skip connections
        x = self.decoder_block1(sam_features, skip_conv3_3)  # [B, 256, H/4, W/4]
        x = self.decoder_block2(x, skip_conv2_2)             # [B, 128, H/2, W/2]
        x = self.decoder_block3(x, skip_conv1_2)             # [B, 64, H, W]
        
        # Output
        policy = self.output_head(x)  # [B, num_actions, H, W]
        
        return policy


class ValueNetworkWithDecoder(nn.Module):
    """Value Network com Decoder e Dilated Convolutions"""
    def __init__(self):
        super().__init__()
        
        # Decoder blocks
        self.decoder_block1 = DilatedDecoderBlock(
            in_channels=512,      # do SAM
            skip_channels=256,    # do conv3-3
            out_channels=256,
            dilation=4
        )
        
        self.decoder_block2 = DilatedDecoderBlock(
            in_channels=256,
            skip_channels=128,    # do conv2-2
            out_channels=128,
            dilation=2
        )
        
        self.decoder_block3 = DilatedDecoderBlock(
            in_channels=128,
            skip_channels=64,     # do conv1-2
            out_channels=64,
            dilation=2
        )
        
        # Output head - value map
        self.output_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=2, dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=2, dilation=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1)  # 1 canal para value
        )
    
    def forward(self, sam_features, skip_conv3_3, skip_conv2_2, skip_conv1_2):
        """
        Args:
            sam_features: [B, 512, H/8, W/8]
            skip_conv3_3: [B, 256, H/4, W/4]
            skip_conv2_2: [B, 128, H/2, W/2]
            skip_conv1_2: [B, 64, H, W]
        
        Returns:
            value: [B, 1, H, W]
        """
        # Decoder com skip connections
        x = self.decoder_block1(sam_features, skip_conv3_3)  # [B, 256, H/4, W/4]
        x = self.decoder_block2(x, skip_conv2_2)             # [B, 128, H/2, W/2]
        x = self.decoder_block3(x, skip_conv1_2)             # [B, 64, H, W]
        
        # Output
        value = self.output_head(x)  # [B, 1, H, W]
        
        return value

