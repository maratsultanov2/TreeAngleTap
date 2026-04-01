"""
TreeAngleTap (TAT-7) - Lightweight neural network architecture.

Author: Marat Sultanow
License: MIT (code), CC BY-NC-ND 4.0 (weights)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TAT7(nn.Module):
    """
    TreeAngleTap (TAT-7) - 5 pyramids architecture.
    
    Pyramid 1: FILTER - splits features into three types
        - SOLO: main features (weight=0.55)
        - RHYTHM: structural patterns (weight=0.43)
        - HARMONY: connections between features (weight=2.0)
    
    Pyramid 2: COMPRESSOR - data compression (1 layer, 128 units)
    Pyramid 3: HEADS - parallel processing (5 heads, 64 dim)
    Pyramid 4: MEMORY - context storage (T=0.7, entropy=1.18)
    Pyramid 5: OUTPUT - classification (1x128_silu, dropout=0.3)
    """
    
    def __init__(self, input_dim=3072, n_classes=10, n_heads=5, head_dim=64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        
        # Pyramid 1: FILTER
        self.register_buffer('filter_solo', torch.tensor(0.55))
        self.register_buffer('filter_rhythm', torch.tensor(0.43))
        self.register_buffer('filter_harmony', torch.tensor(2.0))
        
        # Pyramid 2: COMPRESSOR
        self.compressor = nn.Linear(input_dim, 128)
        
        # Pyramid 3: HEADS
        self.heads = nn.ModuleList([nn.Linear(128, head_dim) for _ in range(n_heads)])
        
        # Head roles (2 solo, 2 rhythm, 3 harmony)
        self.head_roles = []
        for i in range(n_heads):
            if i < 2:
                self.head_roles.append('solo')
            elif i < 4:
                self.head_roles.append('rhythm')
            else:
                self.head_roles.append('harmony')
        
        # Pyramid 4: MEMORY
        self.temperature = nn.Parameter(torch.tensor(0.7), requires_grad=False)
        self.harmony_matrix = nn.Parameter(torch.eye(n_heads) * 0.7)
        
        # Pyramid 5: OUTPUT
        self.classifier = nn.Sequential(
            nn.Linear(n_heads * head_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        if len(x.shape) == 4:
            x = x.view(x.size(0), -1)
        x = F.relu(self.compressor(x))
        
        role_weights = {
            'solo': self.filter_solo,
            'rhythm': self.filter_rhythm,
            'harmony': self.filter_harmony
        }
        
        head_outputs = []
        for i, head in enumerate(self.heads):
            out = head(x)
            out = out * role_weights[self.head_roles[i]]
            head_outputs.append(out.unsqueeze(1))
        
        head_outputs = torch.cat(head_outputs, dim=1)
        harmony = F.softmax(self.harmony_matrix / self.temperature.clamp(min=0.01), dim=-1)
        harmonized = torch.einsum('bnd,mn->bmd', head_outputs, harmony)
        flat = harmonized.reshape(harmonized.size(0), -1)
        return self.classifier(flat)
    
    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=1)
    
    def get_params_count(self):
        return sum(p.numel() for p in self.parameters())
