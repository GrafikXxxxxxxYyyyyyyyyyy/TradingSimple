import torch
import torch.nn as nn
import torch.nn.functional as F

from src.final_head import FinalHead



class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_dim=256, num_classes=3, head_dim=768):
        super(CNNFeatureExtractor, self).__init__()
        
        # Стек 1D сверток
        self.conv1 = nn.Conv1d(input_dim, 512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 1024, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(1024)
        self.conv3 = nn.Conv1d(1024, 1024, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(1024)
        
        # Глобальный пулинг для агрегации всей последовательности в один вектор
        self.global_pool = nn.AdaptiveAvgPool1d(1) # или nn.AdaptiveMaxPool1d(1)
        
        # Проекция в head_dim
        self.head_proj = nn.Sequential(
            nn.Linear(1024, head_dim),
            nn.LayerNorm(head_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Финальная голова
        self.final_head = FinalHead(head_dim)

    def forward(self, x):
        # x: [B, 256, 256] -> [B, 256, C] -> нужно [B, C, 256] для Conv1d
        x = x.transpose(1, 2) # [B, 256, 256] -> [B, 256, 256] (каналы, длина)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Агрегация: [B, 1024, 256] -> [B, 1024, 1] -> [B, 1024]
        x = self.global_pool(x).squeeze(-1)
        
        # Проекция и предсказание
        x = self.head_proj(x)
        scores = self.final_head(x)
        
        return scores