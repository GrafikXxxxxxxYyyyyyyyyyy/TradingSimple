import torch
import torch.nn as nn
import torch.nn.functional as F

from src.final_head import FinalHead



class LSTMWithAttention(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, num_layers=2, head_dim=768):
        super(LSTMWithAttention, self).__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1, bidirectional=True)
        self.hidden_dim = hidden_dim * 2 # из-за bidirectional
        
        # Слой внимания
        self.attention = nn.Linear(self.hidden_dim, 1)
        
        self.head_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, head_dim),
            nn.LayerNorm(head_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.final_head = FinalHead(head_dim)

    def forward(self, x):
        # x: [B, 256, 256]
        lstm_out, _ = self.lstm(x) # [B, 256, hidden_dim*2]
        
        # Вычисляем веса внимания
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1) # [B, 256, 1]
        
        # Взвешенная сумма скрытых состояний
        context_vector = torch.sum(attn_weights * lstm_out, dim=1) # [B, hidden_dim*2]
        
        x = self.head_proj(context_vector)
        scores = self.final_head(x)
        
        return scores