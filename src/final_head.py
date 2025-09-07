import torch
import torch.nn as nn


class FinalHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim*3),
            nn.GELU(),
            nn.Linear(input_dim*3, input_dim*3)
        )
        
        self.min_proj = nn.Linear(input_dim, 1)
        self.max_proj = nn.Linear(input_dim, 1)
        self.prob_proj = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.mlp(x)

        min_emb, max_emb, prob_emb = torch.chunk(x, 3, dim=-1)

        min_pred = self.min_proj(min_emb)
        max_pred = self.max_proj(max_emb)
        prob_pred = self.prob_proj(prob_emb)

        out = torch.cat([min_pred, max_pred, prob_pred], dim=-1)

        return out 