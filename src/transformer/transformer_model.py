import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.final_head import FinalHead



class MultiheadAttention(nn.Module):
    """
    Многоголовое самовнимание (Multi-Head Self-Attention).
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            embed_dim (int): Размерность эмбеддинга (d_model).
            num_heads (int): Количество голов внимания.
            dropout (float): Вероятность дропаута.
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Линейные проекции для Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Выходная проекция
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (Tensor): Входной тензор [B, L, D] (Batch, Length, Dim).
            mask (Tensor, optional): Маска внимания [B, L, L] или [1, L, L].
            
        Returns:
            Tensor: Выходной тензор [B, L, D].
        """
        B, L, D = x.shape

        # Проекции Q, K, V: [B, L, D]
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Разделяем на головы: [B, L, H, D/H] -> [B, H, L, D/H]
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Вычисляем attention scores: [B, H, L, L]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = self.softmax(attn_scores)
        attn_weights = self.dropout(attn_weights)

        # Применяем веса внимания к V: [B, H, L, D/H]
        output = torch.matmul(attn_weights, V)

        # Объединяем головы: [B, H, L, D/H] -> [B, L, H, D/H] -> [B, L, D]
        output = output.transpose(1, 2).contiguous().view(B, L, D)

        # Финальная проекция
        output = self.out_proj(output)

        return output



class TransformerEncoderBlock(nn.Module):
    """
    Один блок энкодера Transformer с residual connections и layernorm.
    """
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        """
        Args:
            embed_dim (int): Размерность эмбеддинга.
            num_heads (int): Количество голов внимания.
            ff_dim (int): Размерность скрытого слоя в Feed-Forward сети.
            dropout (float): Вероятность дропаута.
        """
        super().__init__()

        # Слой Multi-Head Attention
        self.attention = MultiheadAttention(embed_dim, num_heads, dropout)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

        # Layer Normalization
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (Tensor): Входной тензор [B, L, D].
            mask (Tensor, optional): Маска внимания.
            
        Returns:
            Tensor: Выходной тензор [B, L, D].
        """
        # Sub-layer 1: Multi-Head Attention + Residual + Norm
        attn_out = self.attention(x, mask)
        x = x + self.dropout(attn_out)
        x = self.ln1(x)

        # Sub-layer 2: Feed-Forward + Residual + Norm
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.ln2(x)

        return x



class TSTransformer(nn.Module):
    """
    Полная архитектура Transformer для временных рядов.
    Принимает [B, 256, 256], возвращает [B, 3].
    """
    def __init__(
        self,
        input_seq_len: int = 256,
        input_dim: int = 256,
        embed_dim: int = 512, # Увеличиваем размерность для большей емкости
        num_heads: int = 8,
        num_layers: int = 6, # Глубокая сеть
        ff_dim: int = 2048,  # Размер FFN в 4 раза больше embed_dim
        dropout: float = 0.1,
        final_head_dim: int = 768, # Размерность для FinalHead
    ):
        """
        Args:
            input_seq_len (int): Длина входной последовательности (256).
            input_dim (int): Размерность входных признаков (256).
            embed_dim (int): Размерность внутреннего представления Transformer.
            num_heads (int): Количество голов внимания.
            num_layers (int): Количество слоев энкодера.
            ff_dim (int): Размерность скрытого слоя FFN.
            dropout (float): Вероятность дропаута.
            head_dim (int): Размерность вектора перед подачей в FinalHead.
        """
        super().__init__()

        self.input_seq_len = input_seq_len
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.final_head_dim = final_head_dim

        # Проекция входных признаков в пространство embed_dim
        self.input_projection = nn.Linear(input_dim, embed_dim)

        # Позиционные эмбеддинги
        self.pos_embedding = nn.Parameter(torch.randn(1, input_seq_len, embed_dim))

        # Стек энкодеров Transformer
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Проекция для головы: преобразует [B, 256 * embed_dim] -> [B, head_dim]
        self.head_proj = nn.Sequential(
            nn.Linear(input_seq_len * embed_dim, final_head_dim),
            nn.LayerNorm(final_head_dim), # Стабилизация
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Финальная голова для предсказания scores
        self.final_head = FinalHead(final_head_dim)

        # Инициализация весов
        self._init_weights()

    def _init_weights(self):
        """Инициализация весов."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход модели.
        
        Args:
            x (Tensor): Входной тензор [B, 256, 256].
            
        Returns:
            Tensor: Предсказанные scores [B, 3].
        """
        B, L, D = x.shape

        # Проекция входа: [B, 256, 256] -> [B, 256, embed_dim]
        x = self.input_projection(x)

        # Добавление позиционных эмбеддингов
        x = x + self.pos_embedding

        # Пропускаем через стек энкодеров
        for layer in self.encoder_layers:
            x = layer(x) # [B, 256, embed_dim]

        # Преобразуем последовательность в один вектор:
        # [B, 256, embed_dim] -> [B, 256 * embed_dim]
        x = x.view(B, -1)

        # Проекция в head_dim: [B, 256 * embed_dim] -> [B, head_dim]
        x = self.head_proj(x)

        # Предсказание scores: [B, head_dim] -> [B, 3]
        scores = self.final_head(x)

        return scores