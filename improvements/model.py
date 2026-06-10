"""
Shared model definition — CLS-token Temporal Transformer.

Kept byte-compatible with the architecture in ../train.py so that
best_model.pth (the frozen-feature checkpoint) loads here unchanged.
"""

import torch
import torch.nn as nn

EMBED_DIM = 768
SEQUENCE_LENGTH = 30


class TemporalTransformer(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, num_heads=8, num_layers=4,
                 ff_dim=2048, dropout=0.3, seq_len=SEQUENCE_LENGTH):
        super().__init__()
        self.input_norm = nn.LayerNorm(embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, enable_nested_tensor=False,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        B, T, D = x.shape
        x = self.input_norm(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, :T + 1, :]
        x = self.transformer(x)
        return self.head(x[:, 0, :]).squeeze(1)
