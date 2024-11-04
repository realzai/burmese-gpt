import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from .config import ModelConfig

class BurmeseGPT(nn.Module):
    def __init__(self,config:ModelConfig):
        super(BurmeseGPT, self).__init__()
        self.config = config
        # Token and positional embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)

        # Transformer layers
        encoder_layers = TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=4 * config.embed_dim,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layers, config.num_layers)

        # Output layer
        self.ln = nn.LayerNorm(config.embed_dim)
        self.fc = nn.Linear(config.embed_dim, config.vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, attention_mask=None):
        device = x.device
        seq_len = x.size(1)

        # Create position ids
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)

        # Get embeddings
        token_embeds = self.token_embedding(x)
        pos_embeds = self.pos_embedding(position_ids)
        x = token_embeds + pos_embeds

        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

        # Transformer
        x = self.transformer(x, mask=mask, src_key_padding_mask=attention_mask)
        x = self.ln(x)
        logits = self.fc(x)

        return logits