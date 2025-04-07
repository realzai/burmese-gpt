import torch
from torch import nn
from burmese_gpt.config import ModelConfig


class BurmeseGPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super(BurmeseGPT, self).__init__()
        self.config = config

        # Simple embedding layer
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)

        # Positional embeddings
        self.pos_embedding = nn.Embedding(1024, config.embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )

        # Final projection layer
        self.fc = nn.Linear(config.embed_dim, config.vocab_size)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x: input tensor [batch_size, seq_len]
        Returns:
            output tensor [batch_size, seq_len, vocab_size]
        """
        device = x.device
        batch_size, seq_len = x.size()

        # Create position IDs
        positions = torch.arange(seq_len, device=device).expand(batch_size, seq_len)

        # Get embeddings
        token_embeds = self.embedding(x)
        pos_embeds = self.pos_embedding(positions)
        x = token_embeds + pos_embeds
        x = self.dropout(x)

        # Create attention mask
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

        # Transformer
        x = self.transformer(x, mask)

        # Final projection
        return self.fc(x)
