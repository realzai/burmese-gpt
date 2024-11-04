from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 30000
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    max_seq_len: int = 512

@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 5e-5
    num_epochs: int = 10
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"