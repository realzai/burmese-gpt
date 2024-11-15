from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 30000
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1

@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 5e-5
    num_epochs: int = 5
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    save_every: int = 1
    eval_every: int = 1
    dataset_url: str = "zaibutcooler/wiki-burmese"