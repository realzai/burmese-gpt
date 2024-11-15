import os
import logging

from burmese_gpt.models import BurmeseGPT
from burmese_gpt.training import BurmeseGPTTrainer
from burmese_gpt.data import BurmeseDataset
from burmese_gpt.config import ModelConfig, TrainingConfig

from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    model_config = ModelConfig()
    training_config = TrainingConfig()

    os.makedirs(training_config.checkpoint_dir, exist_ok=True)

    logger.info(f"Loading dataset from {training_config.dataset_url}")

    train_dataset = BurmeseDataset(split="train[:90%]")  # First 90% for training
    val_dataset = BurmeseDataset(split="train[90%:]")  # Last 10% for validation

    model_config.vocab_size = train_dataset.tokenizer.vocab_size
    logger.info(f"Using vocab size: {model_config.vocab_size}")

    logger.info("Initializing model...")
    model = BurmeseGPT(model_config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size
    )

    logger.info("Starting training...")
    trainer = BurmeseGPTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config
    )

    metrics = trainer.train()

    logger.info("Training completed!")


