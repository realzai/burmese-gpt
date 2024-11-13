import torch
from torch.optim import AdamW
from tqdm import tqdm
import logging
from typing import Dict

from burmese_gpt.config import TrainingConfig

logger = logging.getLogger(__name__)

class BurmeseGPTTrainer:
    def __init__(self, model, train_loader, val_loader, config:TrainingConfig):
        """
        Trainer for BurmeseGPT model

        Args:
            model: Initialized BurmeseGPT model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            config: Training configuration
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device
        self.model.to(self.device)

        # Initialize optimizer (using same settings as your original)
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay if hasattr(config, 'weight_decay') else 0.01
        )

        # Loss function (ignoring padding tokens)
        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=train_loader.dataset.tokenizer.pad_token_id
        )

    def train_epoch(self) -> float:
        """Run one training epoch, return average loss"""
        self.model.train()
        total_loss = 0

        for batch in tqdm(self.train_loader, desc="Training"):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)

            # Shift inputs and targets (as in your original code)
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # Calculate loss (same as original)
            loss = self.criterion(
                outputs.reshape(-1, outputs.size(-1)),
                targets.reshape(-1)
            )

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self) -> float:
        """Run validation, return average loss"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids = batch["input_ids"].to(self.device)
                inputs = input_ids[:, :-1]
                targets = input_ids[:, 1:]

                outputs = self.model(inputs)
                loss = self.criterion(
                    outputs.reshape(-1, outputs.size(-1)),
                    targets.reshape(-1)
                )
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self) -> Dict[str, list]:
        """
        Full training loop

        Returns:
            Dictionary with training metrics
        """
        metrics = {'train_loss': [], 'val_loss': []}
        best_loss = float('inf')

        for epoch in range(1, self.config.num_epochs + 1):
            logger.info(f"Epoch {epoch}/{self.config.num_epochs}")

            # Training
            train_loss = self.train_epoch()
            metrics['train_loss'].append(train_loss)

            # Validation
            val_loss = self.validate()
            metrics['val_loss'].append(val_loss)

            logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                self.save_checkpoint("best_model.pth")
                logger.info("Saved best model")

            # Save periodic checkpoint
            if epoch % self.config.save_every == 0:
                self.save_checkpoint(f"epoch_{epoch}.pth")

        return metrics

    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, f"{self.config.checkpoint_dir}/{filename}")