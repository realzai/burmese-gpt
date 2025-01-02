import unittest
from burmese_gpt.data.dataset import BurmeseDataset
from burmese_gpt.config import TrainingConfig


class TestData(unittest.TestCase):
    def test_data(self):
        training_config = TrainingConfig()
        train_dataset = BurmeseDataset(split="train[:90%]", config=training_config)
        val_dataset = BurmeseDataset(split="train[90%:]", config=training_config)

        self.assertIsNotNone(train_dataset)
        self.assertIsNotNone(val_dataset)
