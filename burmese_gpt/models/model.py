from torch import nn
from burmese_gpt.config import ModelConfig

class BurmeseGPT(nn.Module):
    def __init__(self,config:ModelConfig):
        super(BurmeseGPT, self).__init__()
        self.config = config
        # Continue the rest