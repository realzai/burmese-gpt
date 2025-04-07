import torch
from transformers import AutoTokenizer
from burmese_gpt.config import ModelConfig
from burmese_gpt.models import BurmeseGPT

VOCAB_SIZE = 119547
CHECKPOINT_PATH = "checkpoints/best_model.pth"


def download_pretrained_model(path: str):
    pass


def load_model(path: str):
    model_config = ModelConfig()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_config.vocab_size = VOCAB_SIZE

    model = BurmeseGPT(model_config)

    # Load checkpoint
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, tokenizer, device


def generate_sample(model, tokenizer, device, prompt="မြန်မာ", max_length=50):
    """Generate text from prompt"""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token = outputs[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat((input_ids, next_token), dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    # Download the pretrained model
    # download_pretrained_model(CHECKPOINT_PATH)

    print("Loading model...")
    model, tokenizer, device = load_model(CHECKPOINT_PATH)

    while True:
        prompt = input("\nEnter prompt (or 'quit' to exit): ")
        if prompt.lower() == "quit":
            break

        print("\nGenerating...")
        generated = generate_sample(model, tokenizer, device, prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated}")
