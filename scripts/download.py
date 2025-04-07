from huggingface_hub import hf_hub_download
import shutil
import os


def download_pretrained_model():
    downloaded_path = hf_hub_download(
        repo_id="zaibutcooler/burmese-gpt", filename="GPT.pth", cache_dir="checkpoint"
    )

    target_path = os.path.join("checkpoints", "best_model.pth")
    shutil.copy(downloaded_path, target_path)

    print(f"Saved to {target_path}")


if __name__ == "__main__":
    download_pretrained_model()
