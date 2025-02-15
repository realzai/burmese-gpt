from huggingface_hub import upload_file
import os


def upload_model():
    if not os.path.exists("checkpoints/best_model.pth"):
        print("File does not exist.")
        return

    upload_file(
        path_or_fileobj="checkpoints/best_model.pth",
        path_in_repo="GPT.pth",
        repo_id="zaibutcooler/burmese-gpt",
        repo_type="model",
    )


if __name__ == "__main__":
    upload_model()
