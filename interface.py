import torch
from transformers import AutoTokenizer
import streamlit as st
from burmese_gpt.config import ModelConfig
from burmese_gpt.models import BurmeseGPT
from scripts.download import download_pretrained_model
import os

# Configuration
VOCAB_SIZE = 119547
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

# Create checkpoints directory if it doesn't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- App Layout ---
st.set_page_config(
    page_title="Burmese GPT",
    page_icon=":speech_balloon:",
    layout="wide"
)


# --- Text Generation Function ---
def generate_text(model, tokenizer, device, prompt, max_length=50, temperature=0.7):
    """Generate text from prompt"""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            logits = outputs[:, -1, :]

            # Apply temperature
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat((input_ids, next_token), dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


# --- Download Screen ---
def show_download_screen():
    """Shows download screen until model is ready"""
    st.title("Burmese GPT")
    st.warning("Downloading required model files...")

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        download_pretrained_model()

        # Verify download completed
        if os.path.exists(CHECKPOINT_PATH):
            st.success("Download completed successfully!")
            st.rerun()  # Restart the app
        else:
            st.error("Download failed - file not found")
            st.stop()

    except Exception as e:
        st.error(f"Download failed: {str(e)}")
        st.stop()


# --- Main App ---
def main_app():
    """Main app UI after model is loaded"""

    @st.cache_resource
    def load_model():
        model_config = ModelConfig()
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_config.vocab_size = VOCAB_SIZE
        model = BurmeseGPT(model_config)

        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        return model, tokenizer, device

    # Load model with spinner
    with st.spinner("Loading model..."):
        model, tokenizer, device = load_model()

    # Sidebar
    st.sidebar.title("Burmese GPT")
    st.sidebar.write("A language model for generating and chatting in Burmese")

    # View selection
    view_options = ["Text Generation", "Chat Mode"]
    selected_view = st.sidebar.selectbox("Select Mode", view_options)

    # Generation parameters
    st.sidebar.header("Generation Settings")
    max_length = st.sidebar.slider("Max Length", 20, 500, 100)
    temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 0.7, 0.1)

    # Main content area
    if selected_view == "Text Generation":
        st.header("Burmese Text Generation")

        # Prompt input
        prompt = st.text_area(
            "Enter your prompt in Burmese:",
            value="မြန်မာစာပေ",
            height=100
        )

        # Generate button
        if st.button("Generate Text"):
            if prompt.strip():
                with st.spinner("Generating..."):
                    generated = generate_text(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        prompt=prompt,
                        max_length=max_length,
                        temperature=temperature
                    )
                st.subheader("Generated Text:")
                st.write(generated)
            else:
                st.warning("Please enter a prompt")

    elif selected_view == "Chat Mode":
        st.header("Chat in Burmese")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "မင်္ဂလာပါ! ကျေးဇူးပြု၍ စကားပြောပါ။"}
            ]

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Type your message..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate assistant response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                with st.spinner("Thinking..."):
                    # Combine chat history for context
                    chat_history = "\n".join(
                        f"{msg['role']}: {msg['content']}"
                        for msg in st.session_state.messages[:-1]
                    )
                    full_prompt = f"{chat_history}\nuser: {prompt}\nassistant:"

                    # Generate response
                    full_response = generate_text(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        prompt=full_prompt,
                        max_length=max_length,
                        temperature=temperature
                    )

                # Display response
                message_placeholder.markdown(full_response)

            # Add assistant response to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )


# --- App Flow Control ---
if not os.path.exists(CHECKPOINT_PATH):
    show_download_screen()
else:
    main_app()