import torch
from transformers import AutoTokenizer
import streamlit as st
from burmese_gpt.config import ModelConfig
from burmese_gpt.models import BurmeseGPT
from scripts.download import download_pretrained_model
import os

# Model configuration
VOCAB_SIZE = 119547
CHECKPOINT_PATH = "checkpoints/best_model.pth"

if os.path.exists(CHECKPOINT_PATH):
    st.warning("Model already exists, skipping download.")
else:
    st.info("Downloading model...")
    download_pretrained_model()
    st.success("Model downloaded successfully.")


# Load model function (cached to avoid reloading on every interaction)
@st.cache_resource
def load_model():
    model_config = ModelConfig()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_config.vocab_size = VOCAB_SIZE
    model = BurmeseGPT(model_config)

    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
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


# Set up the page layout
st.set_page_config(
    page_title="Burmese GPT", page_icon=":speech_balloon:", layout="wide"
)

# Create a sidebar with a title and a brief description
st.sidebar.title("Burmese GPT")
st.sidebar.write("A language models app for generating and chatting in Burmese.")

# Create a selectbox to choose the view
view_options = ["Sampling", "Chat Interface"]
selected_view = st.sidebar.selectbox("Select a view:", view_options)

# Load the model once (cached)
model, tokenizer, device = load_model()

# Create a main area
if selected_view == "Sampling":
    st.title("Sampling")
    st.write("Generate text using the pre-trained models:")

    # Create a text input field for the prompt
    prompt = st.text_input("Prompt:", value="မြန်မာ")

    # Add additional generation parameters
    col1, col2 = st.columns(2)
    with col1:
        max_length = st.slider("Max Length:", min_value=10, max_value=500, value=50)
    with col2:
        temperature = st.slider(
            "Temperature:", min_value=0.1, max_value=2.0, value=0.7, step=0.1
        )

    # Create a button to generate text
    if st.button("Generate"):
        if prompt.strip():
            with st.spinner("Generating text..."):
                generated = generate_sample(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    prompt=prompt,
                    max_length=max_length,
                )
            st.text_area("Generated Text:", value=generated, height=200)
        else:
            st.warning("Please enter a prompt")

elif selected_view == "Chat Interface":
    st.title("Chat Interface")
    st.write("Chat with the fine-tuned models:")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            with st.spinner("Thinking..."):
                # Generate response
                generated = generate_sample(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    prompt=prompt,
                    max_length=100,
                )
                full_response = generated

            message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
