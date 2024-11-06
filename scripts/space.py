import streamlit as st

# Set up the page layout
st.set_page_config(
    page_title="Burmese GPT",
    page_icon=":speech_balloon:",
    layout="wide"
)

# Create a sidebar with a title and a brief description
st.sidebar.title("Burmese GPT")
st.sidebar.write("A language models app for generating and chatting in Burmese.")

# Create a selectbox to choose the view
view_options = ["Sampling", "Chat Interface"]
selected_view = st.sidebar.selectbox("Select a view:", view_options)

# Create a main area
if selected_view == "Sampling":
    st.title("Sampling")
    st.write("Generate text using the pre-trained models:")

    # Create a text input field for the prompt
    prompt = st.text_input("Prompt:", value="")

    # Create a slider to choose the temperature
    temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.5)

    # Create a button to generate text
    generate_button = st.button("Generate")

    # Create an output area to display the generated text
    output_area = st.text_area("Generated Text:", height=200, disabled=True)

    # Add some space between the input and output areas
    st.write("")

elif selected_view == "Chat Interface":
    st.title("Chat Interface")
    st.write("Chat with the fine-tuned models:")

    # Create a text input field for the user input
    user_input = st.text_input("You:", value="")

    # Create a button to send the input to the models
    send_button = st.button("Send")

    # Create an output area to display the models's response
    response_area = st.text_area("Model:", height=200, disabled=True)

    # Add some space between the input and output areas
    st.write("")