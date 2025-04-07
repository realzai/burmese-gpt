# Burmese GPT

## Overview
Burmese GPT is a large language model with 20M parameters trained on the Burmese language. This project aims to provide a foundation for building and experimenting with language models for the Burmese language.

## Installation
To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Training
To train the model, use the following command:
```bash
python -m scripts.train
```

## Push Model to Hugging Face Hub
To push the trained model to the Hugging Face Hub, use the following command:
```bash
python -m scripts.upload
```

## Download Pre-trained Model
To download the pre-trained model, use the following command:
```bash
python -m scripts.download
```

## Sampling
To sample text from the trained model, use the following command:
```bash
python -m scripts.sample
```

## Run Streamlit App
To run the Streamlit app, use the following command:
```bash
streamlit run interface.py
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
