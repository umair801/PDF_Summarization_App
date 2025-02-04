# PDF Summarization App

## Overview
This is a powerful and user-friendly web application for summarizing PDF files using advanced natural language processing techniques. It is built with **Streamlit** and integrates the **LaMini-Flan-T5 language model** for efficient and high-quality text summarization.

## Features
- **PDF Uploading**: Easily upload PDF files for processing.
- **PDF Display**: View the uploaded PDF within the app interface.
- **Summarization**: Generate concise and meaningful summaries of uploaded PDFs.
- **Interactive Interface**: A clean and responsive UI powered by Streamlit.

## Technology Stack
- **Python**: Core programming language.
- **Streamlit**: For creating the web interface.
- **Transformers**: Leveraging T5Tokenizer and T5ForConditionalGeneration.
- **LangChain**: For text splitting and preprocessing.
- **PyTorch**: For model inference.

## Setup and Installation

### Prerequisites
- Python 3.8 or higher.
- Basic knowledge of Python virtual environments.

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/umair801/PDF_Summarization_App.git

## Model Information
This application uses the LaMini-Flan-T5-248M model for summarization. The model is available for free on Huggingface: https://huggingface.co/MBZUAI/LaMini-Flan-T5-248M. If it has not been downloaded, you can manually download it from the link above and place it in the model/LaMini-Flan-T5-248M directory within the project folder. The model does not require an API key to use.
