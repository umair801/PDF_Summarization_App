import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch
import base64
import os


# Load Model and Tokenizer
MODEL_DIRECTORY = 'LaMini-Flan-T5-248M'
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIRECTORY)
model = T5ForConditionalGeneration.from_pretrained(
    MODEL_DIRECTORY, device_map='auto', torch_dtype=torch.float32
)

def preprocess_file(file):
    """Preprocess the uploaded PDF file."""
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    return ''.join(text.page_content for text in texts)

def generate_summary(filepath):
    """Generate summary using the language model."""
    summarization_pipeline = pipeline(
        'summarization',
        model=model,
        tokenizer=tokenizer,
        max_length=500,
        min_length=50,
    )
    input_text = preprocess_file(filepath)
    result = summarization_pipeline(input_text)
    return result[0]['summary_text']

@st.cache_data
def display_pdf(file):
    """Embed PDF file in the Streamlit app."""
    with open(file, 'rb') as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def main():
    """Main function for Streamlit app."""
    st.set_page_config(layout='wide', page_title='PDF Summarization App')
    st.title('PDF Summarization App using Language Model')

    uploaded_file = st.file_uploader('Upload your PDF file', type=['pdf'])

    if uploaded_file:
        filepath = os.path.join('uploads', uploaded_file.name)
        with open(filepath, 'wb') as f:
            f.write(uploaded_file.read())

        if st.button('Summarize'):
            col1, col2 = st.columns(2)

            with col1:
                st.info('Uploaded PDF:')
                display_pdf(filepath)

            with col2:
                st.info('Generated Summary:')
                try:
                    summary = generate_summary(filepath)
                    st.success(summary)
                except Exception as e:
                    st.error(f"An error occurred: {e}")


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    main()
