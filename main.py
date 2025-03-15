import streamlit as st
import os
import tempfile
import json
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import ollama
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings

# Initialize session state for chat history and extracted text
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'saved_text' not in st.session_state:
    st.session_state.saved_text = ""
if 'duplicates' not in st.session_state:
    st.session_state.duplicates = []
if 'topic_memory' not in st.session_state:
    st.session_state.topic_memory = {}

st.title("📚 Study Helper Chatbot")

# Sidebar for Notepad
st.sidebar.header("📝 Notepad")
st.session_state.saved_text = st.sidebar.text_area("Write your notes here:", st.session_state.saved_text, height=200)

# Sidebar for Duplicate Extracted Text
st.sidebar.header("📄 Duplicate Extracted Text")
if st.session_state.duplicates:
    st.sidebar.write("\n".join(st.session_state.duplicates))
    if st.sidebar.button("Clear Duplicates"):
        st.session_state.duplicates = []
        st.sidebar.success("Duplicates cleared!")

# File uploader
uploaded_files = st.file_uploader("Upload PDFs, Images, or Text Files", accept_multiple_files=True)

# Function to extract text from PDF
@st.cache_data
def extract_text_from_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    extracted_text = ""
    for img in images:
        text = pytesseract.image_to_string(img, lang="eng+hin")
        extracted_text += text + "\n"
    return extracted_text

# Function to extract text from an image
@st.cache_data
def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang="eng+hin")
    return text

# Function to process text files
@st.cache_data
def process_text_file(text_path):
    with open(text_path, 'r', encoding='utf-8') as file:
        return file.read()

# Process uploaded files
def process_uploaded_files():
    if uploaded_files:
        st.session_state.extracted_text = ""
        text_set = set()
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as temp_file:
                temp_file.write(file.read())
                temp_path = temp_file.name
            
            if file.type == "application/pdf":
                extracted_text = extract_text_from_pdf(temp_path)
            elif file.type.startswith("image"):
                extracted_text = extract_text_from_image(temp_path)
            elif file.type == "text/plain":
                extracted_text = process_text_file(temp_path)
            
            for line in extracted_text.split("\n"):
                if line in text_set:
                    st.session_state.duplicates.append(line)
                else:
                    text_set.add(line)
            
            st.session_state.extracted_text += extracted_text + "\n"

if uploaded_files and st.button("Process Uploaded Files"):
    process_uploaded_files()
    st.success("Files processed successfully!")

# Buttons to manage extracted text
if st.session_state.extracted_text:
    if st.button("View and Edit Extracted Text"):
        updated_text = st.text_area("Extracted Text:", st.session_state.extracted_text, height=200)
        st.session_state.extracted_text = updated_text
    
    if st.button("Clear Extracted Text"):
        st.session_state.extracted_text = ""
        st.success("Extracted text cleared!")
    
    if st.button("Save Extracted Text Permanently"):
        st.session_state.saved_text = st.session_state.extracted_text
        st.success("Extracted text saved permanently!")

# Button to view and edit permanent memory
if st.session_state.saved_text:
    if st.button("View and Edit Permanent Memory"):
        updated_saved_text = st.text_area("Permanent Memory:", st.session_state.saved_text, height=200)
        st.session_state.saved_text = updated_saved_text

# Convert text into embeddings and store in FAISS
@st.cache_resource
def create_faiss_index(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    texts = text_splitter.split_text(text)
    embeddings = OllamaEmbeddings(model="llama3")
    vector_store = FAISS.from_texts(texts, embeddings)
    return vector_store

if st.session_state.extracted_text:
    faiss_index = create_faiss_index(st.session_state.extracted_text)
    st.success("Text indexed successfully!")

# Chatbot interaction
user_query = st.text_input("Ask a question:")

if user_query:
    relevant_docs = faiss_index.similarity_search(user_query, k=5)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    if context.strip():
        prompt = f"Context: {context}\n\nQuestion: {user_query}\n\nAnswer:"
        response = ollama.chat(model="llama3", messages=[{"role": "system", "content": "You are a helpful study assistant."},
                                                              {"role": "user", "content": prompt}])
        answer = response['message']['content']
    else:
        answer = "Answer is not available in the context. Generating an AI-based response..."
        response = ollama.chat(model="llama3", messages=[{"role": "system", "content": "You are a helpful study assistant."},
                                                              {"role": "user", "content": user_query}])
        answer += "\n" + response['message']['content']
    
    st.session_state.chat_history.append((user_query, answer))
    st.write("**AI:**", answer)

# Chat history with persistent memory
if st.session_state.chat_history:
    st.subheader("Chat History")
    for query, response in st.session_state.chat_history:
        st.write(f"**You:** {query}")
        st.write(f"**AI:** {response}")
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")
