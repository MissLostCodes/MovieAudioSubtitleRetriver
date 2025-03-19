import os
import json
import streamlit as st
import pandas as pd
import torch
import torchaudio
import whisper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import OpenAI

def load_whisper_model():
    """Loads the Whisper model for speech-to-text conversion."""
    model = whisper.load_model("base")
    return model

def convert_audio_to_text(audio_file, model):
    """Converts input audio to text using Whisper."""
    waveform, sample_rate = torchaudio.load(audio_file)
    audio = waveform.squeeze().numpy()
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)
    return result.text

def initialize_llm():
    """Initializes the Gemini LLM using an API key."""
    with open('gemini.txt', 'r') as f:
        api_key = f.read().strip()
    return ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

def load_and_process_subtitles(directory="data"):
    """Loads subtitle files from a directory and processes them into a retriever."""
    loader = DirectoryLoader(directory, glob="*.srt", loader_cls=TextLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(texts, embeddings)
    return db

def query_llm(llm, query):
    """Queries the LLM for context and retrieves relevant information."""
    response = llm.invoke(query)
    return response

def main():
    st.title("Subtitle Retriever Box")
    st.write("Upload an audio file")
    
    whisper_model = load_whisper_model()
    llm = initialize_llm()
    retriever = load_and_process_subtitles()
    
    uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "wav", "m4a"])
    
    if uploaded_file is not None:
        with st.spinner("Transcribing audio..."):
            transcription = convert_audio_to_text(uploaded_file, whisper_model)
        st.text_area("Transcription:", transcription, height=150)
        
        with st.spinner("Searching subtitles..."):
            docs = retriever.similarity_search(transcription, k=3)
        
        st.write("### Retrieved Subtitles and Context")
        for doc in docs:
            movie_name = doc.metadata.get("source", "Unknown Movie")
            subtitle_text = doc.page_content
            
            llm_query = f"Extract key details from this subtitle: {subtitle_text}"
            context = query_llm(llm, llm_query)
            
            st.subheader(movie_name)
            st.write(f"**Timestamp:** {doc.metadata.get('timestamp', 'N/A')}")
            st.write(f"**Subtitle:** {subtitle_text}")
            st.write(f"**Context:** {context}")

if __name__ == "__main__":
    main()
