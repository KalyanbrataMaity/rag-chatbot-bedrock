import boto3
import streamlit as st
import os 
import uuid


# S3 client
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# Bedrock
from langchain_community.embeddings import BedrockEmbeddings

# Pdf Loader
from langchain_community.document_loaders import PyPDFLoader

# Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# FAISS
from langchain_community.vectorstores import FAISS


bedrock_client = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",
                                       client=bedrock_client)

def get_unique_id():
    return str(uuid.uuid4())



def main():
    st.header("This is Client site for chat with pdf demo using Bedrock, RAG etc.")
    
if __name__ == "__main__":
    main()