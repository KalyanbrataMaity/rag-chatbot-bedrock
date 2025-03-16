import boto3
import streamlit as st
import os 
import uuid

from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# Load AWS Credentials
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET_NAME = os.getenv("BUCKET_NAME", "bedrock-chat-s3")

# Initialize S3 client
s3_client = boto3.client("s3", region_name=AWS_REGION)

# Initialize bedrock client
bedrock_client = boto3.client(service_name="bedrock-runtime",
                              region_name=AWS_REGION)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",
                                       client=bedrock_client)

# Genrate unique request ID
def get_unique_id():
    return str(uuid.uuid4())

# Split PDF into chunks
def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

# Create and upload FAISS vector store
def create_vector_store(request_id, documents):
    vectorstore_faiss = FAISS.from_documents(documents, bedrock_embeddings)
    folder_path = "/tmp/"
    file_name = f"{request_id}.bin"

    # Save vector store
    vectorstore_faiss.save_local(index_name=file_name,
                                 folder_path=folder_path)
    
    # upload to s3
    s3_client.upload_file(Filename=f"{folder_path}/{file_name}.faiss", Bucket=BUCKET_NAME, Key="my_faiss.faiss")
    s3_client.upload_file(Filename=f"{folder_path}/{file_name}.pkl", Bucket=BUCKET_NAME, Key="my_faiss.pkl")
    
    return True


# Streamlit UI
def main():
    st.title("Chat with PDF - Admin Dashboard")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        request_id = get_unique_id()
        st.write(f"Request ID: {request_id}")

        # Save the uploaded file locally
        saved_filename = f"{request_id}.pdf"
        with open(saved_filename, mode="wb") as w:
            w.write(uploaded_file.getvalue())

        # Load and split the PDF
        loader = PyPDFLoader(saved_filename)
        pages = loader.load_and_split()
        st.write(f"Total Pages: {len(pages)}")

        # Split text
        splitted_docs = split_text(pages, chunk_size=1000, chunk_overlap=200)
        st.write(f"Splitted Docs: {len(splitted_docs)}")

        # Create Vector Store
        st.write("Creating the Vector Store...")
        result = create_vector_store(request_id=request_id,
                                     documents=splitted_docs)
        if result:
            st.success("PDF processed and uploaded successfully!")
        else:
            st.error("Error processing PDF. Please check logs.")


if __name__ == "__main__":
    main()