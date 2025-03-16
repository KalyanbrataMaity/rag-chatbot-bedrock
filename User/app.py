import boto3
import streamlit as st
import os 
import uuid

from langchain_aws.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load AWS Credentials
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET_NAME = os.getenv("BUCKET_NAME", "bedrock-chat-s3")
FOLDER_PATH = "/tmp/"

# Initialize AWS clients
s3_client = boto3.client("s3", region_name=AWS_REGION)
bedrock_client = boto3.client(service_name="bedrock-runtime",
                              region_name=AWS_REGION)

# Bedrock Embeddings
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",
                                       client=bedrock_client)

# Genrate unique request ID
def get_unique_id():
    return str(uuid.uuid4())

# Function to download FAISS index from S3
def load_index():
    st.write("⬇️ Loading vector database from S3...")
    try:
        s3_client.download_file(BUCKET_NAME, "my_faiss.faiss", f"{FOLDER_PATH}my_faiss.faiss")
        s3_client.download_file(BUCKET_NAME, "my_faiss.pkl", f"{FOLDER_PATH}my_faiss.pkl")

        st.success("✅ Vector database loaded successfully!")

    except Exception as e:
        st.error(f"⚠️ Error downloading FAISS index: {e}")

# Function to load FAISS index from local storage
def get_vector_store():
    try:
        return FAISS.load_local(
            index_name="my_faiss",
            folder_path=FOLDER_PATH,
            embeddings=bedrock_embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"⚠️ Error loading FAISS index: {e}")
        return None

# Function to initialize LLM
def get_llm():
    return Bedrock(
        model_id="anthropic.claude-v2:1",
        client=bedrock_client,
        model_kwargs={'max_tokens_to_sample': 512}
    )

# Function to retrieve answers using RAG
def get_response(llm, vectorestore, question):
    prompt_template="""
    Human: Please use the given context to provide a concise answer to the question.
    If you don't know the answer, hust say that you don't know. Do not make up an answer.
    <context>
    {context}
    </context>
    
    Question: {question}
    
    Assistant:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorestore.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    response = qa({"query": question})
    return response.get("result", "⚠️ No answer found.")


# Streamlit UI
def main():
    st.title("💬 Chat with PDF (Amazon Bedrock & RAG)")
    st.write("Ask questions based on the uploaded PDF document.")

    # Download the FAISS index to local
    load_index()

    # Verify loaded files
    dir_list = os.listdir(FOLDER_PATH)
    st.write(f"📁 Files in `{FOLDER_PATH}`: {dir_list}")

    # Load FAISS index from local
    vector_store = get_vector_store()

    if vector_store:
        st.success("✅ FAISS index is ready!")
    else:
        st.error("⚠️ Failed to load vector store. Check logs.")
        return
    
    # User input
    question = st.text_input("📝 Ask your question:")

    if st.button("Ask"):
        if question.strip():
            with st.spinner("🤖 Thinking..."):
                llm = get_llm()
                answer = get_response(llm, vector_store, question)
                st.write(f"**Answer:** {answer}")
                st.success("✅ Done!")
        else:
            st.warning("⚠️ Please enter a question before clicking 'Ask'.")

if __name__ == "__main__":
    main()