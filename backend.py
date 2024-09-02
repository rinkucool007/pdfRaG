# backend.py

import os
import boto3
import json
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS

# 1. Load PDFs from a directory
def load_pdfs_from_directory(directory: str):
    documents = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(directory, file_name))
            documents.extend(loader.load())
    return documents

# 2. Split text recursively by characters, tokens, etc.
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)

# 3. Create embeddings using Bedrock client connection
def create_embeddings():
    return BedrockEmbeddings()

# 4. Create Vector DB, store embeddings, and create index using FAISS
def create_vector_db(documents, embeddings):
    # Create embeddings for the documents
    document_texts = [doc.page_content for doc in documents]
    doc_embeddings = embeddings.embed_documents(document_texts)
    
    # Create a FAISS index
    vector_db = FAISS(embeddings=doc_embeddings, documents=documents)
    return vector_db

# 5. Build index for all PDFs in the specified folder
def build_index(folder_path):
    documents = load_pdfs_from_directory(folder_path)
    split_docs = split_documents(documents)
    embeddings = create_embeddings()
    vector_db = create_vector_db(split_docs, embeddings)
    return vector_db

# 6a. Connect to Bedrock using boto3
def connect_to_bedrock():
    # Initialize the boto3 client for Bedrock
    client = boto3.client('bedrock-runtime', region_name='us-west-2')
    return client

# 6b. Query Bedrock Claude Foundation Model
def query_bedrock(client, model_id, input_text):
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps({"input": input_text}),
        contentType="application/json"
    )
    
    result = json.loads(response['body'].read().decode())
    return result['output']

# 6c. Search the vector DB and query the LLM with best matches
def query_llm(vector_db, query):
    results = vector_db.similarity_search(query)
    context = " ".join([result.page_content for result in results])
    
    # Combine the context with the user query
    full_input = f"Context: {context}\n\nQuestion: {query}"
    
    # Connect to Bedrock and query the model
    client = connect_to_bedrock()
    model_id = "anthropic.claude-v1"  # Example model ID, adjust based on actual model available
    response = query_bedrock(client, model_id, full_input)
    
    return response
