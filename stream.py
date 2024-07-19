import streamlit as st
import base64
import json
import torch
from transformers import AutoTokenizer, AutoModel
from langchain import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PDFMinerLoader
from qdrant_client import QdrantClient
from langchain.retrievers import EnsembleRetriever
from streamlit_chat import message
from huggingface_hub import hf_hub_download
from tempfile import NamedTemporaryFile
from ingest import ClinicalBertEmbeddings
from qdrant_client.models import VectorParams, Distance



# Load environment variables from .env file

# Function to test Qdrant connection
# def test_qdrant_connection():
#     try:
#         # Initialize the client with the cluster URL and headers
#         headers = {
#             'api-key': 'REXlX_PeDvCoXeS9uKCzC--e3-LQV0lw3_jBTdcLZ7P5_F6EOdwklA'
#         }
#         client = QdrantClient(
#             url="https://f1e9a70a-afb9-498d-b66d-cb248e0d5557.us-east4-0.gcp.cloud.qdrant.io",
#             headers=headers,
#             prefer_grpc=False
#         )
#         print("Qdrant client connected successfully.")
        
#         # Check if the collection exists
#         collections = client.get_collections()
#         print("Collections:", collections)
        
#         # Create a collection if it doesn't exist
#         collection_name = "vector_db"
#         if collection_name not in collections:
#             client.create_collection(
#                 collection_name=collection_name,
#                 vector_size=768,  # Example vector size, adjust as needed
#                 distance="Cosine"  # Example distance metric
#             )
#             print(f"Collection '{collection_name}' created successfully.")
#         else:
#             print(f"Collection '{collection_name}' already exists.")
        
#     except Exception as e:
#         print(f"Error: {e}")

# Initialize model

qdrant_url = st.secrets["QDRANT_URL"]
qdrant_api_key = st.secrets["QDRANT_API_KEY"]

# For debugging only - remove or comment out these lines after verification
st.write(f"QDRANT_URL: {qdrant_url}")
st.write(f"QDRANT_API_KEY: {qdrant_api_key}")
@st.cache_resource
def load_model():
    model_name = "aaditya/OpenBioLLM-Llama3-8B-GGUF"
    model_file = "openbiollm-llama3-8b.Q5_K_M.gguf"
    model_path = hf_hub_download(model_name, filename=model_file, local_dir='./')
    return LlamaCpp(
        model_path=model_path,
        temperature=0.3,
        n_ctx=2048,
        top_p=1
    )

# Initialize embeddings
@st.cache_resource
def load_embeddings():
    return ClinicalBertEmbeddings(model_name="medicalai/ClinicalBERT")

# Initialize database
@st.cache_resource
def setup_qdrant():
    try:
        # Load environment variables
        qdrant_url = st.secrets["QDRANT_URL"],
        qdrant_api_key = st.secrets["QDRANT_API_KEY"]

        if not qdrant_url or not qdrant_api_key:
            raise ValueError("QDRANT_URL or QDRANT_API_KEY not set in environment variables.")

        # Initialize Qdrant client
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
        print("Qdrant client initialized successfully.")

        # Create or recreate collection
        collection_name = "vector_db"
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        print(f"Collection '{collection_name}' created successfully.")

        embeddings = load_embeddings()
        print("Embeddings loaded successfully.")

        return Qdrant(client=client, embeddings=embeddings, collection_name=collection_name)
    
    except Exception as e:
        st.error(f"Failed to initialize Qdrant: {e}")
        return None

# Initialize database
db = setup_qdrant()

if db is None:
    st.error("Qdrant setup failed, exiting.")
else:
    st.success("Qdrant setup successful.")

# Test Qdrant connection
# test_qdrant_connection()

# Load models
llm = load_model()
embeddings = load_embeddings()
db = setup_qdrant()

# Define prompt template
prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

# Define retriever
retriever = db.as_retriever(search_kwargs={"k":1})
ensemble_retriever = EnsembleRetriever(retrievers=[retriever], weights=[1.0])

# Define Streamlit app
st.set_page_config(layout="wide")

def process_answer(query):
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=ensemble_retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True
    )
    response = qa(query)
    answer = response['result']
    source_document = response['source_documents'][0].page_content
    doc = response['source_documents'][0].metadata['source']
    return answer, source_document, doc

def display_pdf(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def main():
    st.title("PDF Question Answering System")

    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

    if uploaded_file is not None:
        # Save uploaded PDF
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        
        # Display PDF
        st.subheader("PDF Preview")
        display_pdf(temp_file_path)
        
        # Load and process PDF
        loader = PDFMinerLoader(temp_file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        
        # Update the Qdrant database with the new PDF content
        try:
            db.add_documents(texts)
            st.success("PDF processed and vector database updated!")
        except Exception as e:
            st.error(f"Error updating database: {e}")

        st.subheader("Ask a question about the PDF")
        user_input = st.text_input("Your question:")

        if st.button('Get Response'):
            if user_input:
                try:
                    answer, source_document, doc = process_answer(user_input)
                    st.write("**Answer:**", answer)
                    st.write("**Source Document:**", source_document)
                    st.write("**Document Source:**", doc)
                except Exception as e:
                    st.error(f"Error processing query: {e}")
            else:
                st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
