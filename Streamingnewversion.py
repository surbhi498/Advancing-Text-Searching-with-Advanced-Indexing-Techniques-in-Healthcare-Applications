import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader, PDFMinerLoader
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.retrievers import BM25Retriever
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from glob import glob
from llama_index.vector_stores.qdrant import QdrantVectorStore
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import models, SentenceTransformer
from langchain.embeddings.base import Embeddings
from qdrant_client.models import VectorParams
import torch
import base64
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from huggingface_hub import hf_hub_download
from tempfile import NamedTemporaryFile
from langchain.retrievers import EnsembleRetriever
import urllib

# Add this at the beginning of your script
import logging
logging.basicConfig(level=logging.DEBUG)

# Set page configuration
st.set_page_config(layout="wide")
st.markdown("""
    <meta http-equiv="Content-Security-Policy" 
    content="default-src 'self'; object-src 'self'; frame-src 'self' data:; 
    script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';">
""", unsafe_allow_html=True)
# Streamlit secrets
qdrant_url = st.secrets["QDRANT_URL"]
qdrant_api_key = st.secrets["QDRANT_API_KEY"]

# For debugging only - remove or comment out these lines after verification
#st.write(f"QDRANT_URL: {qdrant_url}")
#st.write(f"QDRANT_API_KEY: {qdrant_api_key}")

class ClinicalBertEmbeddings(Embeddings):
    def __init__(self, model_name: str = "medicalai/ClinicalBERT"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def embed(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
        return embeddings.squeeze().numpy()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def embed_documents(self, texts):
        return [self.embed(text) for text in texts]

    def embed_query(self, text):
        return self.embed(text)

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
        if not qdrant_url or not qdrant_api_key:
            raise ValueError("QDRANT_URL or QDRANT_API_KEY not set in environment variables.")

        # Initialize Qdrant client
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            port=443,  # Assuming HTTPS should use port 443
        )
        st.write("Qdrant client initialized successfully.")

        # Create or recreate collection
        collection_name = "vector_db"
        try:
            collection_info = client.get_collection(collection_name=collection_name)
            st.write(f"Collection '{collection_name}' already exists.")
        except ResponseHandlingException:
            st.write(f"Collection '{collection_name}' does not exist. Creating a new one.")
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=768, distance="Cosine")
            )
            st.write(f"Collection '{collection_name}' created successfully.")

        embeddings = load_embeddings()
        st.write("Embeddings model loaded successfully.")

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

# Load models
llm = load_model()
embeddings = load_embeddings()

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

# Define Streamlit app

def process_answer(query):
    chain_type_kwargs = {"prompt": prompt}
    global ensemble_retriever
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
    
    # Displaying File
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
            global ensemble_retriever
            # Initialize retriever after documents are added
            bm25_retriever  = BM25Retriever.from_documents(documents=texts)
            bm25_retriever.k = 3
            qdrant_retriever = db.as_retriever(search_kwargs={"k":1})
            # Combine both retrievers using EnsembleRetriever
            ensemble_retriever = EnsembleRetriever(
            retrievers=[qdrant_retriever, bm25_retriever],
            weights=[0.5, 0.5]  # Adjust weights based on desired contribution
            )

        except Exception as e:
            st.error(f"Error updating database: {e}")

        st.subheader("Ask a question about the PDF")
        user_input = st.text_input("Your question:")

        if st.button('Get Response'):
            if user_input:
                try:
                    answer, source_document, doc = process_answer(user_input)
                    st.write("*Answer:*", answer)
                    st.write("*Source Document:*", source_document)
                    st.write("*Document Source:*", doc)
                except Exception as e:
                    st.error(f"Error processing query: {e}")
            else:
                st.warning("Please enter a query.")

if __name__ == "__main__":
    main()