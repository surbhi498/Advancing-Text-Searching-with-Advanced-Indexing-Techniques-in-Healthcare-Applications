import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader,UnstructuredFileLoader
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.retrievers import BM25Retriever
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from glob import glob
from llama_index.vector_stores.qdrant import QdrantVectorStore
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import models, SentenceTransformer
from langchain.embeddings.base import Embeddings
from qdrant_client.models import VectorParams
import torch

# from llama_index import SimpleDirectoryReader, StorageContext
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

# Initialize our custom embeddings class
embeddings = ClinicalBertEmbeddings()    

# # Load ClinicalBERT
# tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
# model = AutoModel.from_pretrained("medicalai/ClinicalBERT")

# # Use mean pooling
# pooling_model = models.Pooling(model.config.hidden_size, pooling_mode_mean_tokens=True)

# # Create custom SentenceTransformer model
# # sentence_transformer_model = SentenceTransformer(modules=[model, pooling_model])
# # Use mean pooling
# sentence_transformer_model = SentenceTransformer(modules=[model])
# embeddings = SentenceTransformerEmbeddings(model=sentence_transformer_model)
print(embeddings)

# Tokenize sentences
# loading from directory
loader = DirectoryLoader("Data/", glob="**/*.pdf", show_progress=True, loader_cls=UnstructuredFileLoader)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
texts = text_splitter.split_documents(documents)

# Get elements

print("Create Vector Database")
url = "http://localhost:6333"
client = QdrantClient(url=url, prefer_grpc=False)
collection_name = "vector_db"

try:
# Check if the collection already exists
    collection_info = client.get_collection(collection_name=collection_name)
    print(f"Collection '{collection_name}' already exists.")
except Exception as e:
    print(f"Collection '{collection_name}' does not exist. Creating a new one.")
    print(f"Error: {e}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config= VectorParams(
            size=768,
            distance="Cosine"
        )
    )
    qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url=url,
    prefer_grpc=False,
    collection_name=collection_name,
    # optimizer_config=optimizer_config
    )
    keyword_retriever = BM25Retriever.from_documents(texts)
    keyword_retriever.k =  3
    print("vector database created.................")

