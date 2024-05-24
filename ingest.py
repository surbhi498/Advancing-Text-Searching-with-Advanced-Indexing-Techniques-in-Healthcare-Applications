import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader,UnstructuredFileLoader
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings


# tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
# model = AutoModel.from_pretrained("medicalai/ClinicalBERT")
# # Mean Pooling - Take attention mask into account for correct averaging
# def meanpooling(output, mask):
#     embeddings = output[0] # First element of model_output contains all token embeddings
#     mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
#     return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
embeddings = SentenceTransformerEmbeddings(model_name="medicalai/ClinicalBERT")
print(embeddings)

# Tokenize sentences
# loading from directory
loader = DirectoryLoader("Data/", glob="**/*.pdf", show_progress=True, loader_cls=UnstructuredFileLoader)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
texts = text_splitter.split_documents(documents)
print("Create Vector Database")
url = "http://localhost:6333"
qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url=url,
    prefer_grpc=False,
    collection_name="vector_db"
)
print("vector database created.................")