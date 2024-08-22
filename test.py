import io
import gradio as gr
import base64
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.retrievers import BM25Retriever
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings
from qdrant_client.models import VectorParams
import torch
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from huggingface_hub import hf_hub_download
from tempfile import NamedTemporaryFile
from langchain.retrievers import EnsembleRetriever
import os
import nltk
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.DEBUG)

# Define the path for NLTK data
nltk_data_path = '/tmp/nltk_data'
os.makedirs(nltk_data_path, exist_ok=True)

# Set NLTK data path environment variable
nltk.data.path.append(nltk_data_path)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

# Qdrant configuration (replace with your own credentials)
load_dotenv()

# Get Qdrant credentials from environment variables
qdrant_url = os.getenv('QDRANT_URL')
qdrant_api_key = os.getenv('QDRANT_API_KEY')


class ClinicalBertEmbeddings(Embeddings):
    def __init__(self, model_name: str = "medicalai/ClinicalBERT"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def embed(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt",
                                padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
        return embeddings.squeeze().numpy()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def embed_documents(self, texts):
        return [self.embed(text) for text in texts]

    def embed_query(self, text):
        return self.embed(text)


def load_model():
    model_name = "aaditya/OpenBioLLM-Llama3-8B-GGUF"
    model_file = "openbiollm-llama3-8b.Q5_K_M.gguf"
    model_path = hf_hub_download(
        model_name, filename=model_file, local_dir='./')
    return LlamaCpp(
        model_path=model_path,
        temperature=0.3,
        n_ctx=2048,
        top_p=1
    )


def load_embeddings():
    return ClinicalBertEmbeddings(model_name="medicalai/ClinicalBERT")


def setup_qdrant():
    try:
        if not qdrant_url or not qdrant_api_key:
            raise ValueError("QDRANT_URL or QDRANT_API_KEY not set.")

        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            port=443,
        )
        print("Qdrant client initialized successfully.")

        collection_name = "vector_db"
        try:
            client.get_collection(collection_name=collection_name)
            print(f"Collection '{collection_name}' already exists.")
        except ResponseHandlingException:
            print(
                f"Collection '{collection_name}' does not exist. Creating a new one.")
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=768, distance="Cosine")
            )
            print(f"Collection '{collection_name}' created successfully.")

        embeddings = load_embeddings()
        print("Embeddings model loaded successfully.")

        return Qdrant(client=client, embeddings=embeddings, collection_name=collection_name)

    except Exception as e:
        print(f"Failed to initialize Qdrant: {e}")
        return None


db = setup_qdrant()
llm = load_model()
embeddings = load_embeddings()

prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""
prompt = PromptTemplate(template=prompt_template,
                        input_variables=['context', 'question'])


def encode_pdf(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def display_pdf(file_path):
    base64_pdf = encode_pdf(file_path)
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    return pdf_display


def process_pdf(file):
    if file is None:
        return "Please upload a PDF file.", None

    try:
        # Convert the NamedString to bytes
        file_content = file.decode('utf-8').encode('latin-1')

        # Create a BytesIO object
        file_like_object = io.BytesIO(file_content)

        # Create a temporary file and write the content
        temp_file = NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_file.write(file_like_object.getvalue())
        temp_file_path = temp_file.name
        temp_file.close()

        # Now proceed with the rest of your function
        loader = PDFMinerLoader(temp_file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        try:
            db.add_documents(texts)
            print("PDF processed and vector database updated!")

            global ensemble_retriever
            bm25_retriever = BM25Retriever.from_documents(documents=texts)
            bm25_retriever.k = 3
            qdrant_retriever = db.as_retriever(search_kwargs={"k": 1})
            ensemble_retriever = EnsembleRetriever(
                retrievers=[qdrant_retriever, bm25_retriever],
                weights=[0.5, 0.5]
            )

            pdf_display = display_pdf(temp_file_path)
            return "PDF processed successfully. You can now ask questions about it.", pdf_display
        except Exception as e:
            return f"Error updating database: {e}", None

    except Exception as e:
        return f"Error processing PDF: {e}", None


def process_query(query):
    if not query:
        return "Please enter a question."

    try:
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
        return f"Answer: {answer}\n\nSource Document: {source_document}\n\nDocument Source: {doc}"
    except Exception as e:
        return f"Error processing query: {e}"

# Gradio interface


def gradio_interface(file, query):
    global pdf_display
    if file:
        result, pdf_display = process_pdf(file)
        if "successfully" not in result:
            return result, pdf_display if pdf_display else None

    if query:
        return process_query(query), pdf_display if pdf_display else None
    else:
        return "Please enter a question.", pdf_display if pdf_display else None


pdf_display = None

iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.File(label="Upload PDF"),
        gr.Textbox(label="Enter your question")
    ],
    outputs=[
        gr.Textbox(label="Response"),
        gr.HTML(label="PDF Preview")
    ],
    title="PDF Question Answering System",
    description="Upload a PDF and ask questions about its content."
)

if __name__ == "__main__":
    iface.launch()
