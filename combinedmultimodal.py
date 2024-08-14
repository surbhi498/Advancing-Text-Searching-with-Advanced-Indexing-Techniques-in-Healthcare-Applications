import os
import uuid
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
import qdrant_client
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
import clip
from llama_index.core import Document 
from langchain_community.llms import LlamaCpp
import numpy as np
from huggingface_hub import hf_hub_download
from langchain_community.llms import LlamaCpp
from llama_index.core import (
    ServiceContext,
    SimpleDirectoryReader,
)

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import Settings
import hashlib
import uuid
import os
import torch
import clip
import numpy as np
from llama_index.core.schema import ImageDocument
from llama_index.core.vector_stores import VectorStoreQuery
import cv2
import matplotlib.pyplot as plt
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from unstructured.partition.pdf import partition_pdf
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from PIL import Image 
import logging
import concurrent.futures
import logging
from llama_index.core import set_global_service_context

# Configure logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class MetadataMode:
    EMBED = "embed"
    INLINE = "inline"
    NONE = "none"
    
# Define the vectors configuration
vectors_config = {
    "vector_size": 512,  # or whatever the dimensionality of your vectors is
    "distance": "Cosine"  # can be "Cosine", "Euclidean", etc.
}
class ClinicalBertEmbeddingWrapper:
    def __init__(self, model_name: str = "medicalai/ClinicalBERT"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def embed(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
        return embeddings.squeeze().tolist() 

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def embed_documents(self, texts):
        return [self.embed(text) for text in texts]

    def embed_query(self, text):
        return self.embed(text)
     # Implement this method if needed
    def get_text_embedding_batch(self, text_batch, show_progress=False):
        embeddings = []
        num_batches = len(text_batch)
        
        # Process in batches of size 8
        batch_size = 8
        for i in tqdm(range(0, num_batches, batch_size), desc="Processing Batches", disable=not show_progress):
            batch_texts = text_batch[i:i + batch_size]
            batch_embeddings = self.embed_documents(batch_texts)
            embeddings.extend(batch_embeddings)
        
        return embeddings
    def get_agg_embedding_from_queries(self, queries):
        # Get embeddings for each query using the embed method
        embeddings = [torch.tensor(self.embed(query)) for query in queries]

        # Convert list of tensors to a single tensor for aggregation
        embeddings_tensor = torch.stack(embeddings)

        # Example: averaging embeddings
        agg_embedding = embeddings_tensor.mean(dim=0)

        return agg_embedding.tolist()

model_name = "aaditya/OpenBioLLM-Llama3-8B-GGUF"
model_file = "openbiollm-llama3-8b.Q5_K_M.gguf"

# Download model
model_path = hf_hub_download(model_name, filename=model_file, local_dir='./')

local_llm = "openbiollm-llama3-8b.Q5_K_M.gguf"
# Initialize ClinicalBert embeddings model


# text_embed_model = ClinicalBertEmbeddings(model_name="medicalai/ClinicalBERT")
text_embed_model = ClinicalBertEmbeddingWrapper(model_name="medicalai/ClinicalBERT")
llm1 = LlamaCpp(
        model_path=local_llm,
        temperature=0.3,
        n_ctx=2048,
        top_p=1
    )
Settings.llm = llm1
Settings.embed_model = text_embed_model
# Define ServiceContext with ClinicalBertEmbeddings for text
service_context = ServiceContext.from_defaults(
    llm = llm1,
    embed_model=text_embed_model  # Use ClinicalBert embeddings model
)
set_global_service_context(service_context)
# Log ServiceContext details
# logging.debug(f"LLM: {service_context.llm}")
# logging.debug(f"Embed Model: {service_context.embed_model}")
# logging.debug(f"Node Parser: {service_context.node_parser}")
# logging.debug(f"Prompt Helper: {service_context.prompt_helper}")
# Create QdrantClient with the location set to ":memory:", which means the vector db will be stored in memory
try:
    text_client = qdrant_client.QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        port=443, 
    )
    print("Qdrant client initialized successfully.")
except Exception as e:
    print(f"Error initializing Qdrant client: {e}")
    raise

# Create QdrantVectorStore using QdrantClient and the collection name "pdf_text"
try:
    text_vector_store = QdrantVectorStore(
        client=text_client, collection_name="pdf_text"
    )
    print("Qdrant vector store initialized successfully.")
except Exception as e:
    print(f"Error initializing Qdrant vector store: {e}")
    raise

try:
    image_vector_store = QdrantVectorStore(
        client=text_client, collection_name="pdf_img"
    )
    print("Qdrant vector store initialized successfully.")
except Exception as e:
    print(f"Error initializing Qdrant vector store: {e}")
    raise

storage_context = StorageContext.from_defaults(vector_store=text_vector_store)

# load Text documents from the data_wiki directory
text_documents = SimpleDirectoryReader("./Data").load_data()
# for i, doc in enumerate(text_documents):
#     print(f"Document {i+1} length: {len(doc.text)}")
#     print(f"Document {i+1} content (first 100 chars): {doc.text[:100]}")

# for i, doc in enumerate(text_documents):
#     embedding = service_context.embed_model.embed(doc.text)
#     print(f"Document {i+1} embedding: {embedding[:10]}")  # Print first 10 values

# sample_text = text_documents[0].text[:512]  # Take a sample from the first document
# embedding = text_embed_model.embed(sample_text)
# print(f"Sample embedding: {embedding[:10]}...")  # Print the first 10 values of the embedding

# create VectorStoreIndex using the text documents and StorageContext
wiki_text_index = VectorStoreIndex.from_documents(
    text_documents,
    storage_context=storage_context,
    service_context=service_context
)
print(f"VectorStoreIndex created with {len(wiki_text_index.docstore.docs)} documents")
# define the text query engine
text_query_engine = wiki_text_index.as_query_engine()
print(f"Text query engine type: {type(text_query_engine)}")

print(len(text_documents))

model, preprocess = clip.load("ViT-B/32")
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print(
    "Model parameters:",
    f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}",
)
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

pdf_directory = Path("./data")
image_path = Path("./images1")
image_path.mkdir(exist_ok=True, parents=True)

# Dictionary to store image metadata
image_metadata_dict = {}

# Limit the number of images downloaded per PDF
MAX_IMAGES_PER_PDF = 15

# Generate a UUID for each image
image_uuid = 0

# Iterate over each PDF file in the data folder
for pdf_file in pdf_directory.glob("*.pdf"):
    images_per_pdf = 0
    print(f"Processing: {pdf_file}")

    # Extract images from the PDF
    try:
        raw_pdf_elements = partition_pdf(
            filename=str(pdf_file),
            extract_images_in_pdf=True,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            extract_image_block_output_dir=image_path,
        )
        # Loop through the elements
    except Exception as e:
        print(f"Error processing {pdf_file}: {e}")
        import traceback
        traceback.print_exc()
        continue   
    # # Iterate through each file in the directory
for image_file in os.listdir(image_path):
    if image_file.endswith(('.jpg', '.jpeg', '.png')):
        # Generate a standard UUID for the image
        image_uuid = str(uuid.uuid4())
        image_file_name = image_file
        image_file_path = image_path / image_file

        # Construct metadata entry for the image
        image_metadata_dict[image_uuid] = {
            "filename": image_file_name,
            "img_path": str(image_file_path)  # Store the absolute path to the image
        }

        # Limit the number of images processed per folder
        if len(image_metadata_dict) >= MAX_IMAGES_PER_PDF:
            break   

print(f"Number of items in image_dict: {len(image_metadata_dict)}")

# Print the metadata dictionary
for key, value in image_metadata_dict.items():
    print(f"UUID: {key}, Metadata: {value}")


def plot_images_with_opencv(image_metadata_dict):
    original_images_urls = []
    images_shown = 0

    plt.figure(figsize=(16, 16))  # Adjust the figure size as needed

    for image_id in image_metadata_dict:
        img_path = image_metadata_dict[image_id]["img_path"]
        if os.path.isfile(img_path):
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    # Convert BGR (OpenCV) to RGB (matplotlib)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    plt.subplot(8, 8, len(original_images_urls) + 1)
                    plt.imshow(img_rgb)
                    plt.xticks([])
                    plt.yticks([])

                    original_images_urls.append(image_metadata_dict[image_id]["filename"])
                    images_shown += 1
                    if images_shown >= 64:
                        break
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

    plt.tight_layout()
    plt.show()

plot_images_with_opencv(image_metadata_dict)
# set the device to use for the CLIP model, either CUDA (GPU) or CPU, depending on availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model, preprocess = clip.load("ViT-B/32", device=device)
print(clip.available_models())

# Function to preprocess image using OpenCV
def preprocess_image(img):
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert the image to a PIL Image and then preprocess
    img_pil = Image.fromarray(img_rgb)
    return preprocess(img_pil)

img_emb_dict = {}
with torch.no_grad():
    for image_id in image_metadata_dict:
        img_file_path = image_metadata_dict[image_id]["img_path"]
        if os.path.isfile(img_file_path):
            try:
                # Load image using OpenCV
                img = cv2.imread(img_file_path)

                if img is not None:
                    # Preprocess image
                    image = preprocess_image(img).unsqueeze(0).to(device)

                    # Extract image features
                    image_features = model.encode_image(image)

                    # Store image features
                    img_emb_dict[image_id] = image_features
                else:
                    print(f"Failed to load image {img_file_path}")
            except Exception as e:
                print(f"Error processing image {img_file_path}: {e}")

len(img_emb_dict) #22 image so 22 img emb 


# create a list of ImageDocument objects, one for each image in the dataset
img_documents = []
for image_filename in image_metadata_dict:
    # the img_emb_dict dictionary contains the image embeddings
    if image_filename in img_emb_dict:
        filename = image_metadata_dict[image_filename]["filename"]
        filepath = image_metadata_dict[image_filename]["img_path"]
        #print(filepath)

        # create an ImageDocument for each image
        newImgDoc = ImageDocument(
            text=filename, metadata={"filepath": filepath}
        )

        # set image embedding on the ImageDocument
        newImgDoc.embedding = img_emb_dict[image_filename].tolist()[0]
        img_documents.append(newImgDoc)
        
# define storage context
storage_context = StorageContext.from_defaults(vector_store=image_vector_store)

# define image index
image_index = VectorStoreIndex.from_documents(
    img_documents,
    storage_context=storage_context
)
# for doc in img_documents:
#     print(f"ImageDocument: {doc.text}, Embedding: {doc.embedding}, Metadata: {doc.metadata}")

def retrieve_results_from_image_index(query):
    """ take a text query as input and return the most similar image from the vector store """

    # first tokenize the text query and convert it to a tensor
    text = clip.tokenize(query).to(device)

    # encode the text tensor using the CLIP model to produce a query embedding
    query_embedding = model.encode_text(text).tolist()[0]

    # create a VectorStoreQuery
    image_vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=1, # returns 1 image
        mode="default",
    )

    # execute the query against the image vector store
    image_retrieval_results = image_vector_store.query(
        image_vector_store_query
    )
    return image_retrieval_results

def plot_image_retrieve_results(image_retrieval_results):
    """ Take a list of image retrieval results and create a new figure """

    plt.figure(figsize=(16, 5))

    img_cnt = 0

    # Iterate over the image retrieval results, and for each result, display the corresponding image and its score in a subplot.
    # The title of the subplot is the score of the image, formatted to four decimal places.

    for returned_image, score in zip(
        image_retrieval_results.nodes, image_retrieval_results.similarities
    ):
        img_name = returned_image.text
        img_path = returned_image.metadata["filepath"]

        # Read image using OpenCV
        image = cv2.imread(img_path)
        # Convert image to RGB format (OpenCV reads in BGR by default)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.subplot(2, 3, img_cnt + 1)
        plt.title("{:.4f}".format(score))

        plt.imshow(image_rgb)
        plt.xticks([])
        plt.yticks([])
        img_cnt += 1

    plt.tight_layout()
    plt.show()
def image_query(query):
    image_retrieval_results = retrieve_results_from_image_index(query)
    plot_image_retrieve_results(image_retrieval_results) 

query1 = "What is gingivitis?"
# generate image retrieval results
image_query(query1)

# generate text retrieval results
text_retrieval_results = text_query_engine.query(query1)
print(f"Type of text retrieval results: {type(text_retrieval_results)}")
print(f"Content of text retrieval results: {text_retrieval_results.response}")
print("Text retrieval results: \n" + str(text_retrieval_results.response)) 


# # import os
# # import uuid
# # from llama_index.vector_stores.qdrant import QdrantVectorStore
# # from llama_index.core import VectorStoreIndex, StorageContext
# # import qdrant_client
# # import torch
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # import clip
# # from llama_index.core import Document 
# # from langchain_community.llms import LlamaCpp
# # import numpy as np
# # from huggingface_hub import hf_hub_download
# # from tqdm import tqdm
# # from transformers import AutoTokenizer, AutoModel
# # from langchain.embeddings.base import Embeddings
# # from llama_index.embeddings.langchain import LangchainEmbedding
# # from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# # from llama_index.core import Settings
# # import hashlib
# # import cv2
# # import matplotlib.pyplot as plt
# # from unstructured.partition.pdf import partition_pdf
# # from pathlib import Path
# # from PIL import Image
# # import logging

# # Configure logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# # Define MetadataMode class (can be adjusted as per your need)
# class MetadataMode:
#     EMBED = "embed"
#     INLINE = "inline"
#     NONE = "none"

# # ClinicalBert embedding wrapper class
# class ClinicalBertEmbeddingWrapper:
#     def __init__(self, model_name: str = "medicalai/ClinicalBERT"):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModel.from_pretrained(model_name)
#         self.model.eval()

#     def embed(self, text: str):
#         inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#         embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
#         return embeddings.squeeze().tolist()

#     def mean_pooling(self, model_output, attention_mask):
#         token_embeddings = model_output[0]
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#         return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
#     def embed_documents(self, texts):
#         return [self.embed(text) for text in texts]

#     def embed_query(self, text):
#         return self.embed(text)

#     def get_text_embedding_batch(self, text_batch, show_progress=False):
#         embeddings = []
#         num_batches = len(text_batch)
#         batch_size = 8
#         for i in tqdm(range(0, num_batches, batch_size), desc="Processing Batches", disable=not show_progress):
#             batch_texts = text_batch[i:i + batch_size]
#             batch_embeddings = self.embed_documents(batch_texts)
#             embeddings.extend(batch_embeddings)
#         return embeddings

#     def get_agg_embedding_from_queries(self, queries):
#         embeddings = [torch.tensor(self.embed(query)) for query in queries]
#         embeddings_tensor = torch.stack(embeddings)
#         agg_embedding = embeddings_tensor.mean(dim=0)
#         return agg_embedding.tolist()

# # Set up the Qdrant client
# QDRANT_URL = "https://f1e9a70a-afb9-498d-b66d-cb248e0d5557.us-east4-0.gcp.cloud.qdrant.io:6333"
# QDRANT_API_KEY = "REXlX_PeDvCoXeS9uKCzC--e3-LQV0lw3_jBTdcLZ7P5_F6EOdwklA"

# # Download the model
# model_name = "aaditya/OpenBioLLM-Llama3-8B-GGUF"
# model_file = "openbiollm-llama3-8b.Q5_K_M.gguf"
# model_path = hf_hub_download(model_name, filename=model_file, local_dir='./')

# local_llm = "openbiollm-llama3-8b.Q5_K_M.gguf"

# # Initialize the embedding model
# text_embed_model = ClinicalBertEmbeddingWrapper(model_name="medicalai/ClinicalBERT")
# Settings.embed_model = text_embed_model

# # Define ServiceContext with ClinicalBertEmbeddings for text
# service_context = ServiceContext.from_defaults(
#     llm=LlamaCpp(
#         model_path=local_llm,
#         temperature=0.3,
#         n_ctx=2048,
#         top_p=1
#     ),
#     embed_model=text_embed_model,
# )


# set_global_service_context(service_context)
# # Initialize Qdrant client and vector stores
# try:
#     text_client = qdrant_client.QdrantClient(
#         url=QDRANT_URL,
#         api_key=QDRANT_API_KEY,
#         port=443,
#     )
#     logging.info("Qdrant client initialized successfully.")
# except Exception as e:
#     logging.error(f"Error initializing Qdrant client: {e}")
#     raise

# try:
#     text_vector_store = QdrantVectorStore(
#         client=text_client, collection_name="pdf_text"
#     )
#     logging.info("Qdrant vector store for text initialized successfully.")
# except Exception as e:
#     logging.error(f"Error initializing Qdrant vector store for text: {e}")
#     raise

# try:
#     image_vector_store = QdrantVectorStore(
#         client=text_client, collection_name="pdf_img"
#     )
#     logging.info("Qdrant vector store for images initialized successfully.")
# except Exception as e:
#     logging.error(f"Error initializing Qdrant vector store for images: {e}")
#     raise

# storage_context = StorageContext.from_defaults(vector_store=text_vector_store)

# # Load Text documents
# try:
#     text_documents = SimpleDirectoryReader("./Data").load_data()
#     logging.info(f"Loaded {len(text_documents)} text documents.")
# except Exception as e:
#     logging.error(f"Error loading text documents: {e}")
#     raise

# # Create VectorStoreIndex using the text documents
# try:
#     wiki_text_index = VectorStoreIndex.from_documents(
#         text_documents,
#         storage_context=storage_context,
#         service_context=service_context,
#     )
#     logging.info(f"VectorStoreIndex created with {len(wiki_text_index.docstore.docs)} documents")
# except Exception as e:
#     logging.error(f"Error creating VectorStoreIndex: {e}")
#     raise

# # Define the text query engine
# text_query_engine = wiki_text_index.as_query_engine()

# # Load and process images
# pdf_directory = Path("./data")
# image_path = Path("./images1")
# image_path.mkdir(exist_ok=True, parents=True)

# image_metadata_dict = {}
# MAX_IMAGES_PER_PDF = 15
# image_uuid = 0

# for pdf_file in pdf_directory.glob("*.pdf"):
#     images_per_pdf = 0
#     logging.info(f"Processing: {pdf_file}")

#     try:
#         raw_pdf_elements = partition_pdf(
#             filename=str(pdf_file),
#             extract_images_in_pdf=True,
#             infer_table_structure=True,
#             chunking_strategy="by_title",
#             max_characters=4000,
#             new_after_n_chars=3800,
#             combine_text_under_n_chars=2000,
#             extract_image_block_output_dir=image_path,
#         )
#     except Exception as e:
#         logging.error(f"Error processing {pdf_file}: {e}")
#         continue

# for image_file in os.listdir(image_path):
#     if image_file.endswith(('.jpg', '.jpeg', '.png')):
#         image_uuid = str(uuid.uuid4())
#         image_file_name = image_file
#         image_file_path = image_path / image_file

#         image_metadata_dict[image_uuid] = {
#             "filename": image_file_name,
#             "img_path": str(image_file_path)
#         }

#         if len(image_metadata_dict) >= MAX_IMAGES_PER_PDF:
#             break

# logging.info(f"Number of items in image_dict: {len(image_metadata_dict)}")

# # Display images using OpenCV
# def plot_images_with_opencv(image_metadata_dict):
#     original_images_urls = []
#     images_shown = 0

#     plt.figure(figsize=(16, 16))

#     for image_id in image_metadata_dict:
#         img_path = image_metadata_dict[image_id]["img_path"]
#         if os.path.isfile(img_path):
#             try:
#                 img = cv2.imread(img_path)
#                 if img is not None:
#                     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                     plt.subplot(8, 8, len(original_images_urls) + 1)
#                     plt.imshow(img_rgb)
#                     plt.xticks([])
#                     plt.yticks([])
#                     original_images_urls.append(image_metadata_dict[image_id]["filename"])
#                     images_shown += 1
#                     if images_shown >= 64:
#                         break
#             except Exception as e:
#                 logging.error(f"Error processing image {img_path}: {e}")

#     plt.tight_layout()
#     plt.show()

# plot_images_with_opencv(image_metadata_dict)

# # Function to preprocess image using OpenCV
# def preprocess_image(img):
#     # Convert BGR to RGB
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # Convert the image to a PIL Image and then preprocess
#     img_pil = Image.fromarray(img_rgb)
#     return preprocess(img_pil)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

# logging.info(f"Loaded CLIP model with device: {device}")

# # Preprocess and encode images
# img_emb_dict = {}
# with torch.no_grad():
#     for image_id in image_metadata_dict:
#         img_file_path = image_metadata_dict[image_id]["img_path"]
#         if os.path.isfile(img_file_path):
#             try:
#                 img = cv2.imread(img_file_path)
#                 if img is not None:
#                     image = preprocess_image(img).unsqueeze(0).to(device)
#                     image_features = model.encode_image(image)
#                     img_emb_dict[image_id] = image_features
#                 else:logging.warning(f"Failed to load image {img_file_path}")
#             except Exception as e:
#                 logging.error(f"Error processing image {img_file_path}: {e}")

# logging.info(f"Number of image embeddings generated: {len(img_emb_dict)}")

# # Create ImageDocument objects for each image in the dataset
# img_documents = []
# for image_filename in image_metadata_dict:
#     if image_filename in img_emb_dict:
#         filename = image_metadata_dict[image_filename]["filename"]
#         filepath = image_metadata_dict[image_filename]["img_path"]

#         newImgDoc = ImageDocument(
#             text=filename, metadata={"filepath": filepath}
#         )

#         # Set image embedding on the ImageDocument
#         newImgDoc.embedding = img_emb_dict[image_filename].tolist()[0]
#         img_documents.append(newImgDoc)

# logging.info(f"Number of ImageDocument objects created: {len(img_documents)}")

# # Define storage context for images
# storage_context = StorageContext.from_defaults(vector_store=image_vector_store)

# # Create image index
# try:
#     image_index = VectorStoreIndex.from_documents(
#         img_documents,
#         storage_context=storage_context
#     )
#     logging.info(f"Image index created with {len(image_index.docstore.docs)} documents.")
# except Exception as e:
#     logging.error(f"Error creating Image Index: {e}")
#     raise

# # Function to retrieve results from the image index
# def retrieve_results_from_image_index(query):
#     text = clip.tokenize(query).to(device)
#     query_embedding = model.encode_text(text).tolist()[0]

#     image_vector_store_query = VectorStoreQuery(
#         query_embedding=query_embedding,
#         similarity_top_k=1,  # returns the top 1 image
#         mode="default",
#     )

#     image_retrieval_results = image_vector_store.query(
#         image_vector_store_query
#     )
#     return image_retrieval_results

# # Function to plot image retrieval results
# def plot_image_retrieve_results(image_retrieval_results):
#     plt.figure(figsize=(16, 5))

#     img_cnt = 0

#     for returned_image, score in zip(
#         image_retrieval_results.nodes, image_retrieval_results.similarities
#     ):
#         img_name = returned_image.text
#         img_path = returned_image.metadata["filepath"]

#         image = cv2.imread(img_path)
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         plt.subplot(2, 3, img_cnt + 1)
#         plt.title("{:.4f}".format(score))
#         plt.imshow(image_rgb)
#         plt.xticks([])
#         plt.yticks([])
#         img_cnt += 1

#     plt.tight_layout()
#     plt.show()

# # Function to query and retrieve images based on text
# def image_query(query):
#     try:
#         image_retrieval_results = retrieve_results_from_image_index(query)
#         plot_image_retrieve_results(image_retrieval_results)
#         return image_retrieval_results
#     except Exception as e:
#         logging.error(f"Error during image retrieval: {e}")
#         return None
# def text_retrieval(query):
#     try:
#         text_retrieval_results = text_query_engine.query(query)
#         logging.info(f"Type of text retrieval results: {type(text_retrieval_results)}")
#         logging.info(f"Content of text retrieval results: {text_retrieval_results.response}")
#         return text_retrieval_results.response
#     except Exception as e:
#         logging.error(f"Error during text retrieval: {e}")
#         return None    
# def parallel_query(query):
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         text_future = executor.submit(text_retrieval, query)
#         image_future = executor.submit(image_query, query)

#         text_result = text_future.result()
#         image_result = image_future.result()

#     return text_result, image_result

# # Query and retrieve text and image results in parallel
# query1 = "What is gingivitis?"
# text_result, image_result = parallel_query(query1)

# # Display the text results
# if text_result:
#     print("Text retrieval results: \n" + str(text_result))

# # # Query and retrieve images based on the text query
# # query1 = "What is gingivitis?"
# # image_query(query1)

# # # Query and retrieve text results
# # try:
# #     text_retrieval_results = text_query_engine.query(query1)
# #     logging.info(f"Type of text retrieval results: {type(text_retrieval_results)}")
# #     logging.info(f"Content of text retrieval results: {text_retrieval_results.response}")
# #     print("Text retrieval results: \n" + str(text_retrieval_results.response))
# # except Exception as e:
# #     logging.error(f"Error during text retrieval: {e}")