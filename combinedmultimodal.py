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
import threading
from dotenv import load_dotenv
from llama_index.llms.nvidia import NVIDIA
from open_clip import create_model_from_pretrained, get_tokenizer
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.query_engine import RetrieverQueryEngine
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import Settings
from transformers import AutoProcessor, AutoModel
import hashlib
import uuid
import os
import gradio as gr
import torch
import clip
import open_clip
import numpy as np
from llama_index.core.schema import ImageDocument
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
from llama_index.core import Document as LlamaIndexDocument
import getpass
import os
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from sentence_transformers import util
from transformers import AutoTokenizer, AutoModelForCausalLM
import base64
from google.generativeai import GenerativeModel, configure
import google.generativeai as genai

# Configure logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class MetadataMode:
    EMBED = "embed"
    INLINE = "inline"
    NONE = "none"
    
# Define the vectors configuration
vectors_config = {
    "vector_size": 768,  # or whatever the dimensionality of your vectors is
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
    

# Load environment variables
load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
nvidia_api_key = os.getenv("NVIDIA_API_KEY")
if not nvidia_api_key:
    raise ValueError("NVIDIA_API_KEY not found in .env file")

os.environ["NVIDIA_API_KEY"] = nvidia_api_key

model_name = "aaditya/OpenBioLLM-Llama3-8B-GGUF"
model_file = "openbiollm-llama3-8b.Q5_K_M.gguf"
QDRANT_URL = "https://f1e9a70a-afb9-498d-b66d-cb248e0d5557.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "REXlX_PeDvCoXeS9uKCzC--e3-LQV0lw3_jBTdcLZ7P5_F6EOdwklA"

# Download model
model_path = hf_hub_download(model_name, filename=model_file, local_dir='./')
llm = NVIDIA(model="writer/palmyra-med-70b")
llm.model
local_llm = "openbiollm-llama3-8b.Q5_K_M.gguf"
# Initialize ClinicalBert embeddings model
# text_embed_model = ClinicalBertEmbeddings(model_name="medicalai/ClinicalBERT")
text_embed_model = ClinicalBertEmbeddingWrapper(model_name="medicalai/ClinicalBERT")
# Intially I was using this biollm but for faster text response during inference I am going for external models
#but with this also it works fine.
llm1 = LlamaCpp(
        model_path=local_llm,
        temperature=0.3,
        n_ctx=2048,
        top_p=1
    )
Settings.llm = llm
Settings.embed_model = text_embed_model
# Define ServiceContext with ClinicalBertEmbeddings for text
service_context = ServiceContext.from_defaults(
    llm = llm,
    embed_model=text_embed_model  # Use ClinicalBert embeddings model
)
set_global_service_context(service_context)
# Just for logging and Debugging
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
# load Text documents from the data_wiki directory
# text_documents = SimpleDirectoryReader("./Data").load_data()
# Load documents
loader = DirectoryLoader("./Data/", glob="**/*.pdf", show_progress=True, loader_cls=UnstructuredFileLoader)
documents = loader.load()
# Print document names
for doc in documents:
    print(f"Processing document: {doc.metadata.get('source', 'Unknown')}")
# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
texts = text_splitter.split_documents(documents)



print(f"Loaded {len(documents)} documents")
print(f"Split into {len(texts)} chunks")
# Convert langchain documents to llama_index documents
text_documents = [
    LlamaIndexDocument(text=t.page_content, metadata=t.metadata)
    for t in texts
]
# Initialize Qdrant vector store
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

wiki_text_index = VectorStoreIndex.from_documents(text_documents
    # , storage_context=storage_context
    , service_context=service_context
    )
print(f"VectorStoreIndex created with {len(wiki_text_index.docstore.docs)} documents")

# define the streaming query engine
streaming_qe = wiki_text_index.as_query_engine(streaming=True)
print(len(text_documents))

# Function to query the text vector database
# Modify the process_query function

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
# Function to summarize images
def summarize_image(image_path):
    # Load and encode the image
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    # Create a GenerativeModel object
    model = GenerativeModel('gemini-1.5-flash')

    # Prepare the prompt
    prompt = """
    You are an expert in analyzing medical images. Please provide a detailed description of this medical image, including:
    1. You are a bot that is good at analyzing images related to Dog's health
    2. The body part or area being examined
    3. Any visible structures, organs, or tissues
    4. Any abnormalities, lesions, or notable features
    5. Any other relevant medical diagram description.

    Please be as specific and detailed as possible in your analysis.
    """

    # Generate the response
    response = model.generate_content([
        prompt,
        {"mime_type": "image/jpeg", "data": encoded_image}
    ])

    return response.text
   
# # Iterate through each file in the directory
for image_file in os.listdir(image_path):
    if image_file.endswith(('.jpg', '.jpeg', '.png')):
        # Generate a standard UUID for the image
        image_uuid = str(uuid.uuid4())
        image_file_name = image_file
        image_file_path = image_path / image_file
        # Generate image summary
        # image_summary = generate_image_summary_with(str(image_file_path), model, feature_extractor, tokenizer, device)
        # image_summary = generate_summary_with_lm(str(image_file_path), preprocess, model, device, tokenizer, lm_model)
        image_summary = summarize_image(image_file_path)
        # Construct metadata entry for the image
        image_metadata_dict[image_uuid] = {
            "filename": image_file_name,
            "img_path": str(image_file_path), # Store the absolute path to the image
            "summary": image_summary  # Add the summary to the metadata
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
# Function to preprocess image using OpenCV
def preprocess_image(img):
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert the image to a PIL Image and then preprocess
    img_pil = Image.fromarray(img_rgb)
    return preprocess(img_pil)
    # Use BiomedCLIP processor for preprocessing
    # return preprocess(images=img_pil, return_tensors="pt")
    # return preprocess(img_pil).unsqueeze(0)
    

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
                    # image = preprocess_image(img).to(device)

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
        summary = image_metadata_dict[image_filename]["summary"]
        #print(filepath)

        # create an ImageDocument for each image
        newImgDoc = ImageDocument(
            text=filename, metadata={"filepath": filepath, "summary": summary}  # Include the summary in the metadata
            
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
    # Encode the query using ClinicalBERT for text similarity
    clinical_query_embedding = text_embed_model.embed_query(query)
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
    if image_retrieval_results.nodes:
        best_score = -1
        best_image = None

        for node, clip_score in zip(image_retrieval_results.nodes, image_retrieval_results.similarities):
            image_path = node.metadata["filepath"]
            image_summary = node.metadata.get("summary", "")  # Assuming summaries are stored in metadata

            # Calculate text similarity between query and image summary
            summary_embedding = text_embed_model.embed_query(image_summary)
            # text_score = util.cosine_similarity(
            #     [clinical_query_embedding], [summary_embedding]
            # )[0][0]
            # Use util.cos_sim for cosine similarity
            text_score = util.cos_sim(torch.tensor([clinical_query_embedding]), 
                                      torch.tensor([summary_embedding]))[0][0].item()


            # Calculate average similarity score
            avg_score = (clip_score + text_score) / 2

            if avg_score > best_score:
                best_score = avg_score
                best_image = image_path

        return best_image, best_score

    return None, 0.0
  
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
def get_all_images():
    image_paths = []
    for _, metadata in image_metadata_dict.items():
        image_paths.append(metadata["img_path"])
    return image_paths

def load_image(image_path):
    return Image.open(image_path)
    
# Define the combined query function
def combined_query(query, similarity_threshold=0.3):
    # Text query
    text_response = streaming_qe.query(query)
    text_result = ""
    for text in text_response.response_gen:
        text_result += text

    # Image query
    top_image_path, similarity_score = retrieve_results_from_image_index(query)
    
    if similarity_score >= similarity_threshold:
        return text_result, top_image_path, similarity_score
    else:
        return text_result, None, similarity_score
def gradio_interface(query):
    text_result, image_path, similarity_score = combined_query(query)
    top_image = load_image(image_path) if image_path else None
    all_images = [load_image(path) for path in get_all_images()]
    return text_result, top_image, all_images, f"Similarity Score: {similarity_score:.4f}"  

with gr.Blocks() as iface:
    gr.Markdown("# Medical Knowledge Base Query System")
    
    with gr.Row():
        query_input = gr.Textbox(lines=2, placeholder="Enter your medical query here...")
        submit_button = gr.Button("Submit")
    
    with gr.Row():
        text_output = gr.Textbox(label="Text Response")
        image_output = gr.Image(label="Top Related Image (if similarity > threshold)")
    
    similarity_score_output = gr.Textbox(label="Similarity Score")
    
    gallery_output = gr.Gallery(label="All Extracted Images", show_label=True, elem_id="gallery")
    
    submit_button.click(
        fn=gradio_interface,
        inputs=query_input,
        outputs=[text_output, image_output, gallery_output, similarity_score_output]
    )

    # Load all images on startup
    iface.load(lambda: ["", None, [load_image(path) for path in get_all_images()], ""], 
               outputs=[text_output, image_output, gallery_output, similarity_score_output])  
# Launch the Gradio interface
iface.launch(share=True)
# just to check if it works or not
# def image_query(query):
#     image_retrieval_results = retrieve_results_from_image_index(query)
#     plot_image_retrieve_results(image_retrieval_results) 

# query1 = "What is gingivitis?"
# # generate image retrieval results
# image_query(query1)

# # Modify your text query function
# # def text_query(query):
# #     text_retrieval_results = process_query(query, text_embed_model, k=10)
# #     return text_retrieval_results
# # Function to query the text vector database


# def text_query(query: str, k: int = 10):
#     # Create a VectorStoreIndex from the existing vector store
#     index = VectorStoreIndex.from_vector_store(text_vector_store)
    
#     # Create a retriever with top-k configuration
#     retriever = index.as_retriever(similarity_top_k=k)
    
#     # Create a query engine
#     query_engine = RetrieverQueryEngine.from_args(retriever)
    
#     # Execute the query
#     response = query_engine.query(query)
    
#     return response

# # text_retrieval_results = text_query(query1)
# streaming_response = streaming_qe.query(
#     query1
# )
# streaming_response.print_response_stream()
