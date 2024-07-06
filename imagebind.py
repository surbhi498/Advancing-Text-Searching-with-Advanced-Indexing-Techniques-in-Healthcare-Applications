import torch
import clip
from qdrant_client import QdrantClient, models
import os
from llama_index.core.schema import ImageDocument
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
import matplotlib.pyplot as plt
from PIL import Image
from llama_index.core.vector_stores import VectorStoreQuery
import numpy as np
from pathlib import Path

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# Qdrant client setup
qdrant_client = QdrantClient("localhost", port=6333)
collection_name = "medical_img"

# Ensure the correct vector size based on CLIP model's output
clip_model = "ViT-B/32"
clip_dim = 512  # Default dimension for ViT-B/32

# Directory containing images
image_directory = Path("images")

# Initialize variables
image_metadata_dict = {}
image_uuid = 0
MAX_IMAGES_PER_FOLDER = 20  # Adjust as needed

# Iterate through each file in the directory
for image_file in os.listdir(image_directory):
    if image_file.endswith(('.jpg', '.jpeg', '.png')):
        image_uuid += 1
        image_file_name = image_file
        image_file_path = image_directory / image_file

        # Construct metadata entry for the image
        image_metadata_dict[image_uuid] = {
            "filename": image_file_name,
            "img_path": str(image_file_path)  # Store the absolute path to the image
        }

        # Limit the number of images processed per folder
        if image_uuid >= MAX_IMAGES_PER_FOLDER:
            break

# Print image_metadata_dict for verification
print(image_metadata_dict)

# Set device to CUDA if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load CLIP model
model, preprocess = clip.load(clip_model, device=device)
print(f"CLIP model {clip_model} loaded.")

# Verify and adjust VectorParams size based on CLIP model's output
vector_params = models.VectorParams(size=clip_dim, distance=models.Distance.COSINE)

# Recreate Qdrant collection if necessary
try:
    qdrant_client.get_collection(collection_name=collection_name)
except:
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=vector_params
    )
    print(f"Collection '{collection_name}' recreated successfully.")

# Dictionary to store image embeddings
img_emb_dict = {}
with torch.no_grad():
    for image_id in image_metadata_dict:
        img_file_path = image_metadata_dict[image_id]["img_path"]
        if os.path.isfile(img_file_path):
            image = preprocess(Image.open(img_file_path)).unsqueeze(0).to(device)
            image_features = model.encode_image(image)

            # Ensure the dimensionality matches Qdrant's expectations
            if image_features.shape[1] != clip_dim:
                raise ValueError(f"Expected CLIP embeddings of size {clip_dim}, got {image_features.shape[1]}")

            img_emb_dict[image_id] = image_features
            # Print the embedding for verification
            print(f"Embedding for image {image_id}: {image_features}")

print("Image embeddings computed for", len(img_emb_dict), "images.")

# Create ImageDocument objects
img_documents = []
for image_id, image_data in image_metadata_dict.items():
    if image_id in img_emb_dict:
        filename = image_data["filename"]
        filepath = image_data["img_path"]

        # Create ImageDocument with metadata and embedding
        newImgDoc = ImageDocument(
            text=filename,
            metadata={"filepath": filepath},
            embedding=img_emb_dict[image_id].squeeze().tolist()
        )
        img_documents.append(newImgDoc)

for doc in img_documents:
    if doc.embedding is None:
        print(f"Warning: Embedding is None for document {doc.text}")
    else:
        print(f"Embedding present for document {doc.text}")

# Create QdrantVectorStore with specified collection name
image_vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name)

# Define storage context
storage_context = StorageContext.from_defaults(vector_store=image_vector_store)

# Create VectorStoreIndex from ImageDocuments
image_index = VectorStoreIndex.from_documents(
    img_documents,
    storage_context=storage_context
)
doc_id = "883f101d-a373-4a42-b3ca-dca46559247a"  # Example document ID
document = qdrant_client.query(collection_name=collection_name, doc_id=doc_id)
# Print the document to inspect it
print(document)
if document:
    embedding = document["_node_content"]["embedding"]
    print(f"Embedding for document {doc_id}: {embedding}")


