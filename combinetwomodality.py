import threading
from ingest import ClinicalBertEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, VectorStruct, ScoredPoint
import numpy as np
from clip_helpers import encode_query_text_with_clip, clip_dim, device  # Import the CLIP helper functions and variables
import torch

# Qdrant client setup
qdrant_client = QdrantClient("localhost", port=6333)

# Collection names
text_collection_name = "vector_db"
image_collection_name = "medical_img"

# Function to query the text vector database
def query_text_vector_db(query_vector, text_results, k=10):
    text_search_result = qdrant_client.search(
        collection_name=text_collection_name,
        query_vector=query_vector,
        limit=k
    )
    text_results.extend(text_search_result)

# Function to query the image vector database
# Function to query the image vector database
def query_image_vector_db(query_vector, image_results, k=10):
    try:
        # Assuming qdrant_client.search requires a specific structure
        named_vector = VectorStruct(vector=query_vector.tolist()[0],  # Convert to list if necessary
                                    metadata={"key": "value"})  # Example metadata, adjust as needed
        image_search_result = qdrant_client.search(
            collection_name=image_collection_name,
            query_vector=named_vector,
            limit=k
        )
        image_results.extend(image_search_result)
    except Exception as e:
        print(f"Error querying image vector database: {e}")


# Modify process_query function to reduce query vector dimensions for image search
def process_query(query_text, text_embeddings, k=10):
    # Embed the query text
    query_vector_text = text_embeddings.embed_query(query_text)

    # Encode the query text using CLIP (for image search)
    query_vector_image = encode_query_text_with_clip(query_text)

    # Initialize result lists
    text_results = []
    image_results = []

    # Create threads for querying both databases
    text_thread = threading.Thread(target=query_text_vector_db, args=(query_vector_text, text_results, k))
    image_thread = threading.Thread(target=query_image_vector_db, args=(query_vector_image, image_results, k))

    # Start the threads
    text_thread.start()
    image_thread.start()

    # Wait for both threads to complete
    text_thread.join()
    image_thread.join()

    return text_results, image_results

# Assuming ClinicalBertEmbeddings class is defined and initialized
text_embeddings = ClinicalBertEmbeddings()

# Example query
query_text = "what is Gingivitis?"

# Process the query
text_results, image_results = process_query(query_text, text_embeddings, k=10)

# Print the results
print("Text Results:")
for result in text_results:
    print(result)

print("Image Results:")
for result in image_results:
    print(result)
