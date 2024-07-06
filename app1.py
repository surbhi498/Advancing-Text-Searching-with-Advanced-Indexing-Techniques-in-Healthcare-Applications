import streamlit as st
from langchain import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain_community.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
import os
import json
from huggingface_hub import hf_hub_download
from langchain.retrievers import EnsembleRetriever
# from ingest import ClinicalBertEmbeddings, keyword_retriever
from langchain_community.llms import CTransformers
from transformers import AutoTokenizer, AutoModel
# # Initialize Streamlit app
# st.set_page_config(page_title="Document Retrieval App", layout='wide')

# # Download and initialize LLM model
# MODEL_PATH = './'

# # Some basic configurations for the model
# config = {
#     "max_new_tokens": 2048,
#     "context_length": 4096,
#     "repetition_penalty": 1.1,
#     "temperature": 0.5,
#     "top_k": 50,
#     "top_p": 0.9,
#     "stream": True,
#     "threads": int(os.cpu_count() / 2)
# }

# # We use Langchain's CTransformers llm class to load our quantized model
# llm = CTransformers(model=MODEL_PATH,
#                     config=config)

# # Tokenizer for Mistral-7B-Instruct from HuggingFace
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
# model_name = "aaditya/OpenBioLLM-Llama3-8B-GGUF"
# model_file = "openbiollm-llama3-8b.Q5_K_M.gguf"
# model_path = hf_hub_download(model_name, filename=model_file, local_dir='./')

# local_llm = "openbiollm-llama3-8b.Q5_K_M.gguf"
# llm = LlamaCpp(
#     model_path=local_llm,
#     temperature=0.3,
#     n_ctx=2048,
#     top_p=1
# )

# st.sidebar.title("Document Retrieval App")

# # Initialize embeddings
# embeddings = ClinicalBertEmbeddings()  

# # Qdrant setup for medical_image collection
# url = "http://localhost:6333"
# client_medical = QdrantClient(url=url, prefer_grpc=False)
# db_medical = Qdrant(client=client_medical, embeddings=embeddings, collection_name="medical_image")

# # Qdrant setup for pdf collection
# client_pdf = QdrantClient(url=url, prefer_grpc=False)
# db_pdf = Qdrant(client=client_pdf, embeddings=embeddings, collection_name="pdf")

# # Define retrievers for both collections
# retriever_medical = db_medical.as_retriever(search_kwargs={"k": 1})
# retriever_pdf = db_pdf.as_retriever(search_kwargs={"k": 1})

# # Ensemble retriever combining both retrievers
# ensemble_retriever = EnsembleRetriever(retrievers=[retriever_medical, retriever_pdf], weights=[0.5, 0.5])

# # Prompt template for querying
# prompt_template = """Use the following pieces of information to answer the user's question.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.

# Context: {context}
# Question: {question}

# Only return the helpful answer. Answer must be detailed and well explained.
# Helpful answer:
# """
# prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

# # Streamlit app layout
# with st.sidebar:
#     query = st.text_area("Enter your query here:")
#     if st.button("Get Response"):
#         st.write("Processing query...")
#         chain_type_kwargs = {"prompt": prompt}
#         qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=ensemble_retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs, verbose=True)
#         response = qa(query)

#         # Process response to extract answer, source document, and metadata
#         answer = response['result']
#         source_document = response['source_documents'][0].page_content
#         doc = response['source_documents'][0].metadata['source']

#         # Display response
#         st.subheader("Answer:")
#         st.write(answer)
#         st.subheader("Source Document:")
#         st.write(source_document)
#         st.subheader("Document Metadata:")
#         st.write(doc)

# # Run the app
# if __name__ == '__main__':
#     st.title("Document Retrieval App")
#     st.write("Enter your query in the sidebar and click 'Get Response' to retrieve relevant documents.")
# Define model and prompt template


# Set your Hugging Face API token
os.environ['HUGGINGFACE_HUB_TOKEN'] = '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'

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
model_file = "mistral-7b-instruct.q4_0.bin"

model_path = hf_hub_download(model_name, filename=model_file, local_dir='./', use_auth_token='HUGGINGFACE_HUB_TOKEN')
