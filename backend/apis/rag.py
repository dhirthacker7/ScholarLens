import os
from dotenv import load_dotenv
import logging
from openai import OpenAI
from pinecone import Pinecone  # Correct import for Pinecone v3.0+
from langchain_pinecone import Pinecone as PineconeVectorStore  # Correct import for LangChain Pinecone
from langchain_openai import OpenAIEmbeddings  # Updated import
from transformers import CLIPProcessor, CLIPModel
import torch
import boto3  # Add for S3 integration

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Pinecone client using the new object-oriented approach (v3.0+)
api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    logging.error("PINECONE_API_KEY is not set in the environment variables.")
    raise ValueError("PINECONE_API_KEY is missing.")
else:
    logging.info("Pinecone API key loaded successfully.")

pc = Pinecone(api_key=api_key)
logging.info("Initialized Pinecone client.")

# Define index names
image_index_name = "md-images"
text_index_name = "md-text"

# Initialize S3 client
s3_client = boto3.client('s3')
s3_bucket_name = os.getenv('S3_BUCKET_NAME')

# Function to retrieve an image from S3
def get_image_from_s3(image_key):
    try:
        logging.info(f"Retrieving image with key: {image_key} from S3.")
        s3_object = s3_client.get_object(Bucket=s3_bucket_name, Key=image_key)
        return s3_object['Body'].read()  # Returns image binary data
    except Exception as e:
        logging.error(f"Failed to retrieve image from S3: {e}")
        return None

# Check if image index exists, if not, create it
if image_index_name not in pc.list_indexes().names():
    logging.info(f"Index '{image_index_name}' does not exist. Creating a new index...")
    pc.create_index(
        name=image_index_name,
        dimension=512,  # Dimension for image embeddings
        metric='cosine'
    )
    logging.info(f"Index '{image_index_name}' created successfully.")
else:
    logging.info(f"Index '{image_index_name}' already exists.")

# Connect to the existing Pinecone image index
image_index = pc.Index(image_index_name)
logging.info(f"Connected to Pinecone index: {image_index_name}")

# Check if text index exists, if not, create it
if text_index_name not in pc.list_indexes().names():
    logging.info(f"Index '{text_index_name}' does not exist. Creating a new index...")
    pc.create_index(
        name=text_index_name,
        dimension=1536,  # Dimension for text embeddings
        metric='cosine'
    )
    logging.info(f"Index '{text_index_name}' created successfully.")
else:
    logging.info(f"Index '{text_index_name}' already exists.")

# Connect to the existing Pinecone text index
text_index = pc.Index(text_index_name)
logging.info(f"Connected to Pinecone index: {text_index_name}")

# Initialize text embeddings using OpenAI's Ada model
text_embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Load CLIP model and processor directly for image embeddings
model_name = "openai/clip-vit-base-patch16"
processor = CLIPProcessor.from_pretrained(model_name)
clip_model = CLIPModel.from_pretrained(model_name)

# Function to generate image embeddings using CLIP model
def get_image_embedding(image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
    return embedding.cpu().numpy().flatten()  # Flatten to 512-dimension vector

logging.info("CLIP model and processor loaded for image embeddings.")

# Initialize LangChain's Pinecone vector store for text
text_vector_store = PineconeVectorStore(index=text_index, embedding=text_embeddings)
logging.info("Initialized LangChain's Pinecone vector store for text.")

# Initialize LangChain's Pinecone vector store for images
def image_embeddings(image):
    embedding = get_image_embedding(image)
    return embedding

logging.info("Initialized custom image embeddings function for images.")

# Create retrievers using the vector stores' as_retriever method
text_retriever = text_vector_store.as_retriever()
logging.info("Text retriever created from the text vector store.")

image_vector_store = PineconeVectorStore(index=image_index, embedding=image_embeddings)
logging.info("Initialized LangChain's Pinecone vector store for images.")

# Create image retriever using the vector store's as_retriever method
image_retriever = image_vector_store.as_retriever()
logging.info("Image retriever created from the image vector store.")

# NVIDIA API Key and Client Initialization
nvidia_api_key = os.getenv("NVIDIA_API_KEY")
if not nvidia_api_key:
    logging.error("NVIDIA_API_KEY is not set in the environment variables.")
    raise ValueError("NVIDIA_API_KEY is missing.")
else:
    logging.info("NVIDIA API key loaded successfully.")

nvidia_api_url = "https://integrate.api.nvidia.com/v1"
nvidia_model_name = "meta/llama3-8b-instruct"

# Initialize the NVIDIA client using OpenAI's interface
client = OpenAI(
    base_url=nvidia_api_url,
    api_key=nvidia_api_key
)

def call_nvidia_llama_api(prompt: str) -> dict:
    """
    Sends a prompt to the NVIDIA Llama3-8B-Instruct API and retrieves a response.

    Args:
        prompt (str): The input prompt for the Llama3-8B-Instruct model.

    Returns:
        dict: The generated response from Llama3-8B-Instruct.
    """
    logging.info(f"Calling NVIDIA Llama3 API with prompt: {prompt[:50]}...")

    try:
        completion = client.chat.completions.create(
            model=nvidia_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )

        return completion

    except Exception as e:
        logging.error(f"Failed to call NVIDIA API: {str(e)}")
        return {"error": str(e)}

def rag_search(query: str, image_key=None) -> dict:
    """
    Retrieves relevant text and image documents from Pinecone based on the query and generates a response using NVIDIA Llama3-8B-Instruct API.

    Args:
        query (str): The user's query.
        image_key (Optional): The key of the input image in S3 for image-based retrieval.

    Returns:
        dict: The generated response from Llama3-8B-Instruct.
    """
    logging.info(f"Performing RAG search for query: {query}")

    # Retrieve relevant text documents from Pinecone
    relevant_text_docs = text_retriever.get_relevant_documents(query)
    if relevant_text_docs:
        logging.info(f"Retrieved {len(relevant_text_docs)} relevant text documents from Pinecone.")
    else:
        logging.warning("No relevant text documents found for the query.")

    # Retrieve relevant image documents if an image is provided
    relevant_image_docs = []
    if image_key is not None:
        logging.info("Image key provided. Retrieving image from S3.")
        image = get_image_from_s3(image_key)
        if image:
            logging.info("Image retrieved from S3. Performing image-based retrieval.")
            image_embedding = get_image_embedding(image)
            relevant_image_docs = image_retriever.get_relevant_documents(image_embedding)
            if relevant_image_docs:
                logging.info(f"Retrieved {len(relevant_image_docs)} relevant image documents from Pinecone.")
            else:
                logging.warning("No relevant image documents found for the provided image.")
        else:
            logging.error("Failed to retrieve image from S3.")

    # Prepare the prompt by combining query, text documents, and image documents (if any)
    text_content = "\n".join([doc.page_content for doc in relevant_text_docs])
    image_content = "\n".join([doc.page_content for doc in relevant_image_docs])
    prompt = f"Query: {query}\nText Documents: {text_content}\nImage Documents: {image_content}"

    # Call NVIDIA Llama3-8B-Instruct API to generate a response based on the combined prompt
    llama_response = call_nvidia_llama_api(prompt)

    return llama_response
