import os
import re
import logging
import requests
import boto3
import openai
from io import BytesIO
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec, Index
import time
import random

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

# Load credentials from .env
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("BUCKET_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TEXT_INDEX_NAME = os.getenv("TEXT_INDEX_NAME")
IMAGE_INDEX_NAME = os.getenv("IMAGE_INDEX_NAME")

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Initialize S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create Pinecone indexes if they do not exist
if TEXT_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=TEXT_INDEX_NAME,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region="us-east-1")
    )

if IMAGE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=IMAGE_INDEX_NAME,
        dimension=512,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

# Connect to existing Pinecone indexes using the specified environment and index names
text_index = Index(
    api_key=PINECONE_API_KEY,
    index_name=TEXT_INDEX_NAME,
    host=f"https://{TEXT_INDEX_NAME}{PINECONE_ENVIRONMENT}"
)

image_index = Index(
    api_key=PINECONE_API_KEY,
    index_name=IMAGE_INDEX_NAME,
    host=f"https://{IMAGE_INDEX_NAME}{PINECONE_ENVIRONMENT}"
)

# Check if indexes are accessible
try:
    _log.info("Connected to text index: %s", text_index.describe_index_stats())
    _log.info("Connected to image index: %s", image_index.describe_index_stats())
except Exception as e:
    _log.error(f"Failed to connect to Pinecone indexes: {e}")


# Initialize CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def download_image_from_url(url):
    """Download an image directly from a URL."""
    try:
        _log.info(f"Downloading image from URL: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        _log.error(f"Error downloading image from URL {url}: {e}")
    return None

def construct_s3_url(bucket, region, path):
    """Construct the full S3 URL from bucket, region, and relative path."""
    return f"https://{bucket}.s3.{region}.amazonaws.com/{path}"

def read_markdown_from_s3(bucket, key):
    """Read a Markdown file from S3."""
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        return response['Body'].read().decode('utf-8')
    except Exception as e:
        _log.error(f"Error reading Markdown file {key} from S3: {e}")
        return None

def get_openai_embedding(text):
    """Generate text embeddings using OpenAI."""
    try:
        response = openai.Embedding.create(model="text-embedding-ada-002", input=[text])
        return response['data'][0]['embedding']
    except Exception as e:
        _log.error(f"Error generating text embedding: {e}")
        return None

def get_clip_embedding(image):
    """Generate image embeddings using CLIP."""
    try:
        inputs = clip_processor(images=image, return_tensors="pt")
        embeddings = clip_model.get_image_features(**inputs)
        return embeddings.detach().numpy()[0]
    except Exception as e:
        _log.error(f"Error generating image embedding: {e}")
        return None

def process_markdown_content(md_content, folder_prefix, file_name):
    """Extract text and images from Markdown content."""
    text_embeddings = []
    image_embeddings = []
    pattern = r'!\[.*?\]\((.*?)\)'

    lines = md_content.split("\n")
    for idx, line in enumerate(lines):
        match = re.search(pattern, line)
        if match:
            image_path = match.group(1).strip()
            
            # Construct the full S3 URL using bucket, region, and relative path
            full_image_url = construct_s3_url(BUCKET_NAME, AWS_REGION, image_path)
            
            _log.info(f"Attempting to download image from URL: {full_image_url}")
            
            # Download the image using the constructed URL
            image = download_image_from_url(full_image_url)
            
            if image:
                image_embedding = get_clip_embedding(image)
                if image_embedding is not None:
                    image_embeddings.append({
                        "id": f"{file_name}-image-{idx}",
                        "embedding": image_embedding,
                        "metadata": {
                            "file_name": file_name,
                            "type": "image",
                            "image_path": image_path
                        }
                    })
                    _log.info(f"Processed image: {image_path}")
                else:
                    _log.error(f"Failed to generate embedding for image: {full_image_url}")
        else:
            if line.strip():
                text = line.strip()
                text_embedding = get_openai_embedding(text)
                if text_embedding is not None:
                    text_embeddings.append({
                        "id": f"{file_name}-text-{idx}",
                        "embedding": text_embedding,
                        "metadata": {
                            "file_name": file_name,
                            "type": "text",
                            "content": text
                        }
                    })
                    _log.info(f"Processed text: {text[:30]}...")

    return text_embeddings, image_embeddings

def upload_to_pinecone(embeddings, index, batch_size=10):
    """Upload embeddings to Pinecone with metadata in batches."""
    for i in range(0, len(embeddings), batch_size):
        batch = embeddings[i:i + batch_size]
        try:
            index.upsert(vectors=[{
                "id": entry["id"],
                "values": entry["embedding"],
                "metadata": entry["metadata"]
            } for entry in batch])
            _log.info(f"Uploaded batch {i // batch_size + 1} to Pinecone.")
        except Exception as e:
            _log.error(f"Failed to upload batch {i // batch_size + 1}: {e}")

def upload_to_pinecone_with_retry(embeddings, index, batch_size=10, max_retries=3):
    """Upload embeddings with retry logic."""
    for i in range(0, len(embeddings), batch_size):
        batch = embeddings[i:i + batch_size]
        retries = 0
        while retries < max_retries:
            try:
                index.upsert(vectors=[{
                    "id": entry["id"],
                    "values": entry["embedding"],
                    "metadata": entry["metadata"]
                } for entry in batch])
                _log.info(f"Uploaded batch {i // batch_size + 1} to Pinecone.")
                break
            except Exception as e:
                _log.error(f"Error uploading batch {i // batch_size + 1}: {e}")
                retries += 1
                time.sleep(2 + random.uniform(0, 1))  # Exponential backoff
        else:
            _log.error(f"Failed to upload batch {i // batch_size + 1} after {max_retries} retries.")

def list_subfolders(bucket, prefix):
    """List all subfolders in a given S3 bucket and prefix."""
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')
    subfolders = [item['Prefix'] for item in response.get('CommonPrefixes', [])]
    return subfolders
    

def process_folder(folder_prefix):
    """Process a single folder."""
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=folder_prefix)
    md_file_key = next((item['Key'] for item in response.get('Contents', []) if item['Key'].endswith('.md')), None)
    
    if not md_file_key:
        _log.warning(f"No Markdown file found in {folder_prefix}")
        return

    md_content = read_markdown_from_s3(BUCKET_NAME, md_file_key)
    folder_name = folder_prefix.split('/')[-2]
    text_embeddings, image_embeddings = process_markdown_content(md_content, folder_prefix, folder_name)

    if text_embeddings:
        upload_to_pinecone(text_embeddings, text_index)
    if image_embeddings:
        upload_to_pinecone(image_embeddings, image_index)


def main():
    """Main function to process all subfolders in the output1/ folder."""
    folder_prefix = "output1/"
    
    # List all subfolders inside 'output1/'
    subfolders = list_subfolders(BUCKET_NAME, folder_prefix)
    
    # Process each subfolder
    for subfolder in subfolders:
        _log.info(f"Processing folder: {subfolder}")
        process_folder(subfolder)

if __name__ == "__main__":
    main()
