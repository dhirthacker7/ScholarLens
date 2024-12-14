#newcode1234
import logging
import boto3
import os
import re
import hashlib
import tempfile
import requests
import openai
from io import BytesIO
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from markdown import read_pdf_from_s3, process_pdf
from airflow.extraction_files_embedd import process_folder, list_subfolders
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec, Index
from docling_core.types.doc import PictureItem, TableItem
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from sentence_transformers import SentenceTransformer
import time
import random

# Load environment variables from .env file
load_dotenv("/Users/nishitamatlani/Documents/Assignment4/.env")

# Set up logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

# Initialize SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# AWS and Pinecone Configuration
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("BUCKET_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COMBINED_INDEX_NAME = os.getenv("COMBINED_INDEX_NAME")

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Initialize S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

# Initialize Pinecone
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

# Create Pinecone combined index if it does not exist
if COMBINED_INDEX_NAME not in pinecone_client.list_indexes().names():
    pinecone_client.create_index(
        name=COMBINED_INDEX_NAME,
        dimension=512,  # Adjust dimension if needed
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region="us-east-1")
    )

# Connect to existing Pinecone combined index using the full host URL
combined_index = Index(
    api_key=PINECONE_API_KEY,
    index_name=COMBINED_INDEX_NAME,
    host=f"https://{COMBINED_INDEX_NAME}{PINECONE_ENVIRONMENT}"
)

# Check if indexes are accessible
try:
    _log.info("Connected to combined index: %s", combined_index.describe_index_stats())
except Exception as e:
    _log.error(f"Failed to connect to Pinecone indexes: {e}")

# Initialize CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define the DAG
dag = DAG(
    'Docling_pipeline_combined1',
    description='Fetch PDFs from S3, convert to markdown and images, and store embeddings in Pinecone',
    schedule_interval='@daily',
    start_date=datetime(2024, 11, 1),
    catchup=False
)

def construct_s3_url(bucket, region, path):
    """Construct the full S3 URL from bucket, region, and relative path."""
    return f"https://{bucket}.s3.{region}.amazonaws.com/{path}"

# Utility function to split text into chunks
def split_text_into_chunks(text, chunk_size=500):
    lines = text.splitlines()
    chunks = []
    current_chunk = []
    current_length = 0

    for line in lines:
        line_length = len(line.split())
        if current_length + line_length > chunk_size:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(line)
        current_length += line_length

    if current_chunk:  # Add the last chunk
        chunks.append("\n".join(current_chunk))
    return chunks

# Task 1: Fetch PDFs from S3 and convert to Markdown and images
def fetch_and_convert_pdfs():
    _log.info("Starting fetch_and_convert_pdfs task")
    s3_folder = "pdfs/"
    output_folder = "outputs/"

    try:
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=s3_folder)
        files = [item['Key'] for item in response.get('Contents', []) if item['Key'].endswith('.pdf')]
        _log.info(f"Found {len(files)} PDF files")

        if not files:
            _log.info("No PDF files found in the S3 bucket.")
            return

        for file_key in files:
            _log.info(f"Processing file: {file_key}")
            pdf_filename = file_key.split('/')[-1]
            doc_name = pdf_filename.split('.')[0]
            output_dir = Path(f"/tmp/{doc_name}")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Read PDF from S3 and process using docling
            pdf_content = read_pdf_from_s3(BUCKET_NAME, file_key)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                tmp_pdf.write(pdf_content.read())
                tmp_pdf_path = Path(tmp_pdf.name)

            # Configure Docling pipeline options for better image extraction
            pipeline_options = PdfPipelineOptions()
            pipeline_options.generate_page_images = True
            pipeline_options.generate_picture_images = True

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                        backend=PyPdfiumDocumentBackend
                    )
                }
            )

            conv_res = converter.convert(tmp_pdf_path)
            _log.info(f"Successfully converted {file_key}")

            # Extract content while maintaining structure
            markdown_content = ""
            picture_counter = 0
            table_counter = 0

            for element, _level in conv_res.document.iterate_items():
                if isinstance(element, PictureItem) and element.image:
                    picture_counter += 1
                    image_filename = output_dir / f"{doc_name}-picture-{picture_counter}.png"
                    element.image.pil_image.save(image_filename, format="PNG")

                    s3_image_key = f"outputs/{doc_name}/{image_filename.name}"
                    s3.upload_file(str(image_filename), BUCKET_NAME, s3_image_key)

                    # Embed image in Markdown with alt text
                    markdown_content += f"\n\n![Figure {picture_counter}]({construct_s3_url(BUCKET_NAME, AWS_REGION, s3_image_key)})\n\n"

                elif isinstance(element, TableItem):
                    table_counter += 1
                    table_md = element.export_to_markdown()
                    markdown_content += f"\n\n### Table {table_counter}\n{table_md}\n\n"

                elif hasattr(element, 'text') and element.text:
                    text = element.text.strip()
                    if text.isupper() and len(text.split()) < 10:
                        markdown_content += f"# {text}\n\n"
                    elif text.istitle() and len(text.split()) < 10:
                        markdown_content += f"## {text}\n\n"
                    else:
                        markdown_content += text + "\n\n"

            # Write markdown to local file
            md_filename = output_dir / f"{doc_name}.md"
            with open(md_filename, 'w', encoding="utf-8") as f:
                f.write(markdown_content)

            # Upload Markdown to S3
            tmp_pdf_path.unlink()  # Delete the temporary PDF file
            s3_key_md = f"outputs/{doc_name}/{doc_name}.md"
            s3.upload_file(str(md_filename), BUCKET_NAME, s3_key_md)
            _log.info(f"Successfully uploaded {file_key}")

    except Exception as e:
        _log.error(f"Error fetching PDFs from S3: {e}")
        raise

# Task 2: Process Markdown and store embeddings in Pinecone
def process_and_store_embeddings():
    folder_prefix = "outputs/"
    try:
        # List all objects under the 'outputs/' folder
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=folder_prefix)

        # Extract Markdown files and image files from the response
        md_files = [item['Key'] for item in response.get('Contents', []) if item['Key'].endswith('.md')]
        image_files = [item['Key'] for item in response.get('Contents', []) if item['Key'].endswith('.png')]

        if not md_files and not image_files:
            _log.info("No Markdown or image files found in the S3 bucket.")
            return

        # Process each Markdown file
        for md_file in md_files:
            _log.info(f"Processing Markdown file: {md_file}")

            try:
                # Fetch Markdown file from S3
                response = s3.get_object(Bucket=BUCKET_NAME, Key=md_file)
                markdown_content = response['Body'].read().decode('utf-8')
            except s3.exceptions.NoSuchKey:
                _log.error(f"No such key found in S3 bucket: '{md_file}'")
                continue  # Skip processing if the file is missing

            # Split text into chunks and create embeddings using CLIP
            text_chunks = split_text_into_chunks(markdown_content)
            for idx, chunk in enumerate(text_chunks):
                inputs = clip_processor(text=[chunk], return_tensors="pt", truncation=True)
                text_features = clip_model.get_text_features(**inputs).detach().numpy()

                # Create unique vector ID for the text chunk
                vector_id = f"{md_file}-text-{idx}"

                # Store text embeddings in Pinecone combined index with metadata
                combined_index.upsert(vectors=[{
                    'id': vector_id,
                    'values': text_features.flatten().tolist(),
                    'metadata': {
                        'type': 'text',
                        'content': chunk,
                        'file_name': md_file
                    }
                }])

        # Process each image file
        for idx, image_file in enumerate(image_files):
            _log.info(f"Processing image file: {image_file}")

            try:
                # Fetch image from S3
                response = s3.get_object(Bucket=BUCKET_NAME, Key=image_file)
                image = Image.open(BytesIO(response['Body'].read()))

                # Extract image embeddings using CLIP
                image_inputs = clip_processor(images=image, return_tensors="pt")
                image_features = clip_model.get_image_features(**image_inputs).detach().numpy()

                # Create unique vector ID for the image
                image_vector_id = f"{image_file}-image-{idx}"

                # Store image embeddings in Pinecone combined index with metadata
                combined_index.upsert(vectors=[{
                    'id': image_vector_id,
                    'values': image_features.flatten().tolist(),
                    'metadata': {
                        'type': 'image',
                        'file_name': image_file,
                        'source': construct_s3_url(BUCKET_NAME, AWS_REGION, image_file)
                    }
                }])

            except s3.exceptions.NoSuchKey:
                _log.error(f"No such key found in S3 bucket for image: '{image_file}'")
                continue

            except Exception as e:
                _log.error(f"Error processing image file '{image_file}': {e}")
                continue

    except Exception as e:
        _log.error(f"Error processing embeddings: {e}")
        raise

# Define Airflow Tasks
task_fetch_and_convert_pdfs = PythonOperator(
    task_id='fetch_and_convert_pdfs',
    python_callable=fetch_and_convert_pdfs,
    dag=dag
)

task_extract_and_store_combined_embeddings = PythonOperator(
    task_id='extract_and_store_combined_embeddings',
    python_callable=process_and_store_embeddings,
    dag=dag
)

# Set task dependencies
task_fetch_and_convert_pdfs >> task_extract_and_store_combined_embeddings
