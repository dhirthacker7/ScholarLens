import logging
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
import boto3
from PIL import Image
from io import BytesIO
import shutil
import os
from docling_core.types.doc import PictureItem, TableItem
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

# Set up logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")

# Configure S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

BUCKET_NAME = "bdiaassignment4"

def upload_file_to_s3(file_path, bucket, s3_key):
    """Upload a file from local storage to S3."""
    s3.upload_file(str(file_path), bucket, s3_key)
    print(f"Uploaded {file_path} to s3://{bucket}/{s3_key}")

def read_pdf_from_s3(bucket, key):
    """Read a PDF file directly from S3 into memory."""
    response = s3.get_object(Bucket=bucket, Key=key)
    return BytesIO(response['Body'].read())

def process_pdf(input_stream, doc_name, output_dir):
    """Process PDF using Docling and extract content."""
    with NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
        temp_pdf.write(input_stream.read())
        temp_pdf.flush()
        temp_path = temp_pdf.name

        # Configure Docling pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.images_scale = 2.0
        pipeline_options.generate_page_images = True
        pipeline_options.generate_table_images = True
        pipeline_options.generate_picture_images = True

        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend
                )
            }
        )

        start_time = time.time()
        conv_result = doc_converter.convert(Path(temp_path))
        end_time = time.time() - start_time
        _log.info(f"Document {doc_name} converted in {end_time:.2f} seconds.")

    markdown_content = ""
    picture_counter = 0
    table_counter = 0

    # Extract content while maintaining the structure
    for element, _level in conv_result.document.iterate_items():
        if isinstance(element, PictureItem) and element.image:
            picture_counter += 1
            image_filename = output_dir / f"{doc_name}-picture-{picture_counter}.png"
            element.image.pil_image.save(image_filename, format="PNG")

            s3_image_key = f"output1/{doc_name}/{image_filename.name}"
            upload_file_to_s3(image_filename, BUCKET_NAME, s3_image_key)

            # Embed image in Markdown with alt text
            markdown_content += f"\n\n![Figure {picture_counter}]({s3_image_key})\n\n"

        elif isinstance(element, TableItem):
            table_counter += 1
            table_md = element.export_to_markdown()
            markdown_content += f"\n\n### Table {table_counter}\n{table_md}\n\n"

        elif hasattr(element, 'text') and element.text:
            # Detect headings based on content and format them
            text = element.text.strip()
            if text.isupper() and len(text.split()) < 10:
                markdown_content += f"# {text}\n\n"
            elif text.istitle() and len(text.split()) < 10:
                markdown_content += f"## {text}\n\n"
            else:
                markdown_content += text + "\n\n"

    # Save Markdown content to a file
    markdown_path = output_dir / f"{doc_name}-complete.md"
    with markdown_path.open("w", encoding="utf-8") as fp:
        fp.write(markdown_content)

    s3_markdown_key = f"output1/{doc_name}/{markdown_path.name}"
    upload_file_to_s3(markdown_path, BUCKET_NAME, s3_markdown_key)
    print(f"Processed {doc_name} and uploaded Markdown to S3.")

    # Cleanup temporary PDF file
    os.remove(temp_path)
    # Cleanup output directory
    shutil.rmtree(output_dir)

def main():
    s3_folder = "pdfs/"
    output_folder = "output1/"

    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=s3_folder)
    files = [item['Key'] for item in response.get('Contents', []) if item['Key'].endswith('.pdf')]

    for file_key in files:
        pdf_filename = file_key.split('/')[-1]
        doc_name = pdf_filename.split('.')[0]
        output_dir = Path(f"/tmp/{doc_name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        pdf_content = read_pdf_from_s3(BUCKET_NAME, file_key)
        process_pdf(pdf_content, doc_name, output_dir)

if __name__ == "__main__":
    main()
