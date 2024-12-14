from fastapi import FastAPI, HTTPException
from copilotkit import CopilotKitSDK, LangGraphAgent
from langgraph.graph import StateGraph, START, END, MessagesState
from apis.rag import rag_search
from apis.arxiv import search_arxiv
from apis.web import search_web
from apis.router import tool_node  # Updated router with both Arxiv and RAG tools
import logging
from fastapi.middleware.cors import CORSMiddleware
import markdown
import pdfkit
from io import BytesIO
import os
from openai import OpenAI
import re
import tempfile
import shutil
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from io import BytesIO

# Initialize FastAPI app
app = FastAPI()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Load environment variables
aws_access_key_id = os.getenv("AWS_ACCESS_KEY")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION")
s3_bucket_name = os.getenv("BUCKET_NAME")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

# Enable CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, or specify specific ones
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods or specify methods like ['GET', 'POST']
    allow_headers=["*"],  # Allow all headers
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define state graph using MessagesState
state_graph = StateGraph(MessagesState)

def selector(state: MessagesState) -> str:
    """
    Selector function to determine which tool to use based on the last message.
    """
    last_message = state["messages"][-1].content.lower()
    
    if "arxiv" in last_message:
        return "fetch_arxiv"
    
    if "rag" in last_message:
        return "rag_search"
    
    if "web" in last_message:
        return "web_search"
    
    return "finalAnswer"


# Add nodes and edges to state graph based on architecture
state_graph.add_node("selector", selector)
state_graph.add_node("fetch_arxiv", tool_node)  # Run Arxiv search if selected by selector
state_graph.add_node("rag_search", tool_node)  # Run RAG search if selected by selector
state_graph.add_node("finalAnswer", lambda state: {"messages": state["messages"]})

state_graph.add_edge(START, "selector")
state_graph.add_edge("selector", "fetch_arxiv")
state_graph.add_edge("selector", "rag_search")
state_graph.add_edge("selector", "finalAnswer")
state_graph.add_edge("fetch_arxiv", "finalAnswer")
state_graph.add_edge("rag_search", "finalAnswer")

state_graph.add_node("web_search", tool_node)  # Run Web Search if selected by selector
state_graph.add_edge("selector", "web_search")
state_graph.add_edge("web_search", "finalAnswer")


# Compile workflow into runnable graph
workflow = state_graph.compile()

# Initialize CopilotKit SDK with LangGraphAgent
sdk = CopilotKitSDK(
    agents=[LangGraphAgent(
        name="combined_agent",
        description="An agent that searches for research papers using Arxiv, retrieves documents using RAG, or performs a web search using Tavily.",
        graph=workflow,
    )]
)

def run_command(command):
    subprocess.run(command, shell=True, check=True)

# Hardcoded metadata to be included in each Markdown file
HARDCODED_METADATA = """id: nishi-tutorial
summary: A brief codelab tutorial example
categories: Tutorial
tags: beginner, example
authors: Nishita Matlani
status: Draft
feedback link: https://example.com/feedback
"""

# @app.post("/convert-md-to-codelab")
# async def convert_md_to_codelab(payload: dict):
#     """
#     Endpoint to convert markdown content to a CodeLabs URL.

#     Args:
#         payload (dict): A dictionary containing the markdown content.

#     Returns:
#         dict: A dictionary containing the URL of the served codelab.
#     """
#     logger.info("Received payload for markdown-to-codelab conversion")

#     markdown_content = payload.get("markdown", "")

#     if not markdown_content:
#         raise HTTPException(status_code=400, detail="No markdown content provided")

#     try:
#         # Create a fixed temporary file for storing the markdown content
#         markdown_filename = "nishi.md"
#         with open(markdown_filename, "w") as tmp_md_file:
#             # Write the markdown content with hardcoded metadata to the file
#             tmp_md_file.write(HARDCODED_METADATA + "\n\n" + markdown_content)

#         # Verify if the file is written properly
#         if os.stat(markdown_filename).st_size == 0:
#             raise HTTPException(status_code=500, detail="Markdown file is empty after writing")

#         # Check if claat tool is already installed, if not, install it
#         if not os.path.exists("/usr/local/bin/claat"):
#             try:
#                 logger.info("Downloading claat tool...")
#                 run_command("curl -LO https://github.com/googlecodelabs/tools/releases/latest/download/claat-darwin-amd64")
#                 run_command("sudo mv claat-darwin-amd64 /usr/local/bin/claat")
#                 run_command("chmod +x /usr/local/bin/claat")
#             except subprocess.CalledProcessError as e:
#                 logger.error(f"Error downloading or setting up claat: {str(e)}")
#                 raise HTTPException(status_code=500, detail=f"Error downloading or setting up claat: {str(e)}")

#         # Export codelab using claat
#         try:
#             logger.info(f"Exporting codelab from markdown file: {markdown_filename}")
#             run_command(f"claat export {markdown_filename}")

#             # The expected directory name after export
#             export_dir_name = "nishi-tutorial"

#             # Verify that the directory was created
#             if not os.path.exists(export_dir_name):
#                 raise FileNotFoundError("Exported directory could not be found")

#             # Serve the codelab using combined command
#             logger.info(f"Serving codelab inside directory: {export_dir_name}")
#             run_command(f"cd nishi-tutorial")
#             run_command(f"claat serve &")  # Run `cd` and `claat serve` together

#             # Return the URL for the served codelab
#             url = "http://localhost:9090"  # Default URL for `claat serve`
#             return {"url": url}

#         except subprocess.CalledProcessError as e:
#             logger.error(f"Error during claat processing: {str(e)}")
#             raise HTTPException(status_code=500, detail=f"Error converting markdown to CodeLabs URL: {str(e)}")

#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/list-s3-files/")
async def list_s3_files():
    try:
        # List objects within the specified S3 bucket
        response = s3_client.list_objects_v2(Bucket=s3_bucket_name)

        if "Contents" not in response:
            return {"message": "No files found in the bucket."}

        # Filter only files in the pdfs/ folder
        file_names = [content["Key"] for content in response["Contents"] if content["Key"].startswith("pdfs/")]

        return {"files": file_names}
    
    except NoCredentialsError:
        raise HTTPException(status_code=500, detail="AWS credentials not found.")
    except PartialCredentialsError:
        raise HTTPException(status_code=500, detail="Incomplete AWS credentials provided.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/smart-query")
async def smart_query_endpoint(payload: dict):
    """
    Smart endpoint that uses an LLM to decide which search method to use based on user query.

    Args:
        payload (dict): A dictionary containing the user's query.

    Returns:
        dict: The response from the selected search endpoint.
    """
    try:
        user_query = payload.get("query", "")
        if not user_query:
            raise HTTPException(status_code=400, detail="No query provided")

        # Use OpenAI LLM to decide which search method to use
        llm_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Analyze the user query: '{user_query}'. "
                        "Choose the most appropriate search method from the following options:\n"
                        "1. 'web' for general web search using online sources.\n"
                        "2. 'rag' for searching documents and retrieving a summarized response.\n"
                        "3. 'arxiv' for searching academic research papers on Arxiv.\n"
                        "Please reply with only one option: 'web', 'rag', or 'arxiv'."
                    )
                }
            ],
        )

        # Extract and log the LLM's decision
        decision = re.sub(r'^["\']|["\']$', '', llm_response.choices[0].message.content.strip().lower())
        logger.info(f"LLM Decision: '{decision}'")

        # Call the appropriate endpoint based on the decision
        if decision == "arxiv":
            logger.info("Selected: Arxiv Search")
            response = await handle_copilotkit_remote(payload)
        elif decision == "rag":
            logger.info("Selected: RAG Search")
            response = await rag_search_endpoint(payload)
        elif decision == "web":
            logger.info("Selected: Web Search")
            response = await web_search_endpoint(payload)
        else:
            logger.error(f"Unrecognized decision: '{decision}'")
            raise HTTPException(status_code=400, detail="Unable to determine the search method.")

        return response

    except Exception as e:
        logger.error(f"Error in smart-query: {str(e)}")
        return {"error": str(e)}


@app.post("/web-search")
async def web_search_endpoint(payload: dict):
    """
    Endpoint to handle web search queries using Tavily.
    
    Args:
        payload (dict): A dictionary containing the user's query.
    
    Returns:
        dict: The response from the web search process.
    """
    logger.info(f"Received payload: {payload}")
    
    query = payload.get("query", "")
    
    if not query:
        return {"error": "No query provided"}
    
    try:
        # Perform web search
        response = search_web.invoke(query)
        
        logger.info(f"Web search response: {response}")
        
        return response
    
    except Exception as e:
        logger.error(f"Error invoking Web Search tool: {str(e)}")
        return {"error": str(e)}

@app.post("/rag-search")
async def rag_search_endpoint(payload: dict):
    """
    Endpoint to handle RAG search queries.

    Args:
        payload (dict): A dictionary containing the user's query.

    Returns:
        dict: The response from the RAG search process.
    """
    logger.info(f"Received payload: {payload}")  # Log the incoming payload

    query = payload.get("query", "")

    if not query:
        return {"error": "No query provided"}

    try:
        # Perform RAG search
        response = rag_search(query)

        # Extract the response content from the ChatCompletion object
        content = response.choices[0].message.content.strip()

        # Log the extracted content
        logger.info(f"RAG search response content: {content}")

        # Return the response in a format compatible with the frontend
        return {"results": [{"title": "RAG Search Result", "summary": content}]}

    except Exception as e:
        logger.error(f"Error invoking RAG search: {str(e)}")
        return {"error": str(e)}


@app.post("/copilotkit_remote")
async def handle_copilotkit_remote(payload: dict):
    """
    Endpoint to handle CopilotKit remote query processing.
    
    Args:
        payload (dict): A dictionary containing the user's query.
    
    Returns:
        dict: The search results from the Arxiv tool or an error message.
    """
    try:
        logger.info(f"Received payload: {payload}")
        
        # Extract query from the payload directly
        user_query = payload.get('query', '')
        logger.info(f"Running agent with payload: {user_query}")
        
        # Check if query is provided
        if user_query:
            logger.info(f"Searching for papers related to: {user_query}")
            
            # Use the invoke method to get the search results
            search_results = search_arxiv.invoke(user_query)
            
            logger.info(f"Search results returned: {search_results}")
            return search_results  # Return the search results
        else:
            logger.error("No query provided in the payload.")
            return {"error": "No query provided"}
    
    except Exception as e:
        logger.error(f"Error invoking Arxiv tool: {str(e)}")
        return {"error": str(e)}

@app.get("/")
def read_root():
    """
    Basic health check route.
    """
    return {"message": "Hello from Combined Agent!"}

@app.post("/convert-text-to-pdf")
async def convert_text_to_pdf(payload: dict):
    """
    Endpoint to convert raw text to markdown and then convert markdown to PDF.
    
    Args:
        payload (dict): A dictionary containing the raw text.
    
    Returns:
        PDF file as a byte stream.
    """
    logger.info(f"Received payload for text-to-pdf conversion: {payload}")
    
    raw_text = payload.get("text", "")
    
    if not raw_text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    try:
        # Convert raw text to markdown
        markdown_content = markdown.markdown(raw_text)
        
        # Convert markdown to PDF using pdfkit
        pdf_output = pdfkit.from_string(markdown_content, False)
        
        # Return PDF as a byte stream in response
        return {"filename": "output.pdf", "pdf": pdf_output}
    
    except Exception as e:
        logger.error(f"Error converting text to PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error converting text to PDF: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
