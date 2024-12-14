from diagrams import Diagram, Cluster
from diagrams.onprem.workflow import Airflow
from diagrams.onprem.analytics import Spark
from diagrams.generic.storage import Storage
from diagrams.onprem.mlops import Mlflow
from diagrams.generic.database import SQL
from diagrams.generic.compute import Rack
from diagrams.custom import Custom
from diagrams.aws.storage import S3

with Diagram("End-to-End Research Tool Architecture", show=False, filename="research_tool_architecture", direction="LR"):
    # Pipeline Cluster
    with Cluster("Airflow Pipeline - Data Acquisition and Preprocessing"):
        # Data acquisition starts with S3 bucket containing PDFs and images
        s3_input = S3("Amazon S3 \n (PDFs)")
        airflow = Airflow("Airflow Scheduler")  # Orchestrates the pipeline
        docling = Custom("Docling Document Parser", "/Users/nishitamatlani/Documents/Assignment4/diagram/images/docling.png")  # Parses the input documents
        s3_output = S3("Amazon S3 \n (MD Files & Images)")  # Stores parsed MD files and images
        openai_embeddings = Custom("OpenAI Embeddings", "/Users/nishitamatlani/Documents/Assignment4/diagram/images/openai.png")  # Generates embeddings for the parsed content
        pinecone = Custom("Pinecone Vector Database", "/Users/nishitamatlani/Documents/Assignment4/diagram/images/pinecone1.png")  # Stores the vector embeddings for efficient querying
        
        # Data flow through the pipeline
        airflow >> s3_input >> docling >> s3_output >> openai_embeddings >> pinecone
    


    # FastAPI Interaction
    langraph = Custom("Langraph Multi-Agent System", "/Users/nishitamatlani/Documents/Assignment4/diagram/images/langraph.png")
    fastapi = Custom("FastAPI Interface", "/Users/nishitamatlani/Documents/Assignment4/diagram/images/fastapi.png")  # Acts as the API interface for interaction
    pinecone >> fastapi  # Pinecone provides data to FastAPI
    fastapi >> langraph  # FastAPI interacts with Langraph for multi-agent processing
    fastapi << langraph  # Langraph also sends data back to FastAPI

    

    # Agent Cluster
    with Cluster("Multi-Agent System - Research Agents"):
        research_agents = [
            Custom("Arxiv Agent", "/Users/nishitamatlani/Documents/Assignment4/diagram/images/download.png"),  # Fetches data from Arxiv
            Custom("Web Search Agent", "/Users/nishitamatlani/Documents/Assignment4/diagram/images/Web_Search.png"),  # Performs web searches
            Custom("RAG Agent", "/Users/nishitamatlani/Documents/Assignment4/diagram/images/Document_Search.png")  # Retrieval-Augmented Generation agent
        ]
        # Langraph interacts with the research agents
        fastapi >> langraph >> research_agents

    # User Interface Cluster
    with Cluster("User Interaction Interface - Frontend"):
        copilot = Custom("Copilot User Interface", "/Users/nishitamatlani/Documents/Assignment4/diagram/images/copilot.png")  # User interacts with the system via Copilot UI
        copilot >> fastapi  # Copilot sends user requests to FastAPI

    # Data Export Cluster
    with Cluster("Data Export - Output Generation"):
        pdf_report = Custom("PDF Export", "/Users/nishitamatlani/Documents/Assignment4/diagram/images/pdf.png")  # Exports results as PDF
        codelabs = Custom("Codelabs Export", "/Users/nishitamatlani/Documents/Assignment4/diagram/images/codelabs.png")  # Exports results for Codelabs
        
        # Research agents produce the final output
        research_agents >> pdf_report
        research_agents >> codelabs
