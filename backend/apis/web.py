from tavily import TavilyClient
from langchain_core.tools import tool
import os
import logging
import re

from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()


# Step 1: Initialize TavilyClient
api_key=os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=api_key)


@tool("web_search")
def search_web(query: str) -> dict:
    """
    Searches the web using TavilyClient and retrieves context based on the provided query.
    
    Args:
        query (str): The search term or topic to look up.
    
    Returns:
        dict: A dictionary containing the context string limited to 4-5 lines or an error message.
    """
    try:
        # Step 2: Execute a context search query using Tavily
        context = tavily_client.get_search_context(query=query)
        
        # Step 3: Clean the response using regex to remove unwanted characters
        cleaned_context = re.sub(r"[\\\/\'\"\(\)]", "", context)

        # Step 4: Return the cleaned context string in a dictionary format
        return {"context": cleaned_context}
    
    except Exception as e:
        logger.error(f"Error in search_web: {str(e)}")
        return {"error": f"An error occurred during web search: {str(e)}"}