from apis.arxiv import search_arxiv
from apis.rag import rag_search
from apis.web import search_web
from langgraph.prebuilt import ToolNode

# Define tool nodes for Arxiv, RAG, and Web Search
tool_node = ToolNode(
    tools=[search_arxiv, rag_search, search_web],
)
