import { useEffect, useState } from "react";
import { useSearchContext } from "./SearchContext";
import { CopilotSidebar } from "@copilotkit/react-ui";
import { useCopilotReadable } from "@copilotkit/react-core";

export default function CopilotSidebarComponent() {
  const { results, setResults } = useSearchContext();
  const [chatResponse, setChatResponse] = useState("");

  // Define readable state for CopilotKit
  useCopilotReadable({
    description: "The current search results from the user's query",
    value: results,
    id: "search-results",
  });

  // Log results for debugging
  useEffect(() => {
    if (Array.isArray(results) && results.length > 0) {
      console.log("Shared Results:", results);
    } else {
      console.log("No results to share.");
    }
  }, [results]);

  // Function to perform Arxiv search
  const searchArxiv = async () => {
    try {
      const response = await fetch("http://localhost:8000/copilotkit_remote", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: "latest", max_results: 5 }),
      });
      const data = await response.json();
      setResults(data.results || [{ title: "Error", summary: "No results found." }]);
      setChatResponse("Arxiv search results updated.");
    } catch (error) {
      console.error("Error fetching Arxiv results:", error);
      setChatResponse("Error fetching Arxiv results.");
    }
  };

  // Function to perform RAG search
  const searchRag = async () => {
    try {
      const response = await fetch("http://localhost:8000/rag-search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: "latest", max_results: 5 }),
      });
      const data = await response.json();
      const content = data.choices?.[0]?.message?.content;
      setResults(content ? [{ title: "RAG Search Result", summary: content }] : [{ title: "Error", summary: "No results found." }]);
      setChatResponse("RAG search results updated.");
    } catch (error) {
      console.error("Error fetching RAG results:", error);
      setChatResponse("Error fetching RAG results.");
    }
  };

  // Function to perform Web Search
  const searchWeb = async () => {
    try {
      const response = await fetch("http://localhost:8000/web-search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: "latest" }),
      });
      const data = await response.json();
      setResults(data.context ? [{ title: "Web Search Result", summary: data.context }] : [{ title: "Error", summary: "No results found." }]);
      setChatResponse("Web search results updated.");
    } catch (error) {
      console.error("Error fetching Web search results:", error);
      setChatResponse("Error fetching Web search results.");
    }
  };

  // Listen for specific chat commands
  useEffect(() => {
    const handleChatInput = (input: string) => {
      const command = input.toLowerCase();

      if (command.includes("search arxiv")) {
        searchArxiv();
      } else if (command.includes("search rag")) {
        searchRag();
      } else if (command.includes("search web")) {
        searchWeb();
      } else {
        setChatResponse("Unrecognized command. Try 'search arxiv', 'search rag', or 'search web'.");
      }
    };

    // Listen for chat input events
    window.addEventListener("copilot-chat-input", (e: any) => handleChatInput(e.detail));

    return () => {
      window.removeEventListener("copilot-chat-input", handleChatInput);
    };
  }, []);

  return (
    <CopilotSidebar
      defaultOpen={true}
      instructions="You can ask for 'search arxiv', 'search rag', or 'search web'."
      labels={{
        title: "Sidebar Assistant",
        initial: "How can I assist you today?",
      }}
    >
      {chatResponse && (
        <div className="chat-response">
          <pre>{chatResponse}</pre>
        </div>
      )}
    </CopilotSidebar>
  );
}
