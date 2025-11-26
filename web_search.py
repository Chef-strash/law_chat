import os
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

# Get API Key: https://tavily.com/
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

class WebSearcher:
    def __init__(self):
        self.client = None
        if TAVILY_API_KEY:
            self.client = TavilyClient(api_key=TAVILY_API_KEY)
        else:
            print("⚠️ TAVILY_API_KEY not found. Web search will be disabled.")

    def search(self, query: str, max_results: int = 5) -> Tuple[str, List[Dict]]:
        """
        Performs a smart search for Indian Legal context.
        Returns:
            - context_text: String formatted for the LLM
            - sources: List of dicts with citations
        """
        if not self.client:
            return "", []

        print(f"🌐 Searching Web for: {query}...")
        
        try:
            # 1. Optimize Query for Indian Law
            # We explicitly ask for case law and sections to guide the search engine
            optimized_query = f"{query} Indian Constitutional Law Supreme Court judgments sections"

            # 2. Perform Advanced Search
            # search_depth="advanced" scrapes the full page content, not just snippets
            response = self.client.search(
                query=optimized_query,
                search_depth="advanced",
                max_results=max_results,
                include_domains=[
                    "indiankanoon.org", 
                    "legalserviceindia.com", 
                    "scconline.com", 
                    "livelaw.in",
                    "barandbench.com",
                    "sci.gov.in"
                ]
            )

            # 3. Format Results
            context_parts = []
            sources = []

            for result in response.get('results', []):
                title = result.get('title', 'Unknown Source')
                url = result.get('url', '#')
                content = result.get('content', '')

                # Formatted block for the LLM
                context_parts.append(f"SOURCE: {title}\nURL: {url}\nCONTENT:\n{content}\n")
                
                # Citation object for the frontend
                sources.append({
                    "id": "web", # Marker to show this is external
                    "title": title,
                    "url": url,
                    "text": content[:200] + "..." # Preview
                })

            full_context = "\n---\n".join(context_parts)
            return full_context, sources

        except Exception as e:
            print(f"❌ Web Search Failed: {e}")
            return "", []

# Singleton instance
web_searcher = WebSearcher()