import os
import asyncio
from typing import List
from langchain.schema import Document
from app.services.vector_store import VectorStoreService

class KnowledgeBaseService:
    def __init__(self):
        self.vector_service = VectorStoreService()
        self.general_namespace = "general_knowledge"

    async def initialize_general_knowledge(self):
        """Initialize general knowledge base - run this once during setup"""
        
        # Sample general knowledge documents
        # In production, you'd load this from your knowledge base files
        general_docs = [
            Document(
                page_content="FastAPI is a modern, fast web framework for building APIs with Python 3.7+ based on standard Python type hints.",
                metadata={"source": "fastapi_docs", "category": "programming"}
            ),
            Document(
                page_content="LangChain is a framework for developing applications powered by language models. It enables applications that are data-aware and agentic.",
                metadata={"source": "langchain_docs", "category": "ai"}
            ),
            Document(
                page_content="Pinecone is a vector database that makes it easy to build high-performance vector search applications.",
                metadata={"source": "pinecone_docs", "category": "database"}
            ),
            Document(
                page_content="MongoDB is a document database with the scalability and flexibility that you want with the querying and indexing that you need.",
                metadata={"source": "mongodb_docs", "category": "database"}
            ),
            # Add more general knowledge documents here
        ]
        
        # Store in general knowledge namespace
        vector_ids = await self.vector_service.add_documents(
            documents=general_docs,
            namespace=self.general_namespace,
            user_id=None,
            document_id="general_knowledge_base"
        )
        
        print(f"Initialized general knowledge base with {len(vector_ids)} documents")
        return vector_ids

    async def add_general_knowledge_document(self, content: str, metadata: dict):
        """Add a single document to general knowledge base"""
        doc = Document(page_content=content, metadata=metadata)
        
        vector_ids = await self.vector_service.add_documents(
            documents=[doc],
            namespace=self.general_namespace,
            user_id=None,
            document_id="general_knowledge_base"
        )
        
        return vector_ids[0] if vector_ids else None

    async def search_general_knowledge(self, query: str, k: int = 5):
        """Search only general knowledge base"""
        return await self.vector_service.similarity_search(
            query=query,
            namespaces=[self.general_namespace],
            k=k
        )