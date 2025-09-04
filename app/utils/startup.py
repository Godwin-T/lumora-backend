from app.services.knowledge_base import KnowledgeBaseService
from app.services.vector_store import VectorStoreService

async def initialize_vector_store():
    """Initialize vector store and general knowledge base"""
    try:
        # Initialize vector store service
        vector_service = VectorStoreService()
        
        # Initialize knowledge base service
        kb_service = KnowledgeBaseService()
        
        # Check if general knowledge base exists
        stats = await vector_service.get_namespace_stats("general_knowledge")
        
        if stats["vector_count"] == 0:
            print("Initializing general knowledge base...")
            await kb_service.initialize_general_knowledge()
        else:
            print(f"General knowledge base already exists with {stats['vector_count']} documents")
        
        print("Vector store initialization complete")
        
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        raise e