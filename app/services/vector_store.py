import os
from pinecone import Pinecone
from typing import List, Dict, Any, Optional
from langchain_pinecone import PineconeVectorStore
from langchain_core.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import uuid
from datetime import datetime

class DummyEmbeddings(Embeddings):
    """Dummy embeddings for when Pinecone handles embeddings internally"""
    
    def embed_documents(self, texts):
        # Return dummy embeddings since Pinecone handles this
        return [[0.0] * 1536 for _ in texts]  # Adjust dimension as needed
    
    def embed_query(self, text):
        return [0.0] * 1536  # Adjust dimension as needed
    


class VectorStoreService:
    def __init__(self):
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        # self.embeddings = OpenAIEmbeddings()
        
        # Initialize index
        self._initialize_index()
        self.index = self.pc.Index(self.index_name)
    
    async def similarity_search(
        self, 
        query: str, 
        namespaces: List[str],
        k: int = 5,
        filter_dict: Dict[str, Any] = None
    ) -> List[Document]:
        """Search for similar documents across multiple namespaces using Pinecone's native embedding"""
        all_results = []
        
        for namespace in namespaces:
            try:
                # Use Pinecone's native query method which will embed the text automatically
                query_response = self.index.search(
                    namespace=namespace,
                    query={
                            "top_k": k,
                            "inputs": {
                                'text': query
                            }
                        },
                    rerank={
                            "model": "bge-reranker-v2-m3",
                            "top_n": 5,
                            "rank_fields": ["chunk_text"]
                        }, 
                )
                
                # Convert Pinecone results to LangChain Documents
                for hit in query_response['result']['hits']:
                    doc = Document(
                        page_content=hit["fields"]['chunk_text'],
                    )
                    all_results.append(doc)
                    
            except Exception as e:
                print(f"Error searching namespace {namespace}: {e}")
                continue
        
        # Sort by score (descending) and return top k results
        all_results.sort(key=lambda x: getattr(x, 'score', 0), reverse=True)
        return all_results[:k]
        
    def _initialize_index(self):
        """Create Pinecone index if it doesn't exist"""
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
           self.pc.create_index_for_model(
            name=self.index_name,
            cloud="aws",      # adapt to your deployment
            region="us-east-1",  # adapt as needed
            embed={
                "model": "llama-text-embed-v2",
                "field_map": {"text": "chunk_text"}
                }
            )
           print(f"Created Pinecone index: {self.index_name}")
        else:
            print(f"Using existing Pinecone index: {self.index_name}")
    
    def get_vector_store(self, namespace: str = None) -> PineconeVectorStore:
        """Get LangChain Pinecone vector store for specific namespace"""
        return PineconeVectorStore(
            index=self.index,
            text_key="chunk_text",
            namespace=namespace,
            embedding=DummyEmbeddings()
        )

    async def add_documents(
        self, 
        documents: List[Document], 
        namespace: str,
        user_id: str = None,
        document_id: str = None
    ) -> List[str]:
        """Add documents to Pinecone with metadata"""
        vector_store = self.get_vector_store(namespace)
        
        # Add metadata to documents
        for i, doc in enumerate(documents):
            doc.metadata.update({
                "user_id": user_id,
                "document_id": document_id,
                "chunk_index": i,
                "created_at": datetime.utcnow().isoformat(),
                "namespace": namespace
            })
        
        # Add documents and return vector IDs
        vector_ids = vector_store.add_documents(documents)
        return vector_ids


    async def similarity_search_with_score(
        self, 
        query: str, 
        namespaces: List[str],
        k: int = 5,
        filter_dict: Dict[str, Any] = None
    ) -> List[tuple]:
        """Search with similarity scores"""
        all_results = []
        
        for namespace in namespaces:
            vector_store = self.get_vector_store(namespace)
            
            try:
                results = vector_store.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter_dict
                )
                all_results.extend(results)
            except Exception as e:
                print(f"Error searching namespace {namespace}: {e}")
                continue
        
        # Sort by score (lower is better for cosine distance)
        all_results.sort(key=lambda x: x[1])
        return all_results[:k]

    async def delete_documents(self, vector_ids: List[str], namespace: str):
        """Delete specific documents by vector IDs"""
        try:
            self.index.delete(ids=vector_ids, namespace=namespace)
            return True
        except Exception as e:
            print(f"Error deleting documents: {e}")
            return False

    async def delete_user_namespace(self, namespace: str):
        """Delete entire user namespace"""
        try:
            self.index.delete(delete_all=True, namespace=namespace)
            return True
        except Exception as e:
            print(f"Error deleting namespace {namespace}: {e}")
            return False

    async def get_namespace_stats(self, namespace: str) -> Dict[str, Any]:
        """Get statistics for a namespace"""
        try:
            stats = self.index.describe_index_stats()
            namespace_stats = stats.namespaces.get(namespace, {})
            return {
                "vector_count": namespace_stats.get("vector_count", 0),
                "namespace": namespace
            }
        except Exception as e:
            print(f"Error getting namespace stats: {e}")
            return {"vector_count": 0, "namespace": namespace}
