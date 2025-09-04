import os
from pinecone import Pinecone
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

class Document:
    """Simple document class to replace LangChain's Document"""
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class VectorStoreService:
    def __init__(self):
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        
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
                
                # Convert Pinecone results to Document objects
                for hit in query_response['result']['hits']:
                    doc = Document(
                        page_content=hit["fields"]['chunk_text'],
                        metadata=hit["fields"]
                    )
                    # Add score to metadata
                    doc.metadata['score'] = hit.get('score', 0)
                    all_results.append(doc)
            except Exception as e:
                print(f"Error searching namespace {namespace}: {e}")
                continue
        
        # Sort by score (descending) and return top k results
        all_results.sort(key=lambda x: x.metadata.get('score', 0), reverse=True)
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

    async def add_documents(
        self, 
        documents: List[Document], 
        namespace: str,
        user_id: str = None,
        document_id: str = None
    ) -> List[str]:
        """Add documents to Pinecone directly with batching"""
        records = []
        vector_ids = []
        batch_size = 90  # Keeping below Pinecone's limit of 96
        
        for i, doc in enumerate(documents):
            # Generate a unique ID for each document chunk
            doc_id = document_id or str(uuid.uuid4())
            vector_id = f"{doc_id}#{i}"
            vector_ids.append(vector_id)
            
            # Create record with metadata
            record = {
                "_id": vector_id,
                "chunk_text": doc.page_content,
                "document_id": doc_id,
                "user_id": user_id,
                "chunk_number": i,
                "created_at": datetime.utcnow().isoformat(),
            }
            
            # Add any additional metadata from the document
            for key, value in doc.metadata.items():
                if key not in record:
                    record[key] = value
                    
            records.append(record)
        
        # Upsert records to Pinecone in batches
        try:
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                self.index.upsert_records(namespace, batch)
                print(f"Upserted batch {i//batch_size + 1} of {(len(records) + batch_size - 1)//batch_size}")
            
            return vector_ids
        except Exception as e:
            print(f"Error upserting records: {e}")
            return []

    async def similarity_search_with_score(
        self, 
        query: str, 
        namespaces: List[str],
        k: int = 5,
        filter_dict: Dict[str, Any] = None
    ) -> List[tuple]:
        """Search with similarity scores"""
        docs = await self.similarity_search(query, namespaces, k, filter_dict)
        return [(doc, doc.metadata.get('score', 0)) for doc in docs]

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
