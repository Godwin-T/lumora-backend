import os
import asyncio
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import(
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader
)
from datetime import datetime
from langchain.schema import Document
from app.services.vector_store import VectorStoreService
from app.database import get_database, get_database_v2
from app.models.document import Document as DocumentModel
from bson import ObjectId
import tempfile

class DocumentProcessor:
    def __init__(self, local:bool = False):
        self.vector_service = VectorStoreService()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=["\n\n", "\n", " "]
        )
        self.local = local

    async def process_document(
        self, 
        file_path: str, 
        document_id: str,
        user_id: str,
        file_type: str
    ) -> bool:
        """Process uploaded document and store in vector database"""
        try:
            # Update processing status
            await self._update_document_status(document_id, "processing")
            
            # Load document based on file type
            documents = await self._load_document(file_path, file_type)
            
            if not documents:
                await self._update_document_status(
                    document_id, "failed", "Could not load document"
                )
                return False

            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Generate namespace for user
            namespace = f"user_{user_id}"
            print(namespace)
            print(len(chunks))
            
            # Store in Pinecone
            vector_ids = await self.vector_service.add_documents(
                documents=chunks,
                namespace=namespace,
                user_id=user_id,
                document_id=document_id
            )
            print(vector_ids)
            print("*"*100)
            # Update document in MongoDB
            await self._update_document_completion(
                document_id, vector_ids, len(chunks)
            )
            
            return True
            
        except Exception as e:
            print(f"Error processing document {document_id}: {e}")
            await self._update_document_status(
                document_id, "failed", str(e)
            )
            return False
        # finally:
        #     # Clean up temporary file
        #     if os.path.exists(file_path):
        #         os.remove(file_path)

    async def _load_document(self, file_path: str, file_type: str) -> List[Document]:
        """Load document based on file type"""
        try:
            if file_type.lower() == "pdf":
                loader = PyPDFLoader(file_path)
            elif file_type.lower() == "txt":
                loader = TextLoader(file_path)
            elif file_type.lower() in ["docx", "doc"]:
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_type.lower() == "csv":
                loader = CSVLoader(file_path)
            else:
                # Try text loader as fallback
                loader = TextLoader(file_path)
            
            documents = loader.load()
            return documents
            
        except Exception as e:
            print(f"Error loading document: {e}")
            return []

    async def _update_document_status(
        self, 
        document_id: str, 
        status: str, 
        error_log: str = None
    ):
        """Update document processing status"""
        if self.local:
            db = await get_database_v2()
        else:
            db = get_database()
        update_data = {
            "processing_status": status,
            "updated_at": datetime.utcnow()
        }
        
        if error_log:
            update_data["error_log"] = error_log
        
        await db.documents.update_one(
            {"_id": ObjectId(document_id)},
            {"$set": update_data}
        )

    async def _update_document_completion(
        self, 
        document_id: str, 
        vector_ids: List[str], 
        chunk_count: int
    ):
        """Update document when processing is complete"""
        if self.local:
            db = await get_database_v2()
        else:
            db = get_database()
        await db.documents.update_one(
            {"_id": ObjectId(document_id)},
            {"$set": {
                "processed": True,
                "processing_status": "completed",
                "chunk_count": chunk_count,
                "pinecone_vector_ids": vector_ids,
                "updated_at": datetime.utcnow()
            }}
        )
