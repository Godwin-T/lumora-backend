import os
import asyncio
import boto3
import tempfile
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
from app.database import get_database
from app.models.document import Document as DocumentModel
from bson import ObjectId
import tempfile

class DocumentProcessor:
    def __init__(self, db=None):
        self.vector_service = VectorStoreService()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=0,
            separators=["\n\n", "\n", " "]
        )
        self.db = None
        if (db != None):
            self.db = db
            
        # S3/R2 configuration
        self.s3_bucket = os.getenv("S3_BUCKET_NAME")
        self.s3_client = boto3.client(
            's3',
            endpoint_url=os.getenv("R2_ENDPOINT_URL"),
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
            region_name="auto"
        )

    async def process_local_document(
        self, 
        file_path: str, 
        document_id: str,
        user_id: str,
        file_type: str
    ) -> bool:
        """Process uploaded document and store in vector database"""
        try:
            # Get database connection
            if self.db is not None:
                db = self.db
            else:
                db = get_database()
                
            # Update processing status
            await self._update_document_status(document_id, "processing", db=db)
            
            # Load document based on file type
            documents = await self._load_document(file_path, file_type)
            
            if not documents:
                await self._update_document_status(
                    document_id, "failed", "Could not load document", db=db
                )
                return False

            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Generate namespace for user
            namespace = f"user_{user_id}"
            
            # Store in Pinecone
            vector_ids = await self.vector_service.add_documents(
                documents=chunks,
                namespace=namespace,
                user_id=user_id,
                document_id=document_id
            )
            # Update document in MongoDB
            await self._update_document_completion(
                document_id, vector_ids, len(chunks), db=db
            )
            
            return True
            
        except Exception as e:
            print(f"Error processing document {document_id}: {e}")
            await self._update_document_status(
                document_id, "failed", str(e)
            )
            return False
        finally:
            # Clean up temporary file
            if os.path.exists(file_path):
                processed_dir = os.path.join(os.path.dirname(file_path), "..", "processed")
                os.makedirs(processed_dir, exist_ok=True)
                new_path = os.path.join(processed_dir, os.path.basename(file_path))
                os.rename(file_path, new_path)
                
    async def process_document_from_s3(
        self, 
        file_key: str, 
        document_id: str,
        user_id: str,
        file_type: str
    ) -> bool:
        """Process document directly from S3/R2 storage"""
        temp_file = None
        try:
            # Get database connection
            if self.db is not None:
                db = self.db
            else:
                db = get_database()
                
            # Update processing status
            await self._update_document_status(document_id, "processing", db=db)
            
            # Create a temporary file to store the downloaded content
            suffix = f".{file_type.lower()}" if file_type else ""
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_file.close()
            
            # Download file from S3/R2
            try:
                self.s3_client.download_file(
                    Bucket=self.s3_bucket,
                    Key=file_key,
                    Filename=temp_file.name
                )
            except Exception as e:
                await self._update_document_status(
                    document_id, "failed", f"Failed to download from S3: {str(e)}", db=db
                )
                return False
            
            # Load document based on file type
            documents = await self._load_document(temp_file.name, file_type)
            
            if not documents:
                await self._update_document_status(
                    document_id, "failed", "Could not load document", db=db
                )
                return False

            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Generate namespace for user
            namespace = f"ID: {user_id}"
            print(f"Processing document in namespace: {namespace}")
            print(f"Total chunks: {len(chunks)}")
            
            # Store in vector database
            vector_ids = await self.vector_service.add_documents(
                documents=chunks,
                namespace=namespace,
                user_id=user_id,
                document_id=document_id
            )
            
            # Update document in MongoDB
            await self._update_document_completion(
                document_id, vector_ids, len(chunks), db=db
            )
            
            return True
            
        except Exception as e:
            print(f"Error processing document {document_id} from S3: {e}")
            await self._update_document_status(
                document_id, "failed", str(e)
            )
            return False
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
                
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
            processed_documents = []
            for doc in documents:
                # Use a replace method to merge single newlines but preserve paragraph breaks
                cleaned_content = doc.page_content.replace('\n\n', 'PARAGRAPH_BREAK').replace('\n', ' ').replace('PARAGRAPH_BREAK', '\n\n')
                
                # Update the document with the cleaned text
                doc.page_content = cleaned_content
                processed_documents.append(doc)
            return processed_documents

        except Exception as e:
            print(f"Error loading document: {e}")
            return []

    async def _update_document_status(
        self, 
        document_id: str, 
        status: str, 
        error_log: str = None,
        db = None
    ):
        """Update document processing status"""
        if db is None:
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
        chunk_count: int,
        db = None
    ):
        """Update document when processing is complete"""
        if db is None:
            db = get_database()
        await db.documents.update_one(
            {"_id": ObjectId(document_id)},
            {"$set": {
                "processed": True,
                "processing_status": "completed",
                "chunk_count": chunk_count,
                "pinecone_vector_ids": vector_ids,
                "updated_at": datetime.utcnow(),
                "document_id": document_id  # Ensure document_id is set
            }}
        )
