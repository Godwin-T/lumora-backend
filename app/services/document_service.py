import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.database import get_database
from app.models.document import Document, DocumentMetadata, DocumentUpload
from app.models.user import User
from app.services.file_service import FileService
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStoreService
from bson import ObjectId
from fastapi import HTTPException, status, UploadFile

class DocumentService:
    def __init__(self):
        self.file_service = FileService()
        self.document_processor = DocumentProcessor()
        self.vector_service = VectorStoreService()
        self.max_documents_per_user = int(os.getenv("MAX_DOCUMENTS_PER_USER", "50"))

    async def upload_document(
        self, 
        file: UploadFile, 

        user: User, 
        document_upload: DocumentUpload
    ) -> Document:
        """Upload and process a document"""
        
        # Check user document limit
        await self._check_document_limit(user.id)
        
        # Validate file
        file_info = await self.file_service.validate_file(file)
        
        # Save file to disk
        file_path = await self.file_service.save_file(file, str(user.id))
        
        # Create document record
        document = Document(
            user_id=user.id,
            filename=Path(file_path).name,
            original_filename=file.filename,
            file_type=self.file_service.get_file_type_from_extension(file.filename),
            file_size_bytes=file_info["file_size"],
            upload_date=datetime.utcnow(),
            processed=False,
            processing_status="pending",
            chunk_count=0,
            pinecone_vector_ids=[],
            metadata=DocumentMetadata(
                title=document_upload.title or file.filename,
                description=document_upload.description,
                tags=document_upload.tags,
                language="en"
            )
        )
        
        # Save to database
        db = get_database()
        result = await db.documents.insert_one(document.dict(by_alias=True))
        document.id = result.inserted_id
        
        # Update user storage stats
        await self._update_user_storage_stats(user.id, file_info["file_size"])
        
        # Process document asynchronously
        asyncio.create_task(
            self.document_processor.process_document(
                file_path=file_path,
                document_id=str(document.id),
                user_id=str(user.id),
                file_type=document.file_type
            )
        )
        
        return document

    async def get_user_documents(
        self, 
        user_id: ObjectId, 
        skip: int = 0, 
        limit: int = 20
    ) -> List[Document]:
        """Get user's documents with pagination"""
        db = get_database()
        
        cursor = db.documents.find(
            {"user_id": user_id}
        ).sort("upload_date", -1).skip(skip).limit(limit)
        
        documents = []
        async for doc_data in cursor:
            documents.append(Document(**doc_data))
        
        return documents

    async def get_document_by_id(self, document_id: str, user_id: ObjectId) -> Optional[Document]:
        """Get specific document by ID (with ownership check)"""
        db = get_database()
        doc_data = await db.documents.find_one({
            "_id": ObjectId(document_id),
            "user_id": user_id
        })
        
        return Document(**doc_data) if doc_data else None

    async def update_document_metadata(
        self, 
        document_id: str, 
        user_id: ObjectId, 
        metadata_update: DocumentUpload
    ) -> bool:
        """Update document metadata"""
        db = get_database()
        
        update_data = {
            "metadata.title": metadata_update.title,
            "metadata.description": metadata_update.description,
            "metadata.tags": metadata_update.tags,
            "updated_at": datetime.utcnow()
        }
        
        result = await db.documents.update_one(
            {"_id": ObjectId(document_id), "user_id": user_id},
            {"$set": update_data}
        )
        
        return result.modified_count > 0

    async def delete_document(self, document_id: str, user_id: ObjectId) -> bool:
        """Delete document and associated vectors"""
        db = get_database()
        
        # Get document details
        document = await self.get_document_by_id(document_id, user_id)
        if not document:
            return False
        
        try:
            # Delete vectors from Pinecone
            if document.pinecone_vector_ids:
                namespace = f"user_{user_id}"
                await self.vector_service.delete_documents(
                    vector_ids=document.pinecone_vector_ids,
                    namespace=namespace
                )
            
            # Delete file from disk
            await self.file_service.delete_file(f"uploads/{document.filename}")
            
            # Delete from database
            result = await db.documents.delete_one({
                "_id": ObjectId(document_id),
                "user_id": user_id
            })
            
            # Update user storage stats
            await self._update_user_storage_stats(user_id, -document.file_size_bytes)
            
            return result.deleted_count > 0
            
        except Exception as e:
            print(f"Error deleting document {document_id}: {e}")
            return False

    async def get_document_processing_status(
        self, 
        document_id: str, 
        user_id: ObjectId
    ) -> Dict[str, Any]:
        """Get document processing status"""
        document = await self.get_document_by_id(document_id, user_id)
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        return {
            "document_id": str(document.id),
            "processing_status": document.processing_status,
            "processed": document.processed,
            "chunk_count": document.chunk_count,
            "error_log": document.error_log,
            "upload_date": document.upload_date,
            "file_type": document.file_type,
            "file_size_bytes": document.file_size_bytes
        }

    async def get_user_document_stats(self, user_id: ObjectId) -> Dict[str, Any]:
        """Get user's document statistics"""
        db = get_database()
        
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {
                "_id": None,
                "total_documents": {"$sum": 1},
                "total_size_bytes": {"$sum": "$file_size_bytes"},
                "processed_documents": {
                    "$sum": {"$cond": [{"$eq": ["$processed", True]}, 1, 0]}
                },
                "pending_documents": {
                    "$sum": {"$cond": [{"$eq": ["$processing_status", "pending"]}, 1, 0]}
                },
                "failed_documents": {
                    "$sum": {"$cond": [{"$eq": ["$processing_status", "failed"]}, 1, 0]}
                }
            }}
        ]
        
        result = await db.documents.aggregate(pipeline).to_list(1)
        
        if result:
            stats = result[0]
            return {
                "total_documents": stats["total_documents"],
                "processed_documents": stats["processed_documents"],
                "pending_documents": stats["pending_documents"],
                "failed_documents": stats["failed_documents"],
                "total_size_mb": round(stats["total_size_bytes"] / (1024 * 1024), 2),
                "remaining_quota": self.max_documents_per_user - stats["total_documents"]
            }
        else:
            return {
                "total_documents": 0,
                "processed_documents": 0,
                "pending_documents": 0,
                "failed_documents": 0,
                "total_size_mb": 0.0,
                "remaining_quota": self.max_documents_per_user
            }

    async def search_user_documents(
        self, 
        user_id: ObjectId, 
        query: str, 
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search within user's documents"""
        namespace = f"user_{user_id}"
        
        results = await self.vector_service.similarity_search_with_score(
            query=query,
            namespaces=[namespace],
            k=k
        )
        
        search_results = []
        for doc, score in results:
            search_results.append({
                "content": doc.page_content,
                "score": score,
                "metadata": doc.metadata,
                "document_id": doc.metadata.get("document_id"),
                "source": doc.metadata.get("source"),
                "chunk_index": doc.metadata.get("chunk_index", 0)
            })
        
        return search_results

    async def _check_document_limit(self, user_id: ObjectId):
        """Check if user has reached document limit"""
        db = get_database()
        count = await db.documents.count_documents({"user_id": user_id})
        
        if count >= self.max_documents_per_user:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Document limit reached. Maximum {self.max_documents_per_user} documents allowed."
            )

    async def _update_user_storage_stats(self, user_id: ObjectId, size_change: int):
        """Update user's storage usage statistics"""
        db = get_database()
        
        # Convert bytes to MB
        size_change_mb = size_change / (1024 * 1024)
        
        await db.users.update_one(
            {"_id": user_id},
            {
                "$inc": {"usage_stats.storage_used_mb": size_change_mb},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )