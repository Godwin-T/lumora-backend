from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, status
from fastapi.responses import JSONResponse
from typing import List, Optional
from app.services.document_service import DocumentService
from app.auth.dependencies import get_premium_user
from app.models.user import User
from app.models.document import Document, DocumentUpload
from pydantic import BaseModel
from datetime import datetime
import os

router = APIRouter(prefix="/api/documents", tags=["documents"])
document_service = DocumentService()

# Response Models
class DocumentResponse(BaseModel):
    id: str
    filename: str
    original_filename: str
    file_type: str
    file_size_bytes: int
    upload_date: str
    processed: bool
    processing_status: str
    chunk_count: int
    metadata: dict

class DocumentListResponse(BaseModel):
    success: bool
    documents: List[DocumentResponse]
    total_count: int
    page: int
    per_page: int

class DocumentStatsResponse(BaseModel):
    success: bool
    stats: dict

class ProcessingStatusResponse(BaseModel):
    success: bool
    status: dict

class SearchResultResponse(BaseModel):
    success: bool
    results: List[dict]
    query: str

@router.post("/upload", response_model=dict)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[str] = None,  # Comma-separated tags
    current_user: User = Depends(get_premium_user)
):
    """Upload a document for processing"""
    
    # Parse tags
    tag_list = [tag.strip() for tag in tags.split(",")] if tags else []
    
    document_upload = DocumentUpload(
        title=title,
        description=description,
        tags=tag_list
    )
    
    try:
        document = await document_service.upload_document(
            file=file,
            user=current_user,
            document_upload=document_upload
        )
        
        return {
            "success": True,
            "message": "Document uploaded successfully",
            "document_id": str(document.id),
            "processing_status": document.processing_status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading document: {str(e)}"
        )

@router.get("", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_premium_user)
):
    """Get user's documents with pagination"""
    
    skip = (page - 1) * per_page
    documents = await document_service.get_user_documents(
        user_id=current_user.id,
        skip=skip,
        limit=per_page
    )
    
    # Convert to response format
    document_responses = []
    for doc in documents:
        document_responses.append(DocumentResponse(
            id=str(doc.id),
            filename=doc.filename,
            original_filename=doc.original_filename,
            file_type=doc.file_type,
            file_size_bytes=doc.file_size_bytes,
            upload_date=doc.upload_date.isoformat(),
            processed=doc.processed,
            processing_status=doc.processing_status,
            chunk_count=doc.chunk_count,
            metadata=doc.metadata.dict()
        ))
    
    return DocumentListResponse(
        success=True,
        documents=document_responses,
        total_count=len(document_responses),
        page=page,
        per_page=per_page
    )

@router.get("/{document_id}", response_model=dict)
async def get_document(
    document_id: str,
    current_user: User = Depends(get_premium_user)
):
    """Get specific document details"""
    
    document = await document_service.get_document_by_id(document_id, current_user.id)
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    return {
        "success": True,
        "document": DocumentResponse(
            id=str(document.id),
            filename=document.filename,
            original_filename=document.original_filename,
            file_type=document.file_type,
            file_size_bytes=document.file_size_bytes,
            upload_date=document.upload_date.isoformat(),
            processed=document.processed,
            processing_status=document.processing_status,
            chunk_count=document.chunk_count,
            metadata=document.metadata.dict()
        )
    }

@router.put("/{document_id}/metadata", response_model=dict)
async def update_document_metadata(
    document_id: str,
    metadata_update: DocumentUpload,
    current_user: User = Depends(get_premium_user)
):
    """Update document metadata"""
    
    success = await document_service.update_document_metadata(
        document_id=document_id,
        user_id=current_user.id,
        metadata_update=metadata_update
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or update failed"
        )
    
    return {
        "success": True,
        "message": "Document metadata updated successfully"
    }

@router.delete("/{document_id}", response_model=dict)
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_premium_user)
):
    """Delete document and associated vectors"""
    
    success = await document_service.delete_document(document_id, current_user.id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or deletion failed"
        )
    
    return {
        "success": True,
        "message": "Document deleted successfully"
    }

@router.get("/{document_id}/status", response_model=ProcessingStatusResponse)
async def get_processing_status(
    document_id: str,
    current_user: User = Depends(get_premium_user)
):
    """Get document processing status"""
    
    try:
        status_info = await document_service.get_document_processing_status(
            document_id=document_id,
            user_id=current_user.id
        )
        
        return ProcessingStatusResponse(
            success=True,
            status=status_info
        )
        
    except HTTPException:
        raise

@router.get("/stats/overview", response_model=DocumentStatsResponse)
async def get_document_stats(
    current_user: User = Depends(get_premium_user)
):
    """Get user's document statistics"""
    
    stats = await document_service.get_user_document_stats(current_user.id)
    
    return DocumentStatsResponse(
        success=True,
        stats=stats
    )

@router.post("/search", response_model=SearchResultResponse)
async def search_documents(
    query: str,
    limit: int = Query(10, ge=1, le=50),
    current_user: User = Depends(get_premium_user)
):
    """Search within user's documents"""
    
    if not query or len(query.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query cannot be empty"
        )
    
    results = await document_service.search_user_documents(
        user_id=current_user.id,
        query=query.strip(),
        k=limit
    )
    
    return SearchResultResponse(
        success=True,
        results=results,
        query=query
    )

# Batch operations
@router.delete("/batch", response_model=dict)
async def batch_delete_documents(
    document_ids: List[str],
    current_user: User = Depends(get_premium_user)
):
    """Delete multiple documents"""
    
    if len(document_ids) > 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 20 documents can be deleted at once"
        )
    
    results = []
    for doc_id in document_ids:
        success = await document_service.delete_document(doc_id, current_user.id)
        results.append({"document_id": doc_id, "deleted": success})
    
    successful_deletes = sum(1 for r in results if r["deleted"])
    
    return {
        "success": True,
        "message": f"Deleted {successful_deletes} out of {len(document_ids)} documents",
        "results": results
    }

# Health check
@router.get("/health/check")
async def document_health_check():
    """Check document service health"""
    try:
        # Test file service
        upload_dir_exists = os.path.exists("uploads")
        
        return {
            "status": "healthy",
            "upload_directory_exists": upload_dir_exists,
            "max_file_size_mb": int(os.getenv("MAX_FILE_SIZE_MB", "10")),
            "max_documents_per_user": int(os.getenv("MAX_DOCUMENTS_PER_USER", "50")),
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Document service unhealthy: {str(e)}"
        )
