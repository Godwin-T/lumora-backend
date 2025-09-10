from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
from app.services.chat_service import ChatService
from app.services.langgraph_chat_service import LumoreRagChatService
from app.auth.dependencies import get_premium_user
from app.models.user import User
from app.models.session import ChatMessage
from datetime import datetime

router = APIRouter(prefix="/api/chat", tags=["chat"])
chat_service = ChatService()
langchain_chat_service = LumoreRagChatService()

# Request/Response Models
class AnonymousChatRequest(BaseModel):
    query: str
    access_token: Optional[str] = None

class PremiumChatRequest(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None    
    access_token: Optional[str] = None
    
class StreamChatRequest(BaseModel):
    query: str
    access_token: Optional[str] = None

class ChatResponse(BaseModel):
    success: bool
    answer: Optional[str] = None
    sources: Optional[List[dict]] = None
    personal_sources: Optional[List[dict]] = None
    access_token: str
    message_id: Optional[str] = None
    response_time_ms: Optional[int] = None
    error: Optional[str] = None

class ChatHistoryResponse(BaseModel):
    success: bool
    chat_history: List[ChatMessage]
    access_token: str

@router.post("/anonymous", response_model=ChatResponse)
async def anonymous_chat(
    request: AnonymousChatRequest,
    http_request: Request
):
    """Handle anonymous user chat queries"""
    
    # Validate query
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty"
        )
    
    if len(request.query) > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query too long. Maximum 1000 characters."
        )
    
    # Get client info
    ip_address = http_request.client.host
    user_agent = http_request.headers.get("User-Agent", "Unknown")
    
    # Process chat
    # result = await chat_service.anonymous_chat(
    #     query=request.query.strip(),
    #     access_token=request.access_token,
    #     ip_address=ip_address,
    #     user_agent=user_agent
    # )
    result = await langchain_chat_service.chat(
    query=request.query.strip(),
    access_token=request.access_token,
    ip_address=ip_address,
    user_agent=user_agent
    )
    
    return ChatResponse(**result)

@router.post("/premium", response_model=ChatResponse)
async def premium_chat(
    request: PremiumChatRequest,
    http_request: Request,
    current_user: User = Depends(get_premium_user)
):
    """Handle premium user chat queries with access to personal data"""
    
    # Validate query
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty"
        )
    
    if len(request.query) > 2000:  # Premium users get longer queries
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query too long. Maximum 2000 characters."
        )
    
    # Get client info
    ip_address = http_request.client.host
    user_agent = http_request.headers.get("User-Agent", "Unknown")
    
    # Process chat
    result = await langchain_chat_service.chat(
        query=request.query.strip(),
        user=current_user,
        access_token=request.access_token,
        ip_address=ip_address,
        user_agent=user_agent,
        namespace=f"ID: {current_user.id}",
        doc_id=request.document_ids[0] if request.document_ids else None
    )
    
    return ChatResponse(**result)

@router.get("/history/{access_token}", response_model=ChatHistoryResponse)
async def get_chat_history(
    access_token: str,
    limit: int = 50
):
    """Get chat history for a session"""
    
    if limit > 100:
        limit = 100  # Maximum limit
    
    chat_history = await chat_service.get_chat_history(access_token, limit)
    
    return ChatHistoryResponse(
        success=True,
        chat_history=chat_history,
        access_token=access_token
    )

@router.delete("/session/{access_token}")
async def clear_chat_session(access_token: str):
    """Clear chat history for a session"""
    
    success = await chat_service.clear_chat_history(access_token)
    
    if success:
        return {"success": True, "message": "Chat history cleared"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

@router.delete("/session/{access_token}/deactivate")
async def deactivate_session(access_token: str):
    """Deactivate a session"""
    
    await chat_service.deactivate_session(access_token)
    return {"success": True, "message": "Session deactivated"}

@router.post("/stream/anonymous")
async def stream_anonymous_chat(
    request: StreamChatRequest,
    http_request: Request
):
    """Stream chat response for anonymous users"""
    
    # Validate query
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty"
        )
    
    if len(request.query) > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query too long. Maximum 1000 characters."
        )
    
    # Get client info
    ip_address = http_request.client.host
    user_agent = http_request.headers.get("User-Agent", "Unknown")
    
    # Process streaming chat
    return await chat_service.stream_chat(
        query=request.query.strip(),
        access_token=request.access_token,
        ip_address=ip_address,
        user_agent=user_agent,
        user=None
    )

@router.post("/stream/premium")
async def stream_premium_chat(
    request: StreamChatRequest,
    http_request: Request,
    current_user: User = Depends(get_premium_user)
):
    """Stream chat response for premium users with access to personal data"""
    
    # Validate query
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty"
        )
    
    if len(request.query) > 2000:  # Premium users get longer queries
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query too long. Maximum 2000 characters."
        )
    
    # Get client info
    ip_address = http_request.client.host
    user_agent = http_request.headers.get("User-Agent", "Unknown")
    
    # Process streaming chat
    return await chat_service.stream_chat(
        query=request.query.strip(),
        access_token=request.access_token,
        ip_address=ip_address,
        user_agent=user_agent,
        user=current_user
    )

# Health check endpoint for chat service
@router.get("/health")
async def chat_health_check():
    """Check if chat service is healthy"""
    try:
        # Test vector store connection
        stats = await chat_service.vector_service.get_namespace_stats("general_knowledge")
        
        return {
            "status": "healthy",
            "general_knowledge_vectors": stats["vector_count"],
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Chat service unhealthy: {str(e)}"
        )
