from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from app.services.chat_service_v2 import ChatService
from app.auth.dependencies import get_current_active_user, get_premium_user
from app.models.user import User
from app.models.session import ChatMessage
from datetime import datetime

router = APIRouter(prefix="/api/chat", tags=["chat"])
chat_service = ChatService()

# Request/Response Models
class AnonymousChatRequest(BaseModel):
    query: str
    session_token: Optional[str] = None

class PremiumChatRequest(BaseModel):
    query: str
    session_token: Optional[str] = None

class ChatResponse(BaseModel):
    success: bool
    answer: Optional[str] = None
    sources: Optional[List[dict]] = None
    personal_sources: Optional[List[dict]] = None
    session_token: str
    message_id: Optional[str] = None
    response_time_ms: Optional[int] = None
    error: Optional[str] = None

class ChatHistoryResponse(BaseModel):
    success: bool
    chat_history: List[ChatMessage]
    session_token: str


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
    result = await chat_service.anonymous_chat(
        query=request.query.strip(),
        session_token=request.session_token,
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
    result = await chat_service.premium_chat(
        query=request.query.strip(),
        user=current_user,
        session_token=request.session_token,
        ip_address=ip_address,
        user_agent=user_agent
    )
    
    return ChatResponse(**result)

@router.get("/history/{session_token}", response_model=ChatHistoryResponse)
async def get_chat_history(
    session_token: str,
    limit: int = 50
):
    """Get chat history for a session"""
    
    if limit > 100:
        limit = 100  # Maximum limit
    
    chat_history = await chat_service.get_chat_history(session_token, limit)
    
    return ChatHistoryResponse(
        success=True,
        chat_history=chat_history,
        session_token=session_token
    )

@router.delete("/session/{session_token}")
async def clear_chat_session(session_token: str):
    """Clear chat history for a session"""
    
    success = await chat_service.clear_chat_history(session_token)
    
    if success:
        return {"success": True, "message": "Chat history cleared"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

@router.delete("/session/{session_token}/deactivate")
async def deactivate_session(session_token: str):
    """Deactivate a session"""
    
    await chat_service.deactivate_session(session_token)
    return {"success": True, "message": "Session deactivated"}

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