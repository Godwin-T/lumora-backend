from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from .user import PyObjectId
from bson import ObjectId

class ChatMessage(BaseModel):
    message_id: str
    user_message: str
    bot_response: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    context_used: List[str] = []
    response_time_ms: Optional[int] = None

class Session(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: Optional[PyObjectId] = None
    access_token: str
    ip_address: str
    user_agent: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    is_active: bool = True
    chat_history: List[ChatMessage] = []

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}