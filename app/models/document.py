from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from .user import PyObjectId
from bson import ObjectId

class DocumentMetadata(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = []
    language: str = "en"

class Document(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: PyObjectId
    filename: str
    original_filename: str
    file_type: str
    file_size_bytes: int
    upload_date: datetime = Field(default_factory=datetime.utcnow)
    processed: bool = False
    processing_status: str = "pending"  # pending, processing, completed, failed
    chunk_count: int = 0
    pinecone_vector_ids: List[str] = []
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    error_log: Optional[str] = None

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class DocumentUpload(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = []