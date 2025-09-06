from pydantic import BaseModel, Field, EmailStr
from typing import Optional, Any
from datetime import datetime
from bson import ObjectId
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema


class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type: Any, _handler: GetCoreSchemaHandler):
        return core_schema.union_schema(
            [
                core_schema.is_instance_schema(ObjectId),  # accept ObjectId directly
                core_schema.str_schema()                  # accept string
            ],
        )

    @classmethod
    def validate(cls, value: Any) -> ObjectId:
        if isinstance(value, ObjectId):
            return value
        if isinstance(value, str):
            return ObjectId(value)
        raise TypeError("Invalid ObjectId")

    @classmethod
    def __get_pydantic_json_schema__(cls, _core_schema, handler):
        schema = handler(core_schema.str_schema())
        schema.update(type="string")
        return schema

class UserProfile(BaseModel):
    name: Optional[str] = None
    avatar_url: Optional[str] = None

class UsageStats(BaseModel):
    queries_this_month: int = 0
    documents_uploaded: int = 0
    storage_used_mb: float = 0.0

class SubscriptionDetails(BaseModel):
    plan: str = "free"
    expires_at: Optional[datetime] = None
    auto_renew: bool = False

class User(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    email: EmailStr
    hashed_password: str
    subscription_type: str = "free"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    profile: UserProfile = Field(default_factory=UserProfile)
    usage_stats: UsageStats = Field(default_factory=UsageStats)
    pinecone_namespace: str
    subscription_details: SubscriptionDetails = Field(default_factory=SubscriptionDetails)
    is_active: bool = True

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None
    subscription_type: str = "free"

class UserResponse(BaseModel):
    id: str
    email: str
    subscription_type: str
    profile: UserProfile
    usage_stats: UsageStats
    created_at: datetime
    access_token: str