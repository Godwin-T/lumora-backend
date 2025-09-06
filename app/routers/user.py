from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import timedelta
from typing import Annotated
import os
from app.database import get_database
from dotenv import load_dotenv

from app.models.user import User, UserCreate, UserResponse
from app.auth.utils import (
    verify_password, 
    get_password_hash, 
    create_access_token, 
    verify_token,
    generate_namespace_id
)

load_dotenv()

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Create router
router = APIRouter(tags=["Authentication"])

# Helper functions
async def get_user_by_email(email: str):
    """Get a user by email from the database"""
    db = get_database()
    user = await db.users.find_one({"email": email})
    if user:
        return User(**user)
    return None

async def authenticate_user(email: str, password: str):
    """Authenticate a user by email and password"""
    user = await get_user_by_email(email)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    """Get the current user from the token"""
    payload = verify_token(token)
    email = payload.get("sub")
    user = await get_user_by_email(email)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    return user

# Endpoints
@router.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def signup(user_data: UserCreate):
    """Register a new user"""
    # Check if user already exists
    existing_user = await get_user_by_email(user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    
    # Generate a unique namespace for Pinecone
    pinecone_namespace = generate_namespace_id(user_data.email)
    
    # Create user object
    user = User(
        email=user_data.email,
        hashed_password=hashed_password,
        pinecone_namespace=pinecone_namespace,
        subscription_type=user_data.subscription_type
    )
    
    # Set name if provided
    if user_data.name:
        user.profile.name = user_data.name
    
    # Insert into database
    db = get_database()
    await db.users.insert_one(user.dict(by_alias=True))

    # Create access token
    access_token_expires = timedelta(minutes=1440)  # 24 hours
    access_token = create_access_token(
        data={"sub": user.email},
        expires_delta=access_token_expires
    )
    
    
    # Convert to response model
    user_response = UserResponse(
        id=str(user.id),
        email=user.email,
        subscription_type=user.subscription_type,
        profile=user.profile,
        usage_stats=user.usage_stats,
        created_at=user.created_at,
        access_token=access_token
    )
    
    return user_response

@router.post("/token", response_model=UserResponse)
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    """Login and get access token"""
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=1440)  # 24 hours
    access_token = create_access_token(
        data={"sub": user.email},
        expires_delta=access_token_expires
    )
    
    # Convert to response model
    user_response = UserResponse(
        id=str(user.id),
        email=user.email,
        subscription_type=user.subscription_type,
        profile=user.profile,
        usage_stats=user.usage_stats,
        created_at=user.created_at,
        access_token=access_token
    )
    
    return user_response

@router.get("/users/me", response_model=UserResponse)
async def get_current_user_info(current_user: Annotated[User, Depends(get_current_user)]):
    """Get current user information"""
    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        subscription_type=current_user.subscription_type,
        profile=current_user.profile,
        usage_stats=current_user.usage_stats,
        created_at=current_user.created_at
    )

@router.put("/users/me", response_model=UserResponse)
async def update_user_profile(
    profile_update: dict,
    current_user: Annotated[User, Depends(get_current_user)]
):
    """Update user profile information"""
    # Only allow updating profile fields
    update_data = {}
    
    if "name" in profile_update:
        update_data["profile.name"] = profile_update["name"]
    
    if "avatar_url" in profile_update:
        update_data["profile.avatar_url"] = profile_update["avatar_url"]
    
    db = get_database()
    if update_data:
        await db.users.update_one(
            {"_id": current_user.id},
            {"$set": update_data}
        )
    
    # Get updated user
    updated_user = await get_user_by_email(current_user.email)
    
    return UserResponse(
        id=str(updated_user.id),
        email=updated_user.email,
        subscription_type=updated_user.subscription_type,
        profile=updated_user.profile,
        usage_stats=updated_user.usage_stats,
        created_at=updated_user.created_at
    )
