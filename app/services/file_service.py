import os
import uuid
import boto3
import aiofiles
from typing import List, Dict, Any, Optional
from fastapi import UploadFile, HTTPException, status
from pathlib import Path
from botocore.exceptions import ClientError
import magic
from datetime import datetime
from app.database import get_database
from app.models.document import Document, DocumentMetadata
from app.models.user import User
from bson import ObjectId

class FileService:
    def __init__(self):
        # R2 configuration
        self.s3_bucket = os.getenv("S3_BUCKET_NAME")
        self.s3_client = boto3.client(
            's3',
            endpoint_url=os.getenv("R2_ENDPOINT_URL"),  # Cloudflare R2 endpoint URL
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
            region_name="auto"  # R2 uses 'auto' for region
        )
        
        # File type configurations
        self.allowed_extensions = {'.pdf', '.txt', '.docx', '.doc', '.csv', '.md'}
        self.allowed_mime_types = {
            'application/pdf',
            'text/plain',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword',
            'text/csv',
            'text/markdown'
        }
        self.max_file_size = int(os.getenv("MAX_FILE_SIZE_MB", "10")) * 1024 * 1024  # Convert to bytes

    async def validate_file(self, file: UploadFile) -> Dict[str, Any]:
        """Validate uploaded file"""
        # Check file size
        if file.size > self.max_file_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {self.max_file_size // (1024*1024)}MB"
            )
        
        # Check file extension
        file_path = Path(file.filename)
        file_extension = file_path.suffix.lower()
        
        if file_extension not in self.allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type not allowed. Allowed types: {', '.join(self.allowed_extensions)}"
            )
        
        # Read file content for MIME type validation
        content = await file.read()
        await file.seek(0)  # Reset file pointer
        
        # Check MIME type
        mime_type = magic.from_buffer(content, mime=True)
        if mime_type not in self.allowed_mime_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type. Detected: {mime_type}"
            )
        
        return {
            "file_extension": file_extension,
            "mime_type": mime_type,
            "file_size": file.size
        }

    async def save_local_file(self, file: UploadFile, user_id: str) -> str:
        """Save uploaded file to disk"""
        # Generate unique filename
        file_extension = Path(file.filename).suffix.lower()
        unique_filename = f"{user_id}_{uuid.uuid4()}{file_extension}"
        file_path = self.upload_dir / unique_filename
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        return str(file_path)
    
    async def save_s3_file(self, file: UploadFile, user_id: str) -> str:
        """Save uploaded file to S3 bucket"""
        # Generate unique filename
        file_extension = Path(file.filename).suffix.lower()
        unique_filename = f"{user_id}_{uuid.uuid4()}{file_extension}"
        
        # Read file content
        content = await file.read()
        
        # Upload to S3
        try:
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=unique_filename,
                Body=content,
                ContentType=file.content_type
            )
            
            # Return the S3 object key
            return unique_filename
        except ClientError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to upload file to S3: {str(e)}"
            )

    async def delete_local_file(self, file_path: str) -> bool:
        """Delete file from disk"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
            return False

    async def delete_s3_file(self, file_key: str) -> bool:
        """Delete file from S3 bucket"""
        try:
            self.s3_client.delete_object(
                Bucket=self.s3_bucket,
                Key=file_key
            )
            return True
        except ClientError as e:
            print(f"Error deleting file {file_key} from S3: {e}")
            return False

    def get_file_type_from_extension(self, filename: str) -> str:
        """Get file type from extension"""
        extension = Path(filename).suffix.lower()
        type_mapping = {
            '.pdf': 'pdf',
            '.txt': 'txt',
            '.docx': 'docx',
            '.doc': 'doc',
            '.csv': 'csv',
            '.md': 'markdown'
        }
        return type_mapping.get(extension, 'unknown')
