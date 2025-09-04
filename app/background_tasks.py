import asyncio
from typing import List
from datetime import datetime, timedelta
from app.database import get_database
from app.services.document_processor import DocumentProcessor
from app.models.document import Document

class BackgroundTaskManager:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.running = False

    async def start_background_tasks(self):
        """Start all background tasks"""
        if not self.running:
            self.running = True
            asyncio.create_task(self.process_pending_documents())
            asyncio.create_task(self.cleanup_failed_documents())
            print("Background tasks started")

    async def stop_background_tasks(self):
        """Stop background tasks"""
        self.running = False
        print("Background tasks stopped")

    async def process_pending_documents(self):
        """Process documents that are stuck in pending status"""
        while self.running:
            try:
                db = get_database()
                
                # Find pending documents older than 5 minutes
                five_minutes_ago = datetime.utcnow() - timedelta(minutes=5)
                
                pending_docs = await db.documents.find({
                    "processing_status": "pending",
                    "upload_date": {"$lt": five_minutes_ago}
                }).to_list(10)  # Process 10 at a time
                
                for doc_data in pending_docs:
                    document = Document(**doc_data)
                    file_path = f"uploads/{document.filename}"
                    
                    # Process document
                    await self.document_processor.process_document(
                        file_path=file_path,
                        document_id=str(document.id),
                        user_id=str(document.user_id),
                        file_type=document.file_type
                    )
                
                # Wait 1 minute before checking again
                await asyncio.sleep(60)
                
            except Exception as e:
                print(f"Error in process_pending_documents: {e}")
                await asyncio.sleep(60)

    async def cleanup_failed_documents(self):
        """Clean up failed documents older than 24 hours"""
        while self.running:
            try:
                db = get_database()
                
                # Find failed documents older than 24 hours
                day_ago = datetime.utcnow() - timedelta(hours=24)
                
                failed_docs = await db.documents.find({
                    "processing_status": "failed",
                    "upload_date": {"$lt": day_ago}
                }).to_list(50)
                
                for doc_data in failed_docs:
                    document = Document(**doc_data)
                    
                    # Delete file from disk
                    file_path = f"uploads/{document.filename}"
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    
                    # Mark for deletion (don't delete immediately to allow user to see error)
                    await db.documents.update_one(
                        {"_id": document.id},
                        {"$set": {"processing_status": "cleanup_pending"}}
                    )
                
                # Wait 6 hours before checking again
                await asyncio.sleep(21600)
                
            except Exception as e:
                print(f"Error in cleanup_failed_documents: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error

# Global background task manager
background_manager = BackgroundTaskManager()