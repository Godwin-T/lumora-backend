import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
from app.database import init_db


# Load environment variables
load_dotenv()

# Import routers
from app.routers import chat_v2, documents, user

# Create FastAPI app
app = FastAPI(
    title="Document AI API",
    description="API for document processing and AI chat",
    version="1.0.0"
)
@app.on_event("startup")
async def on_startup():
    await init_db()

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:5173",
    "http://localhost:3000",  # Frontend development server
    "https://lumora-ai-eo46.onrender.com",  # Production domain
    "https://lumora.hundred.name.ng",
    "https://lab-romantic-phoenix.ngrok-free.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_v2.router)
app.include_router(documents.router)
app.include_router(user.router)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Document AI API is running",
        "docs": "/docs",
        "version": "1.0.0"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "api_version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "development")
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": f"An unexpected error occurred: {str(exc)}"
        }
    )

# Run the application
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
