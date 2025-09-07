import os
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import functools

load_dotenv()

class MongoDB:
    client: AsyncIOMotorClient = None
    database = None

mongodb = MongoDB()

async def connect_to_mongo():
    """Create database connection"""
    mongodb.client = AsyncIOMotorClient(
        os.getenv("MONGODB_URI"),
        server_api=ServerApi('1')
    )
    mongodb.database = mongodb.client[os.getenv("DATABASE_NAME")]
    
    # Create indexes
    await create_indexes()
    print("Connected to MongoDB")
    return mongodb.database

async def close_mongo_connection():
    """Close database connection"""
    mongodb.client.close()
    print("Disconnected from MongoDB")

async def create_indexes():
    """Create necessary indexes"""
    db = mongodb.database
    
    # Users collection indexes
    await db.users.create_index("email", unique=True)
    await db.users.create_index("pinecone_namespace", unique=True)
    
    # Sessions collection indexes
    await db.sessions.create_index("access_token", unique=True)
    await db.sessions.create_index("expires_at", expireAfterSeconds=0)
    await db.sessions.create_index([("user_id", 1), ("created_at", -1)])
    
    # Documents collection indexes
    await db.documents.create_index([("user_id", 1), ("upload_date", -1)])
    await db.documents.create_index([("processed", 1), ("processing_status", 1)])


import asyncio

db = None

async def init_db(res:bool =False):
    global db
    db = await connect_to_mongo()
    if res:
        return db

@functools.cache
def get_database():
    return db


async def get_database_v2():
    db = await init_db(res=True)
    return db

async def get_database_v3():
    global db
    db = await init_db(res=True)
    return db
