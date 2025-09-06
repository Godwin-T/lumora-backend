import os
import uuid
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from app.services.vector_store_v2 import VectorStoreService
from app.database import get_database
from app.models.session import Session, ChatMessage
from app.models.user import User
from app.prompts import base_prompt, premium_additional_prompt
from bson import ObjectId
import json
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse

load_dotenv()

class ChatService:
    
    def __init__(self):
        self.vector_service = VectorStoreService()
        self.llm = ChatOpenAI(
            temperature=0.1,
            model="gpt-5-mini",
            max_tokens=1000
        )
        # self.llm = ChatGoogleGenerativeAI(
        #     model="gemini-2.5-flash",
        #     temperature=0,
        #     max_tokens=1000,
        #     timeout=None,
        #     max_retries=2,
        # )
        
        # Rate limiting
        self.anonymous_rate_limit = int(os.getenv("ANONYMOUS_QUERIES_PER_HOUR", "10"))
        self.premium_rate_limit = int(os.getenv("PREMIUM_QUERIES_PER_HOUR", "100"))
        self.default_namespace = [os.getenv("DEFAULT_NAMESPACE", 'user_Default_User')]

    async def create_anonymous_session(self, ip_address: str, user_agent: str, access_token: str) -> Session:
        """Create session for anonymous user"""
        expires_at = datetime.utcnow() + timedelta(hours=24)
        
        session = Session(
            user_id=None,
            access_token=access_token,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=expires_at,
            is_active=True,
            chat_history=[]
        )
        
        db = get_database()
        result = await db.sessions.insert_one(session.dict(by_alias=True))
        session.id = result.inserted_id
        
        return session

    async def get_session(self, access_token: str) -> Optional[Session]:
        """Get session by token"""
        db = get_database()
        session_data = await db.sessions.find_one({"access_token": access_token})
        
        if not session_data:
            return None
        
        session = Session(**session_data)
        
        # Check if session is expired
        if session.expires_at < datetime.utcnow():
            await self.deactivate_session(access_token)
            return None
            
        return session

    async def deactivate_session(self, access_token: str):
        """Deactivate session"""
        db = get_database()
        await db.sessions.update_one(
            {"access_token": access_token},
            {"$set": {"is_active": False}}
        )

    async def check_rate_limit(self, session: Session, user: Optional[User] = None) -> bool:
        """Check if user has exceeded rate limit"""
        now = datetime.utcnow()
        one_hour_ago = now - timedelta(hours=1)
        
        # Count messages in last hour
        recent_messages = [
            msg for msg in session.chat_history 
            if msg.timestamp >= one_hour_ago
        ]
        
        limit = self.premium_rate_limit if user else self.anonymous_rate_limit
        return len(recent_messages) < limit

    async def anonymous_chat(
        self, 
        query: str, 
        access_token: str,
        ip_address: str,
        user_agent: str
    ) -> Dict[str, Any]:
        """Handle anonymous user chat"""
        start_time = datetime.utcnow()
        
        # Get or create session
        session = await self.get_session(access_token)
        if not session:
            session = await self.create_anonymous_session(ip_address, user_agent, access_token)
        
        # Check rate limit
        if not await self.check_rate_limit(session):
            return {
                "success": False,
                "error": "Rate limit exceeded. Please try again later.",
                "access_token": session.access_token
            }
        
        try:
            # Search general knowledge base
            relevant_docs = await self.vector_service.similarity_search(
                query=query,
                namespaces=self.default_namespace,
                k=3
            )
            
            # Get conversation memory
            memory = self._build_conversation_memory(session.chat_history)
            
            # Generate response
            response = await self._generate_response(
                query=query,
                relevant_docs=relevant_docs,
                memory=memory
            )
            
            # Calculate response time
            end_time = datetime.utcnow()
            response_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Create chat message
            message_id = str(uuid.uuid4())
            chat_message = ChatMessage(
                message_id=message_id,
                user_message=query,
                bot_response=response["answer"],
                timestamp=end_time,
                context_used=[doc.metadata.get("source", "") for doc in relevant_docs],
                response_time_ms=response_time_ms
            )
            
            # Update session with new message
            await self._update_session_history(session.access_token, chat_message)
            
            return {
                "success": True,
                "answer": response["answer"],
                "sources": response.get("sources", []),
                "access_token": session.access_token,
                "message_id": message_id,
                "response_time_ms": response_time_ms
            }
            
        except Exception as e:
            print(f"Error in anonymous chat: {e}")
            return {
                "success": False,
                "error": "Sorry, I encountered an error processing your request.",
                "access_token": session.access_token
            }

    async def premium_chat(
        self,
        query: str,
        user: User,
        access_token: str,
        ip_address: str,
        user_agent: str
    ) -> Dict[str, Any]:
        """Handle premium user chat with access to personal data"""
        start_time = datetime.utcnow()
        
        # Get or create session
        session = await self.get_session(access_token)
        if not session or session.user_id != user.id:
            session = await self.create_premium_session(user, ip_address, user_agent, access_token)
        
        # Check rate limit
        if not await self.check_rate_limit(session, user):
            return {
                "success": False,
                "error": "Rate limit exceeded. Please try again later.",
                "access_token": session.access_token
            }
        
        try:
            # Search both user's personal data and general knowledge
            namespaces = [user.pinecone_namespace, self.default_namespace]
            relevant_docs = await self.vector_service.similarity_search(
                query=query,
                namespaces=namespaces,
                k=8  # More docs for premium users
            )
            
            # Get conversation memory
            memory = self._build_conversation_memory(session.chat_history)
            
            # Generate response with enhanced context
            response = await self._generate_response(
                query=query,
                relevant_docs=relevant_docs,
                memory=memory,
                is_premium=True
            )
            
            # Calculate response time
            end_time = datetime.utcnow()
            response_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Create chat message
            message_id = str(uuid.uuid4())
            chat_message = ChatMessage(
                message_id=message_id,
                user_message=query,
                bot_response=response["answer"],
                timestamp=end_time,
                context_used=[doc.metadata.get("source", "") for doc in relevant_docs],
                response_time_ms=response_time_ms
            )
            
            # Update session with new message
            await self._update_session_history(session.access_token, chat_message)
            
            # Update user usage stats
            await self._update_user_usage(user.id)
            
            return {
                "success": True,
                "answer": response["answer"],
                "sources": response.get("sources", []),
                "personal_sources": response.get("personal_sources", []),
                "access_token": session.access_token,
                "message_id": message_id,
                "response_time_ms": response_time_ms
            }
            
        except Exception as e:
            print(f"Error in premium chat: {e}")
            return {
                "success": False,
                "error": "Sorry, I encountered an error processing your request.",
                "access_token": session.access_token
            }

    async def create_premium_session(self, user: User, ip_address: str, user_agent: str, access_token: str) -> Session:
        """Create session for premium user"""
        expires_at = datetime.utcnow() + timedelta(days=7)  # Longer session for premium
        
        session = Session(
            user_id=user.id,
            access_token=access_token,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=expires_at,
            is_active=True,
            chat_history=[]
        )
        
        db = get_database()
        result = await db.sessions.insert_one(session.dict(by_alias=True))
        session.id = result.inserted_id
        
        return session

    def _build_conversation_memory(self, chat_history: List[ChatMessage]) -> ConversationBufferWindowMemory:
        """Build conversation memory from chat history"""
        # memory = ConversationBufferWindowMemory(
        #     k=5,  # Remember last 5 exchanges
        #     memory_key="chat_history",
        #     return_messages=True
        # )
        memory = []
        for msg in chat_history[-5:]:  # Last 5 messages
            memory.append(f"User:  {msg.user_message}")
            memory.append(f"AI Response: {msg.bot_response}")
        
        # Add recent messages to memory
        # for msg in chat_history[-5:]:  # Last 5 messages
        #     memory.chat_memory.add_user_message(msg.user_message)
        #     memory.chat_memory.add_ai_message(msg.bot_response)
        #     print(msg.user_message)
        #     print(msg.bot_response)
        
        return memory

    async def _generate_response(
        self,
        query: str,
        relevant_docs: List,
        memory: ConversationBufferWindowMemory = None,
        is_premium: bool = False
    ) -> Dict[str, Any]:
        """Generate response using LangChain"""
        
        # Build context from relevant documents
        context_parts = []
        sources = []
        personal_sources = []
        
        for doc in relevant_docs:
            context_parts.append(doc.page_content)
            source_info = {
                "source": doc.metadata.get("source", "Unknown"),
                "category": doc.metadata.get("category", "general")
            }
            sources.append(source_info)
            
            # Track personal vs general sources for premium users
            if is_premium and doc.metadata.get("user_id"):
                personal_sources.append(source_info)
        
        context = "\n\n".join(context_parts)
        
        # Create system prompt
        try:
            chat_history = memory.chat_memory.messages if memory.chat_memory.messages else []
        except:
            chat_history = memory
        system_prompt = self._build_system_prompt(is_premium)
        
        
        # Create the full prompt
        prompt = f"""
        {system_prompt}     

        Context information:
        {context}

        Conversation History:
        {chat_history}
        
        Question: {query}
        
        """
        
        # {memory.chat_memory.messages if memory.chat_memory.messages else 'No prior messages.'}
        # Get chat history for context
        # chat_history = memory.chat_memory.messages if memory.chat_memory.messages else []
        
        # Generate response
        # response = self.llm.invoke([
        #     {"role": "system", "content": system_prompt},
        #     {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        # ])
        response = self.llm.invoke(prompt)
        return {
            "answer": response.content,
            "sources": sources,
            "personal_sources": personal_sources if is_premium else []
        }
        
    async def _prepare_prompt(
        self,
        query: str,
        relevant_docs: List,
        is_premium: bool = False
    ) -> Dict[str, Any]:
        """Prepare prompt and context information for response generation"""
        # Build context from relevant documents
        context_parts = []
        sources = []
        personal_sources = []
        
        for doc in relevant_docs:
            context_parts.append(doc.page_content)
            source_info = {
                "source": doc.metadata.get("source", "Unknown"),
                "category": doc.metadata.get("category", "general")
            }
            sources.append(source_info)
            
            # Track personal vs general sources for premium users
            if is_premium and doc.metadata.get("user_id"):
                personal_sources.append(source_info)
        
        context = "\n\n".join(context_parts)
        
        # Create system prompt
        system_prompt = self._build_system_prompt(is_premium)
        
        # Create the full prompt
        prompt = f"""
        {system_prompt}

        Conversation History:
        
        Context information:
        {context}
        
        Question: {query}
        
        Answer based on the provided context. If the information isn't in the context, say so clearly.
        """
        
        return {
            "prompt": prompt,
            "sources": sources,
            "personal_sources": personal_sources if is_premium else []
        }
        
    async def stream_chat(
        self,
        query: str,
        access_token: str,
        ip_address: str,
        user_agent: str,
        user: Optional[User] = None
    ) -> StreamingResponse:
        """Stream chat response to the client"""
        start_time = datetime.utcnow()
        
        # Get or create session
        session = await self.get_session(access_token)
        if not session:
            if user:
                session = await self.create_premium_session(user, ip_address, user_agent, access_token)
            else:
                session = await self.create_anonymous_session(ip_address, user_agent, access_token)
        
        # Check rate limit
        if not await self.check_rate_limit(session, user):
            async def error_generator():
                yield "Error: Rate limit exceeded. Please try again later."
            return StreamingResponse(error_generator())
        
        try:
            # Search for relevant documents
            namespaces = [user.pinecone_namespace, self.default_namespace] if user else self.default_namespace
            relevant_docs = await self.vector_service.similarity_search(
                query=query,
                namespaces=namespaces,
                k=8 if user else 3
            )
            
            # Prepare prompt and context
            prompt_data = await self._prepare_prompt(
                query=query,
                relevant_docs=relevant_docs,
                is_premium=bool(user)
            )
            import markdown
            # Create streaming response
            async def response_generator():
                full_response = ""
                
                # Stream the response
                async for chunk in self.llm.astream(prompt_data["prompt"]):
                    content = chunk.content if hasattr(chunk, "content") else str(chunk)
                    words = content.split()  # Split the chunk into words
                    for word in words:
                        await asyncio.sleep(0.07)  # Adds a half-second delay between each word
                        yield f"data: {word} \n\n"  # Yield each word separately

                # async for chunk in self.llm.astream(prompt_data["prompt"]):
                #     # Extract the chunk content
                #     content = chunk.content if hasattr(chunk, "content") else str(chunk)
                    
                #     # Convert markdown to HTML for proper rendering
                #     html_content = markdown.markdown(content)
                    
                #     # Optionally, strip HTML tags if needed
                #     # soup = BeautifulSoup(html_content, "html.parser")
                #     # clean_text = soup.get_text()  # Get just the plain text
                    
                #     # Split the HTML content into words
                #     words = html_content.split()  # This splits into words, preserving HTML formatting
                    
                #     # Yield each word one by one with a delay
                #     for word in words:
                #         await asyncio.sleep(0.07)  # Adjust this delay as needed
                #         yield f"data: {word} \n\n"  # Yield each word separately

                yield "data: [DONE]\n\n"
                # After streaming is complete, save the message
                end_time = datetime.utcnow()
                response_time_ms = int((end_time - start_time).total_seconds() * 1000)
                
                # Create chat message
                message_id = str(uuid.uuid4())
                chat_message = ChatMessage(
                    message_id=message_id,
                    user_message=query,
                    bot_response=full_response,
                    timestamp=end_time,
                    context_used=[doc.metadata.get("source", "") for doc in relevant_docs],
                    response_time_ms=response_time_ms
                )
                
                # Update session with new message
                await self._update_session_history(session.access_token, chat_message)
                
                # Update user usage stats if premium
                if user:
                    await self._update_user_usage(user.id)
            
            # return StreamingResponse(response_generator(), media_type="text/plain")
            return StreamingResponse(response_generator(), media_type="text/event-stream")            
        
        except Exception as e:
            print(f"Error in streaming chat: {e}")
            async def error_generator():
                yield f"Sorry, I encountered an error processing your request: {str(e)}"
            return StreamingResponse(error_generator())

    def _build_system_prompt(self, is_premium: bool = False) -> str:
        """Build system prompt based on user type"""
        if is_premium:
            return base_prompt + premium_additional_prompt
        
        return base_prompt

    async def _update_session_history(self, access_token: str, chat_message: ChatMessage):
        """Update session with new chat message"""
        db = get_database()
        await db.sessions.update_one(
            {"access_token": access_token},
            {
                "$push": {"chat_history": chat_message.dict()},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )

    async def _update_user_usage(self, user_id: ObjectId):
        """Update user's monthly usage stats"""
        db = get_database()
        await db.users.update_one(
            {"_id": user_id},
            {
                "$inc": {"usage_stats.queries_this_month": 1},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )

    async def get_chat_history(self, access_token: str, limit: int = 50) -> List[ChatMessage]:
        """Get chat history for a session"""
        session = await self.get_session(access_token)
        if not session:
            return []
        
        # Return last N messages
        return session.chat_history[-limit:] if session.chat_history else []

    async def clear_chat_history(self, access_token: str) -> bool:
        """Clear chat history for a session"""
        db = get_database()
        result = await db.sessions.update_one(
            {"access_token": access_token},
            {
                "$set": {
                    "chat_history": [],
                    "updated_at": datetime.utcnow()
                }
            }
        )
        return result.modified_count > 0
