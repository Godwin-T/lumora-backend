import os
import logging
from typing import List, Dict, Any, Optional, Annotated, TypedDict, Sequence, Tuple, Literal
from datetime import datetime, timedelta
import uuid
import traceback
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from app.services.vector_store_v2 import VectorStoreService
from app.database import get_database
from app.models.session import Session, ChatMessage
from app.models.user import User
from bson import ObjectId
from fastapi.responses import StreamingResponse
import asyncio
import json

# Define the state for our graph
class GraphState(TypedDict):
    messages: Sequence[Any]  # The messages passed between nodes
    next: Optional[Annotated[Literal["retriever", "generator"], "The next node to call"]]
    retrieval_context: Optional[List[str]]  # Context from retrieval
    should_retrieve: bool  # Whether to use retrieval
    is_detail_request: bool  # Whether user is asking for more details
    session_id: str  # Session identifier
    user_id: Optional[str]  # User ID if authenticated
    query_start_time: datetime  # When the query started

# Nigerian tax and business regulation system prompt
SYSTEM_PROMPT = """You are Lumora, an AI assistant specializing in Nigerian taxes, business regulations, and related matters.

Your primary goal is to provide accurate, helpful information about:
- Nigerian tax laws and regulations
- Business registration and compliance requirements
- Corporate governance in Nigeria
- Import/export regulations
- Labor laws and employment regulations
- Financial reporting requirements
- Investment regulations and incentives

Guidelines:
1. Initially provide concise, basic responses that cover just the essential information.
2. Only provide detailed explanations when the user explicitly asks for more information or details.
3. If you don't know the answer or if the information isn't in your knowledge base, acknowledge this and avoid making up information.
4. When appropriate, mention the relevant Nigerian laws, regulations, or government agencies.
5. Maintain a professional, helpful tone.
6. Do not provide legal advice - clarify when appropriate that users should consult qualified professionals.
7. Focus only on Nigerian regulations unless explicitly asked about international comparisons.
8. At the end of your basic responses, you can add "Would you like me to explain this in more detail?" to encourage follow-up questions.
9. Format your responses using proper Markdown:
   - Use headings (## and ###) for main sections and subsections
   - Use bullet points (*) or numbered lists (1.) for listing items
   - Use **bold** for emphasis on important terms or concepts
   - Use *italics* for definitions or specialized terms
   - Use `code blocks` for specific values, rates, or figures
   - Use > blockquotes for important notes or warnings
   - Use horizontal rules (---) to separate major sections when appropriate
   - Use tables for comparing information when relevant

Remember that users rely on your information for important business decisions, so accuracy is crucial and proper formatting improves readability.
"""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("lumora_rag")

class LumoreRagChatService:
    def __init__(self):
        logger.info("Initializing LumoreRagChatService")
        self.vector_service = VectorStoreService()
        self.llm = ChatOpenAI(
            temperature=0.1,
            model="gpt-3.5-turbo",  # Changed from gpt-5-mini to a model that exists
            max_tokens=1000
        )
        
        # Rate limiting
        self.anonymous_rate_limit = int(os.getenv("ANONYMOUS_QUERIES_PER_HOUR", "10"))
        self.premium_rate_limit = int(os.getenv("PREMIUM_QUERIES_PER_HOUR", "100"))
        self.default_namespace = [os.getenv("DEFAULT_NAMESPACE", 'user_Default_User')]
        
        # Initialize the graph
        self.workflow = self._create_graph()
        
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow"""
        # Define the nodes
        
        # 1. Router node: Decides if we need to retrieve information
        def router(state: GraphState) -> Dict[str, Any]:
            """Determine if we need to retrieve information"""
            messages = state["messages"]
            
            # Get the last user message
            last_message = messages[-1].content if messages else ""
            logger.info(f"Router processing query: {last_message[:50]}...")
            
            # Check if this is a request for more details
            is_detail_request = False
            detail_keywords = ["more detail", "elaborate", "explain more", "tell me more", "additional information", 
                              "can you expand", "in depth", "further explanation", "more about", "more information"]
            
            if len(messages) > 1:  # If there's conversation history
                for keyword in detail_keywords:
                    if keyword.lower() in last_message.lower():
                        is_detail_request = True
                        logger.info("Detected request for more details")
                        break
            
            # Create a prompt to decide if retrieval is needed
            router_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are a routing assistant that determines if external information is needed to answer a user's question.
                Analyze the user's question and the conversation history to decide if retrieval of information about Nigerian taxes and business regulations is required.
                Output ONLY "retrieve" or "direct_answer" as your decision."""),
                MessagesPlaceholder(variable_name="messages"),
                HumanMessage(content="Based on the conversation, should I retrieve external information or answer directly? Reply with ONLY 'retrieve' or 'direct_answer'")
            ])
            
            # Run the router
            chain = router_prompt | self.llm | StrOutputParser()
            result = chain.invoke({"messages": messages})
            logger.info(f"Router decision: {result}")
            
            should_retrieve = "retrieve" in result.lower()
            
            # Update the state with only one next value
            return {
                "next": "retriever" if should_retrieve else "generator",
                "should_retrieve": should_retrieve,
                "is_detail_request": is_detail_request
            }
        
        # 2. Retriever node: Gets relevant information
        async def retriever(state: GraphState) -> Dict[str, Any]:
            """Retrieve relevant information from vector store"""
            messages = state["messages"]
            
            # Get the last user message
            last_message = messages[-1].content if isinstance(messages[-1], HumanMessage) else ""
            logger.info(f"Retriever processing query: {last_message[:50]}...")
            
            # Get user_id if available
            user_id = state.get("user_id")
            
            # Determine namespaces to search
            namespaces = self.default_namespace
            logger.info(f"Searching namespaces: {namespaces}")
            
            try:
                # Search for relevant documents
                relevant_docs = await self.vector_service.similarity_search(
                    query=last_message,
                    namespaces=namespaces,
                    k=5
                )
                
                # Extract the content from the documents
                retrieval_context = [doc.page_content for doc in relevant_docs]
                sources = [doc.metadata.get("source", "Unknown") for doc in relevant_docs]
                
                logger.info(f"Retrieved {len(relevant_docs)} documents")
                
                # Return the updated state
                return {
                    "retrieval_context": retrieval_context,
                    "sources": sources,
                    "next": "generator"
                }
            except Exception as e:
                logger.error(f"Error in retriever: {str(e)}")
                logger.error(traceback.format_exc())
                # If retrieval fails, continue to generator with empty context
                return {
                    "retrieval_context": [],
                    "sources": [],
                    "next": "generator"
                }
        
        # 3. Generator node: Creates the response
        def generator(state: GraphState) -> Dict[str, Any]:
            """Generate a response based on the conversation and retrieved information"""
            messages = state["messages"]
            retrieval_context = state.get("retrieval_context", []) or []
            is_detail_request = state.get("is_detail_request", False)
            
            logger.info(f"Generator processing with {len(retrieval_context)} context items")
            logger.info(f"Is detail request: {is_detail_request}")
            
            try:
                # Create the prompt
                if retrieval_context:
                    context_text = "\n\n".join(retrieval_context)
                    system_message = f"{SYSTEM_PROMPT}\n\nRelevant information:\n{context_text}"
                else:
                    system_message = SYSTEM_PROMPT
                
                # Add instruction about response verbosity based on whether this is a detail request
                if is_detail_request:
                    system_message += "\n\nThe user is asking for more detailed information. Provide a comprehensive, detailed response."
                else:
                    system_message += "\n\nProvide a concise, basic response that covers just the essential information."
                
                generator_prompt = ChatPromptTemplate.from_messages([
                    SystemMessage(content=system_message),
                    MessagesPlaceholder(variable_name="messages")
                ])
                
                # Generate the response
                chain = generator_prompt | self.llm | StrOutputParser()
                response = chain.invoke({"messages": messages})
                
                logger.info(f"Generated response: {response[:50]}...")
                
                # Add the AI message to the conversation
                new_messages = list(messages) + [AIMessage(content=response)]
                
                # Return the updated state
                return {
                    "messages": new_messages,
                    "next": END
                }
            except Exception as e:
                logger.error(f"Error in generator: {str(e)}")
                logger.error(traceback.format_exc())
                # Return a fallback response
                fallback_response = "I apologize, but I encountered an error while generating a response. Please try again."
                new_messages = list(messages) + [AIMessage(content=fallback_response)]
                return {
                    "messages": new_messages,
                    "next": END
                }
        
        # Create the graph
        workflow = StateGraph(GraphState)
        
        # Add the nodes
        workflow.add_node("router", router)
        workflow.add_node("retriever", retriever)
        workflow.add_node("generator", generator)
        
        # Define conditional edges
        workflow.add_conditional_edges(
            "router",
            lambda state: state["next"]
        )
        workflow.add_edge("retriever", "generator")
        
        # Set the entry point
        workflow.set_entry_point("router")
        
        # Compile the graph
        return workflow.compile()
    
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
    
    async def chat(
        self, 
        query: str, 
        access_token: str,
        ip_address: str,
        user_agent: str,
        user: Optional[User] = None
    ) -> Dict[str, Any]:
        """Handle chat request using LangGraph"""
        start_time = datetime.utcnow()
        logger.info(f"Chat request received: {query[:50]}...")
        
        try:
            # Get or create session
            session = await self.get_session(access_token)
            if not session:
                logger.info(f"Creating new session for token: {access_token[:8]}...")
                if user:
                    session = await self.create_premium_session(user, ip_address, user_agent, access_token)
                else:
                    session = await self.create_anonymous_session(ip_address, user_agent, access_token)
            
            # Check rate limit
            if not await self.check_rate_limit(session, user):
                logger.warning(f"Rate limit exceeded for session: {access_token[:8]}...")
                return {
                    "success": False,
                    "error": "Rate limit exceeded. Please try again later.",
                    "access_token": session.access_token
                }
            
            # Convert chat history to messages format
            messages = []
            for msg in session.chat_history[-5:]:  # Last 5 messages for context
                messages.append(HumanMessage(content=msg.user_message))
                messages.append(AIMessage(content=msg.bot_response))
            
            # Add the current query
            messages.append(HumanMessage(content=query))
            
            logger.info(f"Processing with {len(messages)} messages in context")
            
            # Initial state
            initial_state = {
                "messages": messages,
                "next": None,
                "retrieval_context": None,
                "should_retrieve": False,
                "is_detail_request": False,
                "session_id": session.access_token,
                "user_id": str(user.id) if user else None,
                "query_start_time": start_time
            }
            
            # Execute the graph
            logger.info("Executing LangGraph workflow")
            try:
                final_state = await self.workflow.ainvoke(initial_state)
                logger.info(f"Workflow completed. Final state keys: {list(final_state.keys())}")
            except Exception as e:
                logger.error(f"Error in workflow execution: {str(e)}")
                logger.error(traceback.format_exc())
                raise
            
            # Extract the response
            final_messages = final_state.get("messages", []) or []
            if not final_messages or len(final_messages) == 0:
                logger.error("No messages in final state")
                response = "I couldn't generate a response due to a system error."
            else:
                response = final_messages[-1].content if hasattr(final_messages[-1], 'content') else "I couldn't generate a response."
                logger.info(f"Response generated: {response[:50]}...")
            
            # Calculate response time
            end_time = datetime.utcnow()
            response_time_ms = int((end_time - start_time).total_seconds() * 1000)
            logger.info(f"Response time: {response_time_ms}ms")
            
            # Create chat message
            message_id = str(uuid.uuid4())
            chat_message = ChatMessage(
                message_id=message_id,
                user_message=query,
                bot_response=response,
                timestamp=end_time,
                context_used=final_state.get("sources", []),
                response_time_ms=response_time_ms
            )
            
            # Update session with new message
            await self._update_session_history(session.access_token, chat_message)
            
            # Update user usage stats if premium
            if user:
                await self._update_user_usage(user.id)
            
            return {
                "success": True,
                "answer": response,
                "sources": final_state.get("sources", []),
                "access_token": session.access_token,
                "message_id": message_id,
                "response_time_ms": response_time_ms
            }
                
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": f"Sorry, I encountered an error processing your request: {str(e)}",
                "access_token": access_token
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
        logger.info(f"Stream chat request received: {query[:50]}...")
        
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
                yield "data: Error: Rate limit exceeded. Please try again later.\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(error_generator(), media_type="text/event-stream")
        
        try:
            # Convert chat history to messages format
            messages = []
            for msg in session.chat_history[-5:]:  # Last 5 messages for context
                messages.append(HumanMessage(content=msg.user_message))
                messages.append(AIMessage(content=msg.bot_response))
            
            # Add the current query
            messages.append(HumanMessage(content=query))
            
            # Initial state
            initial_state = {
                "messages": messages,
                "next": None,
                "retrieval_context": None,
                "should_retrieve": False,
                "is_detail_request": False,
                "session_id": session.access_token,
                "user_id": str(user.id) if user else None,
                "query_start_time": start_time
            }
            
            # Execute the graph to determine if retrieval is needed and get context
            router_state = await self.workflow.get_node("router").ainvoke(initial_state)
            
            # If retrieval is needed, get the context
            retrieval_context = []
            sources = []
            if router_state.get("should_retrieve", False):
                retriever_state = await self.workflow.get_node("retriever").ainvoke({**initial_state, **router_state})
                retrieval_context = retriever_state.get("retrieval_context", []) or []
                sources = retriever_state.get("sources", []) or []
            
            # Create the prompt for streaming
            if retrieval_context:
                context_text = "\n\n".join(retrieval_context)
                system_message = f"{SYSTEM_PROMPT}\n\nRelevant information:\n{context_text}"
            else:
                system_message = SYSTEM_PROMPT
            
            generator_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_message),
                MessagesPlaceholder(variable_name="messages")
            ])
            
            # Create streaming response
            async def response_generator():
                full_response = ""
                
                # Stream the response
                async for chunk in self.llm.astream_chat(
                    generator_prompt.format_messages(messages=messages)
                ):
                    content = chunk.content if hasattr(chunk, "content") else str(chunk)
                    full_response += content
                    
                    # Stream word by word for a more natural feel
                    words = content.split()
                    for word in words:
                        await asyncio.sleep(0.07)  # Delay between words
                        yield f"data: {word} \n\n"
                
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
                    context_used=sources,
                    response_time_ms=response_time_ms
                )
                
                # Update session with new message
                await self._update_session_history(session.access_token, chat_message)
                
                # Update user usage stats if premium
                if user:
                    await self._update_user_usage(user.id)
            
            return StreamingResponse(response_generator(), media_type="text/event-stream")
            
        except Exception as e:
            print(f"Error in streaming chat: {e}")
            async def error_generator():
                yield f"data: Sorry, I encountered an error processing your request: {str(e)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(error_generator(), media_type="text/event-stream")
    
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
