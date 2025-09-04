# backend/app/services/langchain_service.py
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import (
    ConversationalRetrievalChain, 
    RetrievalQA,
    LLMChain
)
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.base import BaseCallbackHandler
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from app.services.vector_store import VectorStoreService
import tiktoken

class TokenCountingHandler(BaseCallbackHandler):
    """Callback handler to count tokens used"""
    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def on_llm_start(self, serialized, prompts, **kwargs):
        # Count input tokens
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        for prompt in prompts:
            self.prompt_tokens += len(encoding.encode(prompt))

    def on_llm_end(self, response, **kwargs):
        # Count output tokens
        if hasattr(response, 'llm_output') and response.llm_output:
            token_usage = response.llm_output.get('token_usage', {})
            self.completion_tokens += token_usage.get('completion_tokens', 0)
            self.total_tokens += token_usage.get('total_tokens', 0)

class HybridRetriever(BaseRetriever):
    """Custom retriever that combines vector search with keyword search"""
    
    def __init__(
        self, 
        vector_service: VectorStoreService,
        namespaces: List[str],
        k: int = 5,
        alpha: float = 0.7  # Weight for vector search vs keyword search
    ):
        super().__init__()
        self.vector_service = vector_service
        self.namespaces = namespaces
        self.k = k
        self.alpha = alpha

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version of get_relevant_documents"""
        return await self._get_relevant_documents_async(query)

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents using hybrid search"""
        import asyncio
        return asyncio.run(self._get_relevant_documents_async(query))

    async def _get_relevant_documents_async(self, query: str) -> List[Document]:
        """Internal async method for document retrieval"""
        # Get vector search results with scores
        vector_results = await self.vector_service.similarity_search_with_score(
            query=query,
            namespaces=self.namespaces,
            k=self.k * 2  # Get more results for reranking
        )
        
        # For now, just return vector results (BM25 would require document corpus)
        # In production, you'd combine with BM25/keyword search here
        documents = []
        for doc, score in vector_results[:self.k]:
            # Add similarity score to metadata
            doc.metadata['similarity_score'] = float(score)
            documents.append(doc)
        
        return documents

class ConversationMemoryManager:
    """Manages conversation memory with different strategies"""
    
    def __init__(self, memory_type: str = "window", k: int = 5):
        self.memory_type = memory_type
        self.k = k

    def create_memory(self, conversation_history: List[Dict[str, str]] = None) -> ConversationBufferWindowMemory:
        """Create conversation memory"""
        memory = ConversationBufferWindowMemory(
            k=self.k,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Load existing conversation history
        if conversation_history:
            for exchange in conversation_history[-self.k:]:
                if 'user_message' in exchange and 'bot_response' in exchange:
                    memory.chat_memory.add_user_message(exchange['user_message'])
                    memory.chat_memory.add_ai_message(exchange['bot_response'])
        
        return memory

class LangChainService:
    """Enhanced LangChain service with advanced retrieval and processing"""
    
    def __init__(self):
        self.vector_service = VectorStoreService()
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize LLMs with different configurations
        self.chat_llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=1500
        )
        
        self.analysis_llm = ChatOpenAI(
            model="gpt-4",  # Use GPT-4 for complex analysis
            temperature=0.0,
            max_tokens=2000
        )
        
        # Text splitters for different document types
        self.text_splitters = {
            "default": RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", " ", ""]
            ),
            "code": RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=300,
                separators=["\nclass ", "\ndef ", "\n\n", "\n", " ", ""]
            ),
            "academic": RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=250,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        }
        
        self.memory_manager = ConversationMemoryManager()

    async def create_document_processing_chain(
        self, 
        documents: List[Document],
        processing_type: str = "default"
    ) -> List[Document]:
        """Process documents with appropriate text splitting strategy"""
        
        # Choose appropriate text splitter
        splitter = self.text_splitters.get(processing_type, self.text_splitters["default"])
        
        # Enhanced document processing
        processed_docs = []
        for doc in documents:
            # Add document-level metadata
            doc.metadata.update({
                "processed_at": datetime.utcnow().isoformat(),
                "processing_type": processing_type,
                "original_length": len(doc.page_content)
            })
            
            # Split document
            chunks = splitter.split_documents([doc])
            
            # Add chunk-specific metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_id": f"{doc.metadata.get('document_id', 'unknown')}_{i}",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_length": len(chunk.page_content)
                })
                processed_docs.append(chunk)
        
        return processed_docs

    async def create_conversational_chain(
        self, 
        namespaces: List[str],
        conversation_history: List[Dict[str, str]] = None,
        chain_type: str = "stuff",
        use_advanced_retrieval: bool = True
    ) -> ConversationalRetrievalChain:
        """Create an advanced conversational retrieval chain"""
        
        # Create memory
        memory = self.memory_manager.create_memory(conversation_history)
        
        # Create retriever
        if use_advanced_retrieval:
            retriever = HybridRetriever(
                vector_service=self.vector_service,
                namespaces=namespaces,
                k=8,
                alpha=0.7
            )
        else:
            # Simple vector retriever
            vector_store = self.vector_service.get_vector_store(namespaces[0] if namespaces else None)
            retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 6,
                    "fetch_k": 20,
                    "lambda_mult": 0.7
                }
            )
        
        # Create custom prompt
        custom_prompt = self._create_conversation_prompt()
        
        # Create the chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.chat_llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=True,
            chain_type=chain_type,
            combine_docs_chain_kwargs={"prompt": custom_prompt}
        )
        
        return chain

    async def create_qa_chain(
        self, 
        namespaces: List[str],
        chain_type: str = "map_rerank"
    ) -> RetrievalQA:
        """Create a question-answering chain for single queries"""
        
        # Create hybrid retriever
        retriever = HybridRetriever(
            vector_service=self.vector_service,
            namespaces=namespaces,
            k=10
        )
        
        # Create QA prompt
        qa_prompt = self._create_qa_prompt()
        
        # Create the chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.chat_llm,
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": qa_prompt,
                "document_variable_name": "context"
            },
            verbose=True
        )
        
        return qa_chain

    async def create_multi_query_retriever(
        self, 
        namespaces: List[str]
    ) -> MultiQueryRetriever:
        """Create a multi-query retriever that generates multiple queries"""
        
        # Base retriever
        base_retriever = HybridRetriever(
            vector_service=self.vector_service,
            namespaces=namespaces,
            k=5
        )
        
        # Multi-query retriever
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self.chat_llm,
            verbose=True
        )
        
        return multi_query_retriever

    async def process_query_with_analysis(
        self, 
        query: str,
        namespaces: List[str],
        analysis_type: str = "comprehensive",
        conversation_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Process query with detailed analysis and multiple retrieval strategies"""
        
        token_handler = TokenCountingHandler()
        
        # Step 1: Query intent analysis
        intent_analysis = await self._analyze_query_intent(query)
        
        # Step 2: Choose appropriate retrieval strategy
        if intent_analysis["complexity"] == "high":
            retriever = await self.create_multi_query_retriever(namespaces)
            relevant_docs = await retriever.aget_relevant_documents(query)
        else:
            retriever = HybridRetriever(
                vector_service=self.vector_service,
                namespaces=namespaces,
                k=8
            )
            relevant_docs = await retriever.aget_relevant_documents(query)
        
        # Step 3: Create appropriate chain based on query type
        if intent_analysis["requires_conversation_context"]:
            chain = await self.create_conversational_chain(
                namespaces=namespaces,
                conversation_history=conversation_history,
                use_advanced_retrieval=True
            )
            result = await chain.acall({
                "question": query,
                "chat_history": []
            })
        else:
            chain = await self.create_qa_chain(namespaces=namespaces)
            result = await chain.acall({"query": query})
        
        # Step 4: Post-process and analyze result
        processed_result = await self._post_process_result(
            result, 
            intent_analysis, 
            relevant_docs
        )
        
        return {
            "answer": processed_result["answer"],
            "source_documents": processed_result["sources"],
            "intent_analysis": intent_analysis,
            "retrieval_metadata": {
                "num_sources": len(relevant_docs),
                "namespaces_searched": namespaces,
                "retrieval_strategy": "hybrid" if intent_analysis["complexity"] == "high" else "vector"
            },
            "tokens_used": {
                "prompt_tokens": token_handler.prompt_tokens,
                "completion_tokens": token_handler.completion_tokens,
                "total_tokens": token_handler.total_tokens
            }
        }

    async def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine the best processing approach"""
        
        intent_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            Analyze the following query and determine its characteristics:
            
            Query: {query}
            
            Please analyze and respond in this exact JSON format:
            {{
                "complexity": "low|medium|high",
                "query_type": "factual|analytical|conversational|creative",
                "requires_conversation_context": true|false,
                "domain": "general|technical|academic|personal",
                "expected_answer_length": "short|medium|long"
            }}
            
            Consider:
            - Complexity: How complex is the reasoning required?
            - Query type: What kind of question is this?
            - Context: Does it reference previous conversation?
            - Domain: What domain does this relate to?
            - Length: How detailed should the answer be?
            """
        )
        
        intent_chain = LLMChain(llm=self.chat_llm, prompt=intent_prompt)
        result = await intent_chain.arun(query=query)
        
        try:
            import json
            return json.loads(result.strip())
        except:
            # Fallback analysis
            return {
                "complexity": "medium",
                "query_type": "factual",
                "requires_conversation_context": "?" in query.lower() and len(query.split()) > 10,
                "domain": "general",
                "expected_answer_length": "medium"
            }

    async def _post_process_result(
        self, 
        result: Dict[str, Any], 
        intent_analysis: Dict[str, Any],
        source_docs: List[Document]
    ) -> Dict[str, Any]:
        """Post-process the chain result for better presentation"""
        
        # Extract answer
        answer = result.get("answer", result.get("result", ""))
        
        # Process source documents
        sources = []
        for doc in source_docs:
            source_info = {
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": doc.metadata.get("similarity_score", 0.0)
            }
            sources.append(source_info)
        
        # Sort sources by relevance
        sources.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return {
            "answer": answer,
            "sources": sources[:5],  # Top 5 sources
            "confidence": self._calculate_confidence_score(result, intent_analysis)
        }

    def _calculate_confidence_score(
        self, 
        result: Dict[str, Any], 
        intent_analysis: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for the answer"""
        
        base_confidence = 0.7
        
        # Adjust based on number of sources
        source_docs = result.get("source_documents", [])
        if len(source_docs) >= 3:
            base_confidence += 0.1
        elif len(source_docs) < 2:
            base_confidence -= 0.1
        
        # Adjust based on query complexity
        if intent_analysis.get("complexity") == "high":
            base_confidence -= 0.1
        elif intent_analysis.get("complexity") == "low":
            base_confidence += 0.1
        
        return min(max(base_confidence, 0.0), 1.0)

    def _create_conversation_prompt(self) -> PromptTemplate:
        """Create a custom prompt for conversational retrieval"""
        
        template = """
        You are a helpful AI assistant. Use the following pieces of context to answer the user's question. 
        If you don't know the answer based on the context provided, say that you don't know, don't make up an answer.
        
        When referencing information from the context, be specific about which source you're drawing from when possible.
        If the question requires information not in the context, clearly state what information is missing.
        
        Context:
        {context}
        
        Chat History:
        {chat_history}
        
        Question: {question}
        
        Helpful Answer:"""
        
        return PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=template
        )

    def _create_qa_prompt(self) -> PromptTemplate:
        """Create a custom prompt for Q&A"""
        
        template = """
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
        
        Be concise but comprehensive. Include relevant details from the context.
        If multiple sources support your answer, synthesize the information effectively.
        
        {context}
        
        Question: {question}
        
        Answer:"""
        
        return PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )

    async def create_document_summary_chain(self) -> LLMChain:
        """Create a chain for document summarization"""
        
        summary_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Summarize the following text in a clear and concise manner. 
            Include the main points and key insights. Keep the summary informative but brief.
            
            Text to summarize:
            {text}
            
            Summary:"""
        )
        
        return LLMChain(llm=self.chat_llm, prompt=summary_prompt)

    async def extract_document_metadata(self, document: Document) -> Dict[str, Any]:
        """Extract enhanced metadata from document"""
        
        metadata_prompt = PromptTemplate(
            input_variables=["content"],
            template="""
            Analyze the following document content and extract key metadata.
            
            Content: {content}
            
            Please provide a JSON response with the following structure:
            {{
                "main_topics": ["topic1", "topic2", "topic3"],
                "document_type": "report|article|manual|other",
                "key_entities": ["entity1", "entity2"],
                "summary": "brief summary",
                "complexity_level": "beginner|intermediate|advanced"
            }}
            """
        )
        
        metadata_chain = LLMChain(llm=self.chat_llm, prompt=metadata_prompt)
        
        # Use first 2000 characters for metadata extraction
        content_sample = document.page_content[:2000]
        result = await metadata_chain.arun(content=content_sample)
        
        try:
            import json
            extracted_metadata = json.loads(result.strip())
            
            # Merge with existing metadata
            enhanced_metadata = document.metadata.copy()
            enhanced_metadata.update(extracted_metadata)
            enhanced_metadata["metadata_extracted_at"] = datetime.utcnow().isoformat()
            
            return enhanced_metadata
            
        except Exception as e:
            print(f"Error extracting metadata: {e}")
            return document.metadata

# Example usage and integration helpers
class ChainManager:
    """Manages different chain types and configurations"""
    
    def __init__(self):
        self.langchain_service = LangChainService()
        self.active_chains = {}

    async def get_or_create_chain(
        self, 
        chain_id: str,
        chain_type: str,
        namespaces: List[str],
        **kwargs
    ):
        """Get existing chain or create new one"""
        
        if chain_id in self.active_chains:
            return self.active_chains[chain_id]
        
        if chain_type == "conversational":
            chain = await self.langchain_service.create_conversational_chain(
                namespaces=namespaces,
                **kwargs
            )
        elif chain_type == "qa":
            chain = await self.langchain_service.create_qa_chain(
                namespaces=namespaces,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown chain type: {chain_type}")
        
        self.active_chains[chain_id] = chain
        return chain

    def cleanup_inactive_chains(self, max_chains: int = 50):
        """Clean up old chains to prevent memory leaks"""
        if len(self.active_chains) > max_chains:
            # Remove oldest chains (simple FIFO)
            chains_to_remove = list(self.active_chains.keys())[:-max_chains]
            for chain_id in chains_to_remove:
                del self.active_chains[chain_id]