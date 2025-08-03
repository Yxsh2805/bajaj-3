import os
import time
import logging
import hashlib
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import requests
from bs4 import BeautifulSoup
import PyPDF2
from docx import Document as DocxDocument
import email
import io
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel

# RAG imports - simplified to avoid init issues
from langchain_together import ChatTogether, TogetherEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Expected Bearer token for authentication
EXPECTED_TOKEN = "5aa05ad358e859e92978582cde20423149f28beb49da7a2bbb487afa8fce1be8"

# ----- Request/Response Models -----
class QuestionRequest(BaseModel):
    documents: str
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]

# ----- Simple Vector Store -----
class SimpleVectorStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.documents = []
        self.vectors = []
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store"""
        for doc in documents:
            vector = self.embeddings.embed_query(doc.page_content)
            self.documents.append(doc)
            self.vectors.append(vector)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents"""
        if not self.vectors:
            return []
        
        query_vector = self.embeddings.embed_query(query)
        similarities = cosine_similarity([query_vector], self.vectors)[0]
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        return [self.documents[i] for i in top_indices if i < len(self.documents)]

# ----- Document Loader -----
def load_document_content(url: str) -> List[Document]:
    """Load content from various document formats"""
    try:
        response = requests.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        
        url_lower = url.lower()
        content_type = response.headers.get('content-type', '').lower()
        
        # Handle PDF files
        if url_lower.endswith('.pdf') or 'pdf' in content_type:
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return [Document(page_content=text.strip(), metadata={"source": url, "type": "pdf"})]
        
        # Handle DOCX files
        elif url_lower.endswith('.docx') or 'wordprocessingml' in content_type:
            docx_file = io.BytesIO(response.content)
            doc = DocxDocument(docx_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return [Document(page_content=text.strip(), metadata={"source": url, "type": "docx"})]
        
        # Handle EML files
        elif url_lower.endswith('.eml') or 'message/rfc822' in content_type:
            eml_content = response.content.decode('utf-8', errors='ignore')
            msg = email.message_from_string(eml_content)
            
            text = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        text += part.get_payload(decode=True).decode('utf-8', errors='ignore')
            else:
                text = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            
            return [Document(page_content=text.strip(), metadata={"source": url, "type": "eml"})]
        
        # Handle HTML/Text files (default)
        else:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            text = ' '.join(text.split())
            
            return [Document(page_content=text, metadata={"source": url, "type": "html"})]
        
    except Exception as e:
        logger.error(f"Failed to load document {url}: {e}")
        raise

# ----- RAG Engine -----
class RailwayRAGEngine:
    def __init__(self):
        self.chat_model = None
        self.embeddings = None
        self.text_splitter = None
        self.initialized = False
        self.document_cache: Dict[str, Any] = {}
        self.max_cache_size = 2
    
    def _get_url_hash(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()[:12]
    
    def initialize(self):
        """Initialize RAG components"""
        if self.initialized:
            return
            
        logger.info("Initializing Railway RAG engine...")
        
        try:
            # Set environment variables
            os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY", "deb14836869b48e01e1853f49381b9eb7885e231ead3bc4f6bbb4a5fc4570b78")
            
            # Initialize embeddings first
            self.embeddings = TogetherEmbeddings(
                model="BAAI/bge-base-en-v1.5"
            )
            
            # Test embeddings
            logger.info("Testing embeddings...")
            test_embedding = self.embeddings.embed_query("test")
            logger.info(f"Embeddings working - dimension: {len(test_embedding)}")

            # Initialize chat model
            self.chat_model = ChatTogether(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                temperature=0,
                max_tokens=3000
            )

            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=80,
                separators=["\n\n", "\n", ". ", " ", ""]
            )

            self.initialized = True
            logger.info("Railway RAG engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {str(e)}")
            raise

    def _simple_rag_query(self, vectorstore: SimpleVectorStore, query: str) -> str:
        """Simple RAG query without complex prompt templates"""
        try:
            # Retrieve relevant documents
            docs = vectorstore.similarity_search(query, k=5)
            context = " ".join([doc.page_content for doc in docs])[:3000]
            
            # Create a simple prompt
            system_prompt = """You are an expert document assistant. Answer questions concisely and accurately.

CRITICAL FORMAT: Input questions are separated by " | ". Output answers MUST be separated by " | " in the same order.

Guidelines:
- Direct, concise answers
- Lead with key information
- Cite specific document content when available
- If unsure, state limitations clearly
- Maintain exact order and use " | " separator between answers"""

            human_prompt = f"""Questions: {query}
Context: {context}"""

            # Use chat model directly
            from langchain_core.messages import HumanMessage, SystemMessage
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = self.chat_model.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"RAG query error: {e}")
            raise

    def _load_and_process_document(self, url: str) -> tuple:
        """Load and process document"""
        if url in self.document_cache:
            logger.info(f"Using cached document for {url}")
            return self.document_cache[url]
        
        logger.info(f"Loading document: {url}")
        start_time = time.time()
        
        try:
            docs = load_document_content(url)
            chunks = self.text_splitter.split_documents(docs)
            
            load_time = time.time() - start_time
            logger.info(f"Document loaded and chunked in {load_time:.2f}s ({len(chunks)} chunks)")
            
            if len(self.document_cache) >= self.max_cache_size:
                oldest_key = next(iter(self.document_cache))
                del self.document_cache[oldest_key]
            
            self.document_cache[url] = (docs, chunks)
            return docs, chunks
            
        except Exception as e:
            logger.error(f"Failed to load document {url}: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to load document: {str(e)}")

    def _create_vectorstore(self, url: str, chunks: List) -> SimpleVectorStore:
        """Create vectorstore"""
        logger.info(f"Creating vectorstore for {url}")
        start_time = time.time()
        
        try:
            vectorstore = SimpleVectorStore(self.embeddings)
            
            batch_size = 20
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                vectorstore.add_documents(batch)
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
            
            creation_time = time.time() - start_time
            logger.info(f"Vectorstore created in {creation_time:.2f}s")
            
            return vectorstore
            
        except Exception as e:
            logger.error(f"Vectorstore creation error: {e}")
            raise HTTPException(status_code=500, detail="Failed to create vectorstore")

    def process_document_questions(self, url: str, questions: List[str]) -> List[str]:
        """Process document and answer questions"""
        if not self.initialized:
            raise RuntimeError("RAG engine not initialized")
        
        total_start_time = time.time()
        
        try:
            docs, chunks = self._load_and_process_document(url)
            vectorstore = self._create_vectorstore(url, chunks)
            
            logger.info(f"Processing {len(questions)} questions in batch...")
            batch_query = " | ".join(questions)
            
            query_start_time = time.time()
            batch_result = self._simple_rag_query(vectorstore, batch_query)
            query_time = time.time() - query_start_time
            
            logger.info(f"LLM query completed in {query_time:.2f}s")
            
            answers = [answer.strip() for answer in batch_result.split(" | ")]
            
            if len(answers) != len(questions):
                logger.warning(f"Answer count mismatch: {len(questions)} questions, {len(answers)} answers")
                while len(answers) < len(questions):
                    answers.append("Unable to generate answer for this question.")
                answers = answers[:len(questions)]
            
            total_time = time.time() - total_start_time
            logger.info(f"Total processing time: {total_time:.2f}s")
            
            return answers

        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise

# Global RAG engine instance
rag_engine = RailwayRAGEngine()

# Token Verifier
def verify_token(authorization: Optional[str] = Header(None)):
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid format")
    
    token = authorization.split("Bearer ")[-1]
    if token != EXPECTED_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid Bearer token")

# Lifespan Management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        rag_engine.initialize()
        logger.info("Railway application startup completed successfully")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        logger.info("Continuing with limited functionality...")
    
    yield
    
    # Shutdown
    logger.info("Application shutting down...")

# FastAPI App
app = FastAPI(title="Railway RAG API", version="1.0.0", lifespan=lifespan)

@app.post("/hackrx/run", response_model=AnswerResponse)
async def ask_questions(
    request: QuestionRequest,
    authorization: str = Depends(verify_token)
):
    """Main endpoint for processing documents and answering questions"""
    try:
        logger.info(f"Received request with {len(request.questions)} questions")

        if not request.documents.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid document URL")
        if not request.questions:
            raise HTTPException(status_code=400, detail="Questions list is empty")

        answers = rag_engine.process_document_questions(request.documents, request.questions)

        logger.info(f"Successfully processed {len(answers)} answers")
        return {"answers": answers}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Internal processing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "rag_initialized": rag_engine.initialized,
        "cached_documents": len(rag_engine.document_cache),
        "timestamp": datetime.now().isoformat()
    }

# Railway startup
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
