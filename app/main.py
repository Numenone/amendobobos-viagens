import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
# from langgraph.checkpoint.redis.aio import AsyncRedisSaver
import redis.asyncio as redis
from app.api.endpoints import router as api_router
from app.core.config import settings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os

# Setup structured logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Connect to Redis using asyncio client
    try:
        from app.core.redis_checkpointer import AsyncStandardRedisSaver
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import SupabaseVectorStore
        from langchain_chroma import Chroma
        from supabase.client import Client, create_client
        
        # 1. Initialize Redis Checkpointer
        connection = redis.from_url(settings.redis_url, decode_responses=False)
        await connection.ping()
        app.state.checkpointer = AsyncStandardRedisSaver(connection)
        logger.info(f"Successfully connected to Standard Redis at {settings.redis_url}")
        
        # 2. Cache Embeddings (Heavy initialization)
        embeddings = OpenAIEmbeddings(
            model="openai/text-embedding-3-small",
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        app.state.embeddings = embeddings
        
        # 3. Cache VectorStore/Retriever
        retriever = None
        if settings.supabase_url and settings.supabase_service_key:
            from supabase.client import Client, create_client
            from langchain_community.vectorstores import SupabaseVectorStore
            supabase: Client = create_client(settings.supabase_url, settings.supabase_service_key)
            vectorstore = SupabaseVectorStore(
                client=supabase,
                embedding=embeddings,
                table_name="documents",
                query_name="match_documents"
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            logger.info("Retriever initialized (Supabase)")
        elif os.path.exists("data/chroma"):
            try:
                vectorstore = Chroma(persist_directory="data/chroma", embedding_function=embeddings)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                logger.info("Retriever initialized (Chroma)")
            except Exception as chroma_err:
                logger.error(f"Failed to initialize Chroma: {chroma_err}")
        
        app.state.retriever = retriever
        
        yield
        
        if hasattr(app.state, "checkpointer") and isinstance(app.state.checkpointer, AsyncStandardRedisSaver):
            logger.info("Closing Redis connection.")
            await connection.aclose()
            
    except Exception as e:
        logger.warning(f"Error during lifespan initialization: {e}")
        from langgraph.checkpoint.memory import MemorySaver
        app.state.checkpointer = MemorySaver()
        app.state.embeddings = None
        app.state.retriever = None
        yield

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse
import os

from app.core.redis_checkpointer import AsyncStandardRedisSaver
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall system status.")
    redis_connected: bool = Field(..., description="Whether the Redis checkpointer is successfully connected.")
    checkpointer_type: str = Field(..., description="The class type of the active checkpointer.")

app = FastAPI(
    title="Amendobobo Viagens - Multi-agent Travel Chatbot",
    description="API for the Amendobobo Viagens technical test featuring LangGraph agents. Integrada com proteção API Key, CORS restrito e Security Headers contra ataques comuns (XSS, MIME Snipping e Framing).",
    version="1.0.0",
    lifespan=lifespan
)

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Verificar integridade do sistema",
    description="Retorna o status atual da API e confirma se o checkpointer do Redis está operacional."
)
async def health_check():
    """Confirma se o serviço está online e conectado ao Redis."""
    checkpointer = getattr(app.state, "checkpointer", None)
    return {
        "status": "ok",
        "redis_connected": isinstance(checkpointer, AsyncStandardRedisSaver),
        "checkpointer_type": str(type(checkpointer))
    }

@app.get(
    "/painel",
    summary="Acessar o Painel de Controle (Frontend)",
    description="Serve o arquivo index.html estático que contém a interface do chat e dashboard."
)
async def get_painel():
    """Retorna a interface visual do Amendobobo Viagens."""
    static_file = os.path.join(os.path.dirname(__file__), "static", "index.html")
    return FileResponse(static_file)

# Include API Routes
app.include_router(api_router)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"event=validation_error method={request.method} path={request.url.path} detail={exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "message": "Erro de validação de tipagem forte (Pydantic)."},
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"event=internal_error method={request.method} path={request.url.path} error={str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "message": str(exc)},
    )
    
import time
from fastapi.middleware.cors import CORSMiddleware

# Security: Restrict CORS
# Note: You should specify your exact frontend domain instead of "*" in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://seu-front-end.vercel.app"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.middleware("http")
async def security_headers(request: Request, call_next):
    # Extracted from standard secure application configurations
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        f"event=request_processed method={request.method} path={request.url.path} "
        f"status={response.status_code} duration={process_time:.4f}s"
    )
    return response

    return response
