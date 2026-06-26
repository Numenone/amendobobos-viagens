from fastapi import APIRouter, HTTPException, Request, Depends, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from app.agents.orchestrator import get_graph
from app.core.config import settings
import json
import os
import shutil
import tempfile
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Autenticação X-API-Key desativada — auto_error=False evita 403 quando o header está ausente.
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    # Verificação desativada: endpoints são públicos. O header é ignorado se enviado.
    return api_key

class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="Unique identifier for the session to maintain state.")
    message: str = Field(..., min_length=1, description="The user's query or message.")
    stream: bool = Field(default=False, description="Set to true to receive an SSE stream of the response.")
    llm_model: str = Field(default="openai/gpt-4o-mini", description="LLM model identifier.")
    llm_gateway: str = Field(default="https://openrouter.ai/api/v1", description="LLM Gateway Base URL.")
    llm_api_key: str = Field(default="", description="API Key for the chosen LLM gateway. If empty, server defaults are used.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": "test_session_123",
                    "message": "Qual é a política de bagagem?",
                    "llm_model": "openai/gpt-4o-mini",
                    "llm_gateway": "https://openrouter.ai/api/v1",
                    "llm_api_key": "sk-or-your-key-here",
                    "stream": False
                }
            ]
        }
    }

class ChatResponse(BaseModel):
    response: str = Field(..., description="The AI's generated response (only used when stream=false).")

class UploadResponse(BaseModel):
    status: str = Field(..., description="Status of the upload (success/error).")
    filename: str = Field(..., description="Name of the uploaded file.")
    chunks: int = Field(..., description="Number of text chunks created from the document.")
    message: str = Field(..., description="Human-readable status message.")

class DocumentListResponse(BaseModel):
    documents: list[str] = Field(..., description="List of unique document filenames ingested into the vector store.")
    error: str = Field(default=None, description="Optional error message if listing fails.")

class DeleteResponse(BaseModel):
    status: str = Field(..., description="Status of the deletion (success/error).")
    message: str = Field(..., description="Human-readable status message.")

class HistoryMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (me/ai).")
    content: str = Field(..., description="The content of the message.")

class HistoryResponse(BaseModel):
    messages: list[HistoryMessage] = Field(..., description="List of formatted chat history messages.")

async def generate_chat_stream(request: ChatRequest, fastapi_req: Request, checkpointer):
    """Generator for Server-Sent Events (SSE)."""
    logger.info(f"event=sse_stream_started session_id={request.session_id}")
    graph = get_graph(checkpointer)
    
    # Get cached components from app state
    retriever = getattr(fastapi_req.app.state, "retriever", None)
    embeddings = getattr(fastapi_req.app.state, "embeddings", None)
    
    config = {
        "configurable": {
            "thread_id": request.session_id,
            "retriever": retriever,
            "embeddings": embeddings
        }
    }
    
    
    # 1. IMMEDIATE FEEDBACK: Handled by frontend to avoid duplication in final message
    try:
        initial_state = {
            "messages": [HumanMessage(content=request.message)],
            "llm_model": request.llm_model,
            "llm_gateway": request.llm_gateway,
            "llm_api_key": request.llm_api_key
        }
        async for event in graph.astream_events(
            initial_state,
            config=config,
            version="v2"
        ):
            # logger.debug(f"event={event['event']} name={event.get('name')}")
            
            # Filter: only stream from agent nodes
            if event["event"] == "on_chat_model_stream":
                node_name = event.get("metadata", {}).get("langgraph_node", "")
                if node_name in ["faq_agent", "search_agent", "agent"]:
                    chunk = event["data"]["chunk"].content
                    if chunk:
                        data = json.dumps({"content": chunk})
                        yield f"data: {data}\n\n"
            
            # Diagnostic logs
            elif event["event"] == "on_node_start":
                logger.debug(f"node_started={event.get('name') or event.get('metadata', {}).get('langgraph_node')}")
        
        logger.info(f"event=sse_stream_completed session_id={request.session_id}")
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"event=sse_stream_error session_id={request.session_id} error={str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

@router.post(
    "/chat", 
    response_model=ChatResponse,
    summary="Interagir com os Agentes de Viagem (FAQ & Search)",
)
async def chat_endpoint(request: ChatRequest, fastapi_req: Request, api_key: str = Depends(verify_api_key)):
    logger.info(f"chat_endpoint: session_id={request.session_id} message='{request.message[:30]}...'")
    checkpointer = getattr(fastapi_req.app.state, "checkpointer", None)
    logger.info(f"chat_endpoint: session_id={request.session_id} cp_type={type(checkpointer)}")
    
    # Get cached components from app state for non-streaming case
    retriever = getattr(fastapi_req.app.state, "retriever", None)
    embeddings = getattr(fastapi_req.app.state, "embeddings", None)

    try:
        if request.stream:
            return StreamingResponse(
                generate_chat_stream(request, fastapi_req, checkpointer),
                media_type="text/event-stream"
            )

        logger.info(f"event=chat_invocation_started session_id={request.session_id}")
        graph = get_graph(checkpointer)
        
        config = {
            "configurable": {
                "thread_id": request.session_id,
                "retriever": retriever,
                "embeddings": embeddings
            }
        }
        
        # Invoke graph asynchronously
        initial_state = {
            "messages": [HumanMessage(content=request.message)],
            "llm_model": request.llm_model,
            "llm_gateway": request.llm_gateway,
            "llm_api_key": request.llm_api_key
        }
        result = await graph.ainvoke(
            initial_state,
            config=config
        )
        
        # The last message is the AI response
        final_message = result["messages"][-1].content
        logger.info(f"event=chat_invocation_completed session_id={request.session_id}")
        return ChatResponse(response=final_message)
    except Exception as e:
        logger.error(f"event=chat_invocation_error session_id={request.session_id} error={str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
@router.post(
    "/api/upload", 
    response_model=UploadResponse,
    summary="Enviar documento para alimentar a IA (RAG)",
    description="Processa um arquivo (PDF, Markdown ou Excel), divide em pedaços (chunks) e os insere no banco de vetores ChromaDB para consulta posterior pelos agentes."
)
async def upload_document(
    fastapi_req: Request,
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    """Processa um arquivo (PDF ou MD) e adiciona ao banco de vetores."""
    logger.info(f"event=upload_started filename={file.filename} content_type={file.content_type}")
    
    if not file.filename.lower().endswith(('.pdf', '.md', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Apenas arquivos .pdf, .md, .xlsx e .xls são suportados.")

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Import standard RAG tools
        from langchain_community.document_loaders import PyPDFLoader, TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_chroma import Chroma
        from langchain_core.documents import Document

        documents = []
        if file.filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
        elif file.filename.lower().endswith(".md"):
            loader = TextLoader(temp_path, encoding="utf-8")
            documents = loader.load()
        elif file.filename.lower().endswith((".xlsx", ".xls")):
            try:
                import pandas as pd
                # Read all sheets or just the first one? Usually first one is enough for simple RAG
                df = pd.read_excel(temp_path)
                # Convert to CSV string which is dense and easy for LLM to parse
                csv_content = df.to_csv(index=False)
                documents = [Document(page_content=csv_content, metadata={"source": file.filename})]
            except ImportError:
                raise HTTPException(status_code=500, detail="Bibliotecas para Excel não instaladas no servidor.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n### ", "\n## ", "\n# ", "\n\n", "\n", " ", ""]
        )
        docs = text_splitter.split_documents(documents)

        embeddings = getattr(fastapi_req.app.state, "embeddings", None)
        persist_directory = "data/chroma"

        # Update Chroma
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        
        # Explicitly ensure metadata source is set for all chunks if not already
        # Chroma.from_documents usually takes it from docs, but let's be safe for listing
        if hasattr(vectorstore, "_collection"):
            # This is a bit internal but helps ensure we can list them
            logger.debug(f"event=chroma_ingestion_verified filename={file.filename}")

        # Immediate refresh of the retriever in app state
        fastapi_req.app.state.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        logger.info(f"event=upload_success filename={file.filename} chunks={len(docs)}")
        return {
            "status": "success",
            "filename": file.filename,
            "chunks": len(docs),
            "message": f"Documento '{file.filename}' processado e integrado com sucesso!"
        }

    except Exception as e:
        logger.error(f"event=upload_error filename={file.filename} error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao processar arquivo: {str(e)}")
    finally:
        shutil.rmtree(temp_dir)

@router.get(
    "/api/documents", 
    response_model=DocumentListResponse,
    summary="Listar documentos ingeridos",
    description="Retorna uma lista de nomes de arquivos únicos que foram processados e estão disponíveis no banco de vetores."
)
async def list_documents(fastapi_req: Request, api_key: str = Depends(verify_api_key)):
    """Retorna uma lista de nomes de arquivos únicos presentes no ChromaDB."""
    try:
        from langchain_chroma import Chroma
        embeddings = getattr(fastapi_req.app.state, "embeddings", None)
        persist_directory = "data/chroma"
        
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        
        # Get all metadata from the collection
        collection = vectorstore._collection
        get_result = collection.get(include=['metadatas'])
        metadata = get_result['metadatas'] if 'metadatas' in get_result else []
        
        # Extract unique sources
        sources = set()
        for m in metadata:
            if m and 'source' in m:
                # Handle potential path separators in source
                source_name = os.path.basename(m['source'])
                sources.add(source_name)
        
        return {"documents": sorted(list(sources))}
    except Exception as e:
        logger.error(f"event=list_documents_error error={str(e)}")
        return {"documents": [], "error": str(e)}

@router.delete(
    "/api/documents/{filename}", 
    response_model=DeleteResponse,
    summary="Excluir um documento",
    description="Remove todos os vetores associados a um arquivo específico do banco de vetores ChromaDB."
)
async def delete_document(filename: str, fastapi_req: Request, api_key: str = Depends(verify_api_key)):
    """Remove todos os chunks de um documento específico do ChromaDB."""
    try:
        from langchain_chroma import Chroma
        embeddings = getattr(fastapi_req.app.state, "embeddings", None)
        persist_directory = "data/chroma"
        
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        
        # Find IDs of documents with this source
        collection = vectorstore._collection
        # We search by source matching filename (basename)
        all_data = collection.get(include=['metadatas'])
        ids_to_delete = []
        for i, m in enumerate(all_data['metadatas']):
            if m and os.path.basename(m.get('source', '')) == filename:
                ids_to_delete.append(all_data['ids'][i])
        
        if not ids_to_delete:
            return {"status": "error", "message": f"Documento '{filename}' não encontrado."}
            
        collection.delete(ids=ids_to_delete)
        
        # Refresh retriever in app state
        fastapi_req.app.state.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        logger.info(f"event=document_deleted filename={filename} count={len(ids_to_delete)}")
        return {"status": "success", "message": f"Documento '{filename}' excluído com sucesso."}
    except Exception as e:
        logger.error(f"event=delete_document_error filename={filename} error={str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/api/history/{session_id}",
    response_model=HistoryResponse,
    summary="Recuperar o histórico do chat",
    description="Busca todas as mensagens trocadas em uma sessão específica (thread_id) do checkpointer do LangGraph."
)
async def get_chat_history(session_id: str, fastapi_req: Request, api_key: str = Depends(verify_api_key)):
    checkpointer = getattr(fastapi_req.app.state, "checkpointer", None)
    logger.info(f"get_chat_history: session_id={session_id} cp_type={type(checkpointer)}")
    try:
        config = {"configurable": {"thread_id": session_id}}
        state_tuple = await checkpointer.aget_tuple(config)
        
        if not state_tuple:
            logger.info(f"event=history_not_found session_id={session_id}")
            return {"messages": []}
            
        checkpoint = state_tuple.checkpoint
        channel_values = checkpoint.get("channel_values", {})
        messages = channel_values.get("messages", [])
        
        logger.info(f"event=history_found session_id={session_id} msg_count={len(messages)} channels={list(channel_values.keys())}")
        
        # Format messages for frontend
        formatted_messages = []
        for msg in messages:
            content = ""
            role = "ai"
            msg_type = ""
            
            if hasattr(msg, "content"):
                content = msg.content
                msg_type = msg.type
            elif isinstance(msg, dict):
                content = msg.get("content", "")
                msg_type = msg.get("type", "")
            
            # Filter: only show human and AI messages, skip tools/system
            if msg_type == "human":
                role = "me"
            elif msg_type == "ai":
                role = "ai"
            else:
                continue # Skip ToolMessage, etc.

            if content:
                formatted_messages.append({"role": role, "content": content})
                
        return {"messages": formatted_messages}
    except Exception as e:
        logger.error(f"event=history_retrieval_error session_id={session_id} error={str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
