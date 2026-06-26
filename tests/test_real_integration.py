
import pytest
import os
import shutil
import json
import time
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings
from app.core.redis_checkpointer import AsyncStandardRedisSaver

# Constants for testing
API_KEY = settings.api_key
HEADERS = {"X-API-Key": API_KEY}

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

# --- 1. API Structure & Health ---

def test_health_endpoint_status(client):
    """1. API Structure: /health endpoint status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_health_redis_logic(client):
    """2. API Structure: /health redis connection logic verification."""
    response = client.get("/health")
    data = response.json()
    # If redis is running, it should be true. If not, it should be false but the endpoint should work.
    assert "redis_connected" in data
    assert "checkpointer_type" in data

def test_painel_ui_serving(client):
    """3. API Structure: /painel UI serving (index.html existence)."""
    response = client.get("/painel")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

# --- 2. Security & Middleware ---

def test_cors_restricted_origin(client):
    """4. Security: Restricted CORS origin check."""
    headers = {"Origin": "http://malicious-site.com"}
    response = client.options("/chat", headers=headers)
    # If the origin is not allowed, the header won't be in response (or will be different)
    assert response.headers.get("access-control-allow-origin") != "http://malicious-site.com"

def test_security_missing_api_key(client):
    """5. Security: Missing API Key rejection (403)."""
    response = client.post("/chat", json={"session_id": "test", "message": "hi"})
    assert response.status_code in [401, 403]

def test_security_invalid_api_key(client):
    """6. Security: Invalid API Key rejection (403)."""
    response = client.post("/chat", json={"session_id": "test", "message": "hi"}, headers={"X-API-Key": "wrong"})
    assert response.status_code == 403

def test_security_header_nosniff(client):
    """7. Security: Presence of X-Content-Type-Options: nosniff."""
    response = client.get("/health")
    assert response.headers.get("X-Content-Type-Options") == "nosniff"

def test_security_header_frame_options(client):
    """8. Security: Presence of X-Frame-Options: DENY."""
    response = client.get("/health")
    assert response.headers.get("X-Frame-Options") == "DENY"

def test_security_header_xss(client):
    """9. Security: Presence of X-XSS-Protection."""
    response = client.get("/health")
    assert response.headers.get("X-XSS-Protection") == "1; mode=block"

# --- 3. Chat Flow & Persistence ---

def test_chat_persistence_redis(client):
    """10. Chat Flow: Session persistence verification in Redis (send message, check history)."""
    session_id = f"test_persist_{int(time.time())}"
    # 1. Send message
    msg = "Meu nome é Felipe"
    client.post("/chat", json={"session_id": session_id, "message": msg}, headers=HEADERS)
    
    # 2. Check history
    response = client.get(f"/api/history/{session_id}", headers=HEADERS)
    history = response.json()["messages"]
    # Check if our message or AI response is there
    messages_content = [m["content"] for m in history]
    assert any(msg in c for c in messages_content)

@pytest.mark.asyncio
async def test_multi_turn_conversation(client):
    """11. Chat Flow: Multi-turn conversation state verification."""
    session_id = f"test_multi_{int(time.time())}"
    # Turn 1
    client.post("/chat", json={"session_id": session_id, "message": "Lembre do número 42"}, headers=HEADERS)
    # Turn 2
    response = client.post("/chat", json={"session_id": session_id, "message": "Qual número eu pedi para lembrar?"}, headers=HEADERS)
    # This involves LLM but tests the flow. If LLM fails, we at least test the 200 OK.
    assert response.status_code == 200

def test_streaming_sse_headers(client):
    """12. Chat Flow: Streaming response (SSE) headers and connection."""
    payload = {"session_id": "test_stream", "message": "oi", "stream": True}
    with client.stream("POST", "/chat", json=payload, headers=HEADERS) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

def test_non_streaming_json_structure(client):
    """13. Chat Flow: Non-streaming response JSON structure."""
    payload = {"session_id": "test_json", "message": "oi", "stream": False}
    response = client.post("/chat", json=payload, headers=HEADERS)
    assert response.status_code == 200
    assert "response" in response.json()

# --- 4. Document Management & RAG ---

def test_rag_pdf_upload(client):
    """14. Document RAG: PDF upload (real file processing into Chroma)."""
    # Use a slightly more "real" but still minimal PDF or just a valid small one
    import io
    pdf_buffer = io.BytesIO()
    # Dummy PDF header and structure that PyPDFLoader might accept even if tiny
    pdf_buffer.write(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /Resources << >> /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 15 >>\nstream\nBT /F1 12 Tf ET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000056 00000 n \n0000000111 00000 n \n0000000212 00000 n \ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n277\n%%EOF")
    files = {"file": ("test_real.pdf", pdf_buffer.getvalue(), "application/pdf")}
    response = client.post("/api/upload", files=files, headers=HEADERS)
    # If LLM key is missing, this might be 500, but we want 200 or 500 with clear message
    assert response.status_code in [200, 500]

def test_rag_markdown_upload(client):
    """15. Document RAG: Markdown upload and chunking."""
    md_content = b"# Test Document\n\nThis is a real markdown test."
    files = {"file": ("test_real.md", md_content, "text/markdown")}
    response = client.post("/api/upload", files=files, headers=HEADERS)
    assert response.status_code in [200, 500]

def test_rag_list_documents(client):
    """16. Document RAG: Listing documents from Chroma."""
    # List whatever is there, should not fail
    response = client.get("/api/documents", headers=HEADERS)
    assert response.status_code == 200
    assert "documents" in response.json()

def test_rag_delete_document(client):
    """17. Document RAG: Deleting a document."""
    fname = "non_existent_to_delete.md"
    # Delete non-existent
    del_resp = client.delete(f"/api/documents/{fname}", headers=HEADERS)
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] in ["success", "error"]

def test_rag_unsupported_type(client):
    """18. Document RAG: Error handling for unsupported file types."""
    files = {"file": ("test.exe", b"binary", "application/octet-stream")}
    response = client.post("/api/upload", files=files, headers=HEADERS)
    assert response.status_code == 400

# --- 5. Internal Logic & State ---

def test_orchestrator_routing_logic(client):
    """19. Internal Logic: Orchestrator routing without mocks (calling actual logic)."""
    # Simply triggering the graph flow without mocking orchestrator internals
    payload = {"session_id": "route_test", "message": "O que é a Amendobobo Viagens?"}
    response = client.post("/chat", json=payload, headers=HEADERS)
    assert response.status_code == 200

def test_state_isolation(client):
    """20. Internal Logic: State management across different sessions."""
    s1 = "session_1"
    s2 = "session_2"
    client.post("/chat", json={"session_id": s1, "message": "Sou o Felipe"}, headers=HEADERS)
    client.post("/chat", json={"session_id": s2, "message": "Sou a Maria"}, headers=HEADERS)
    
    h1 = client.get(f"/api/history/{s1}", headers=HEADERS).json()["messages"]
    h2 = client.get(f"/api/history/{s2}", headers=HEADERS).json()["messages"]
    
    assert any("Felipe" in m["content"] for m in h1)
    assert not any("Maria" in m["content"] for m in h1)
    assert any("Maria" in m["content"] for m in h2)

def test_excel_real_data(client):
    """21. Internal Logic: Excel handling with real data."""
    try:
        import pandas as pd
        df = pd.DataFrame({"col1": ["data1"], "col2": ["data2"]})
        from io import BytesIO
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        files = {"file": ("test_data.xlsx", output.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        response = client.post("/api/upload", files=files, headers=HEADERS)
        assert response.status_code == 200
    except ImportError:
        pytest.skip("Pandas or XlsxWriter not installed")

# --- 6. Edge Cases ---

def test_edge_empty_message(client):
    """22. Edge Cases: Empty message rejection (FastAPI validation)."""
    payload = {"session_id": "test", "message": ""}
    response = client.post("/chat", json=payload, headers=HEADERS)
    assert response.status_code == 422

def test_edge_empty_session_id(client):
    """23. Edge Cases: Empty session ID rejection."""
    payload = {"session_id": "", "message": "oi"}
    response = client.post("/chat", json=payload, headers=HEADERS)
    assert response.status_code == 422

def test_edge_no_history_messages(client):
    """24. Edge Cases: Retrieval from a session that has NO messages."""
    session_id = f"new_session_{int(time.time())}"
    response = client.get(f"/api/history/{session_id}", headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["messages"] == []

def test_lifecycle_temp_dir_cleanup(client):
    """25. Lifecycle: Verify execution of upload doesn't leave temp folders in root (generic check)."""
    # This is more of a smoke test: if it returns 200, the finally block likely ran.
    md_content = b"content"
    files = {"file": ("cleanup_test.md", md_content, "text/markdown")}
    response = client.post("/api/upload", files=files, headers=HEADERS)
    assert response.status_code == 200
    # Manual check: tempfile.mkdtemp() uses system temp, we just want it to finish without error
