# ✈️ Amendobobo Viagens — Chatbot Multi‑Agente de Viagens

Assistente de viagens conversacional construído com **FastAPI + LangGraph**, usando uma
arquitetura **multi‑agente** com roteamento inteligente, memória de sessão persistente
(Redis), busca em tempo real na web (Tavily) e RAG sobre documentos internos (ChromaDB).

O assistente fala **Português (PT‑BR)**, é cordial, lembra do usuário ao longo da conversa
e possui guardrails contra prompt injection e perguntas fora do domínio de viagens.

---

## 📑 Índice

- [Visão geral da arquitetura](#-visão-geral-da-arquitetura)
- [O que você precisa (pré‑requisitos)](#-o-que-você-precisa-pré-requisitos)
- [Configuração das chaves (.env)](#-configuração-das-chaves-env)
- [Como iniciar](#-como-iniciar)
  - [Opção A — Docker Compose (recomendado)](#opção-a--docker-compose-recomendado)
  - [Opção B — Execução local](#opção-b--execução-local)
- [Como usar](#-como-usar)
  - [Painel web](#painel-web)
  - [Documentação interativa (Swagger)](#documentação-interativa-swagger)
- [Referência da API](#-referência-da-api)
- [Gerenciamento de documentos (RAG)](#-gerenciamento-de-documentos-rag)
- [Memória e sessões](#-memória-e-sessões)
- [Segurança](#-segurança)
- [Testes](#-testes)
- [Solução de problemas](#-solução-de-problemas)
- [Estrutura do projeto](#-estrutura-do-projeto)

---

## 🧠 Visão geral da arquitetura

O cérebro do sistema é um **grafo LangGraph** com um roteador que classifica cada
mensagem e a despacha para o agente especialista correto:

```
                ┌──────────────┐
   Usuário ───▶ │    ROUTER    │  (classifica a intenção)
                └──────┬───────┘
            ┌──────────┴──────────┐
            ▼                     ▼
     ┌─────────────┐       ┌──────────────┐
     │  FAQ AGENT  │       │ SEARCH AGENT │
     │  (RAG /     │       │ (ReAct +     │
     │   ChromaDB) │       │  Tavily web) │
     └─────────────┘       └──────────────┘
            │                     │
            └──────────┬──────────┘
                       ▼
                 Resposta (PT‑BR)
       (estado salvo no Redis Checkpointer)
```

| Componente | Responsabilidade |
|------------|------------------|
| **Router** | Decide entre `faq_agent` (políticas internas / off‑topic) e `search_agent` (dados em tempo real). Usa *structured output*. |
| **FAQ Agent** | Responde sobre políticas internas (bagagem, check‑in, documentação) via **RAG** sobre o ChromaDB. |
| **Search Agent** | Agente **ReAct** que usa a ferramenta **TavilySearch** para clima, voos, hotéis e dados ao vivo. |
| **Redis Checkpointer** | Persiste o estado de cada conversa por `session_id` (memória entre mensagens). |
| **Guardian Protocol** | Camada de prompt que impõe domínio de viagens, cordialidade, personalização e defesa contra injeção. |

### Stack
- **FastAPI** + **Uvicorn** — API e SSE (streaming)
- **LangGraph** / **LangChain** — orquestração multi‑agente
- **OpenRouter** — gateway de LLM (GPT‑4o‑mini por padrão, troca de modelo via request)
- **Tavily** — busca na web em tempo real
- **ChromaDB** — vector store local para RAG (ou **Supabase/pgvector** em nuvem)
- **Redis Stack** — checkpointer de sessões
- **Docker Compose** — orquestração dos serviços

---

## ✅ O que você precisa (pré‑requisitos)

### Para rodar com Docker (recomendado)
- **Docker** e **Docker Compose** instalados (`docker --version`, `docker compose version`).

### Para rodar localmente (sem Docker)
- **Python 3.11** (o projeto roda em 3.11; o Dockerfile fixa essa versão).
- Uma instância de **Redis** acessível (local ou Redis Stack).

### Chaves de API (obrigatórias para o chat funcionar de verdade)

| Chave | Para quê | Onde obter |
|-------|----------|------------|
| `OPENROUTER_API_KEY` | LLM (raciocínio dos agentes) e *embeddings* do RAG | https://openrouter.ai |
| `TAVILY_API_KEY` | Busca na web em tempo real (agente de busca) | https://tavily.com |

> ⚠️ **Sem as chaves reais a aplicação sobe normalmente** (API, Redis, painel e health
> funcionam), **mas o chat não gera respostas** — os agentes precisam do LLM. Use chaves
> de verdade para testar a conversa de ponta a ponta.

---

## 🔐 Configuração das chaves (.env)

Copie o exemplo e preencha suas chaves:

```bash
cp .env.example .env
```

Edite o `.env`:

```dotenv
# Chaves de IA (obrigatórias para o chat gerar respostas)
OPENROUTER_API_KEY=sk-or-sua-chave-aqui
TAVILY_API_KEY=tvly-sua-chave-aqui

# Redis (no Docker Compose este valor é sobrescrito para redis://redis:6379)
REDIS_URL=redis://localhost:6379

# Nível de log
LOG_LEVEL=INFO
```

Variáveis suportadas (ver [`app/core/config.py`](app/core/config.py)):

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `OPENROUTER_API_KEY` | `""` | Chave do gateway de LLM/embeddings. |
| `TAVILY_API_KEY` | `""` | Chave da busca web. |
| `REDIS_URL` | `redis://localhost:6379` | URL do Redis (checkpointer). |
| `SUPABASE_URL` | `""` | (Opcional) usa Supabase/pgvector em vez do ChromaDB local. |
| `SUPABASE_SERVICE_KEY` | `""` | (Opcional) chave de serviço do Supabase. |
| `LOG_LEVEL` | `INFO` | Nível de logging. |
| `API_KEY` | `blis_secret_token_123` | Token de API (a verificação está desativada por padrão). |

---

## 🚀 Como iniciar

### Opção A — Docker Compose (recomendado)

Sobe a aplicação **e** o Redis Stack de uma vez, sem instalar nada de Python na máquina.

```bash
# 1. Garanta que o .env está preenchido
# 2. Build + start em background
docker compose up -d --build

# Acompanhar logs
docker compose logs -f web

# Parar tudo
docker compose down
```

Serviços iniciados:
- **web** → API em `http://localhost:8000`
- **redis** → Redis Stack em `localhost:6379` (com healthcheck)

Verifique a saúde:

```bash
curl http://localhost:8000/health
# {"status":"ok","redis_connected":true,"checkpointer_type":"...AsyncStandardRedisSaver..."}
```

### Opção B — Execução local

```bash
# 1. (Recomendado) crie um ambiente virtual com Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\Activate.ps1       # Windows PowerShell

# 2. Instale as dependências do projeto
pip install -e .

# 3. Suba um Redis (ex.: via Docker)
docker run -d -p 6379:6379 redis/redis-stack-server:latest

# 4. Garanta que o .env tem REDIS_URL=redis://localhost:6379 e suas chaves

# 5. Rode a API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

> Se o Redis não estiver disponível, a aplicação faz *fallback* para um `MemorySaver`
> em memória (as sessões não persistem entre reinícios), mas continua funcionando.

---

## 💬 Como usar

### Painel web

Abra no navegador:

```
http://localhost:8000/painel
```

Interface de chat + dashboard onde você pode:
- Conversar com o assistente (com streaming de resposta).
- Fazer **upload** de documentos (PDF, Markdown, Excel) para alimentar o RAG.
- Listar e excluir documentos ingeridos.
- Trocar de modelo de LLM e informar sua própria chave pelo front, se desejar.

### Documentação interativa (Swagger)

```
http://localhost:8000/docs
```

Permite testar todos os endpoints direto pelo navegador.

### Exemplo rápido via `curl`

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "demo_123",
    "message": "Qual é a política de bagagem?",
    "llm_model": "openai/gpt-4o-mini"
  }'
```

A mesma `session_id` mantém o contexto da conversa entre as mensagens.

---

## 📡 Referência da API

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET`  | `/health` | Status do serviço e do checkpointer Redis. |
| `GET`  | `/painel` | Serve o frontend (chat + dashboard). |
| `POST` | `/chat` | Conversa com os agentes. Aceita `stream=true` para SSE. |
| `POST` | `/api/upload` | Envia documento (PDF/MD/XLSX) para o RAG. |
| `GET`  | `/api/documents` | Lista documentos ingeridos. |
| `DELETE` | `/api/documents/{filename}` | Remove um documento do vector store. |
| `GET`  | `/api/history/{session_id}` | Recupera o histórico de uma sessão. |

### Corpo do `POST /chat`

```json
{
  "session_id": "demo_123",
  "message": "Quanto custa um voo para Lisboa amanhã?",
  "stream": false,
  "llm_model": "openai/gpt-4o-mini",
  "llm_gateway": "https://openrouter.ai/api/v1",
  "llm_api_key": ""
}
```

- `session_id` *(obrigatório)* — identificador da conversa (mantém a memória).
- `message` *(obrigatório)* — a mensagem do usuário.
- `stream` — `true` retorna **Server‑Sent Events** (resposta token a token).
- `llm_model` / `llm_gateway` / `llm_api_key` — sobrescrevem o modelo, o gateway e a
  chave por requisição. Se `llm_api_key` estiver vazio, usa a chave do servidor (`.env`).

---

## 📚 Gerenciamento de documentos (RAG)

O agente de FAQ responde com base nos documentos do **ChromaDB** (pasta `data/chroma`).
O projeto já vem com um manual de exemplo em [`data/manual_blis_v2.md`](data/manual_blis_v2.md).

Para adicionar conhecimento, envie um arquivo:

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@meu_manual.pdf"
```

- Formatos aceitos: **`.pdf`, `.md`, `.xlsx`, `.xls`**.
- O arquivo é dividido em *chunks* (1000 chars, overlap 200) e indexado por embeddings.
- O *retriever* é atualizado na hora — o agente já passa a usar o novo conteúdo.

> **Supabase/pgvector**: se `SUPABASE_URL` e `SUPABASE_SERVICE_KEY` estiverem definidos,
> o sistema usa o Supabase como vector store em vez do ChromaDB local (útil para deploy
> serverless onde não há disco persistente).

---

## 🧵 Memória e sessões

- Cada conversa é identificada por `session_id` (`thread_id` no LangGraph).
- O estado é persistido no **Redis** via `AsyncStandardRedisSaver`.
- O assistente **lembra do nome e das preferências** do usuário dentro da sessão e
  personaliza as respostas (ver `GUARDIAN_PROTOCOL` em [`app/agents/prompts.py`](app/agents/prompts.py)).
- Recupere o histórico a qualquer momento via `GET /api/history/{session_id}`.

---

## 🛡️ Segurança

- **Guardian Protocol**: prompt endurecido que restringe o assistente ao domínio de
  viagens, bloqueia pedidos de código/arquivos internos/chaves e resiste a prompt injection.
- **Security headers**: `X-Content-Type-Options`, `X-Frame-Options`, `X-XSS-Protection`,
  `Strict-Transport-Security` aplicados em todas as respostas.
- **CORS restrito** a origens conhecidas (ajuste em [`app/main.py`](app/main.py) para produção).
- **Header `X-API-Key`**: o esqueleto de autenticação existe, mas a verificação está
  **desativada** por padrão (endpoints públicos). Para ativar, ajuste `verify_api_key`
  em [`app/api/endpoints.py`](app/api/endpoints.py).

---

## 🧪 Testes

Os testes ficam em [`tests/`](tests/) e cobrem QA final, memória, isolamento de sessão e
integração real.

```bash
# Localmente (com deps instaladas e Redis no ar)
pytest -v

# Dentro do container
docker compose exec web pytest -v
```

> ⚠️ Os testes de **integração real** exigem chaves válidas (`OPENROUTER_API_KEY`,
> `TAVILY_API_KEY`) e um Redis ativo, pois fazem chamadas de ponta a ponta.

---

## 🩺 Solução de problemas

| Sintoma | Causa provável | Solução |
|---------|----------------|---------|
| Chat não responde / erro 500 | Chaves de IA ausentes ou inválidas | Preencha `OPENROUTER_API_KEY` (e `TAVILY_API_KEY`) no `.env` e reinicie. |
| `/health` mostra `redis_connected: false` | Redis indisponível | Suba o Redis; a app cai para `MemorySaver` (sem persistência) enquanto isso. |
| `redis_connected` falso só no local | `REDIS_URL` apontando errado | Use `redis://localhost:6379` local; no Compose é `redis://redis:6379`. |
| FAQ responde "não sei" | Vector store vazio | Envie documentos via `/api/upload`. |
| Porta 8000 ocupada | Outro serviço usando a porta | Mude o mapeamento em `docker-compose.yml` ou pare o processo conflitante. |
| Erro ao instalar local | Versão de Python diferente de 3.11 | Use Python 3.11 (versões mais novas podem quebrar dependências nativas). |

Ver logs:

```bash
docker compose logs -f web
```

---

## 🗂️ Estrutura do projeto

```
.
├── app/
│   ├── main.py                 # App FastAPI, lifespan, health, painel, middlewares
│   ├── api/
│   │   └── endpoints.py        # Rotas: /chat, /api/upload, /api/documents, /api/history
│   ├── agents/
│   │   ├── orchestrator.py     # Grafo LangGraph (router + edges)
│   │   ├── faq_agent.py        # Agente RAG (ChromaDB/Supabase)
│   │   ├── search_agent.py     # Agente ReAct + Tavily
│   │   ├── prompts.py          # Guardian Protocol e prompts dos agentes
│   │   └── state.py            # Estado compartilhado do grafo
│   ├── core/
│   │   ├── config.py           # Settings (.env)
│   │   └── redis_checkpointer.py
│   └── static/
│       └── index.html          # Frontend (painel de chat + dashboard)
├── data/
│   ├── manual_blis_v2.md       # Documento de exemplo para o RAG
│   └── chroma/                 # Vector store persistido
├── tests/                      # Testes (QA, memória, sessão, integração)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

---

Feito com ☕ para a **Amendobobo Viagens**. Boa viagem! 🌍
