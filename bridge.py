"""
bridge.py — Pont HTTP entre le site web et le MCP server (Ollama + RAG)
Lance avec : uvicorn bridge:app --host 0.0.0.0 --port 8000 --reload

Le site envoie POST /chat  →  bridge appelle les outils du MCP  →  Ollama répond
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))  # importe server.py du même dossier

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio, json
import httpx

# Importe les fonctions du MCP server
from server import search_laws, call_ollama, SYSTEM_PROMPT

app = FastAPI(title="Mizan Legal Bridge")

# Autorise le site web (même machine ou domaine différent)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restreignez à votre domaine en prod
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Gestionnaire d'exceptions global
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    from fastapi.responses import JSONResponse
    import traceback
    error_msg = f"{type(exc).__name__}: {str(exc)}"
    traceback_str = traceback.format_exc()
    print(f"ERROR: {error_msg}\n{traceback_str}")
    return JSONResponse(
        status_code=500,
        content={"error": error_msg, "traceback": traceback_str}
    )

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []   # [{"role": "user"|"assistant", "content": "..."}]
    domain: str | None = None  # filtre optionnel par domaine

class ChatResponse(BaseModel):
    reply: str
    sources: list[dict]
    domain_used: str | None

@app.get("/health")
def health():
    """Vérifie que le bridge est actif."""
    try:
        from server import get_collection
        count = get_collection().count()
        return {"status": "ok", "indexed_chunks": count}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Point d'entrée principal du chatbot.
    1. Recherche RAG dans ChromaDB
    2. Construit le prompt avec contexte
    3. Appelle Ollama (Kimi)
    4. Retourne la réponse + sources
    """
    # 1. RAG — recherche les chunks pertinents
    chunks = search_laws(req.message, n_results=5, domain_filter=req.domain)

    if chunks:
        context_parts = []
        for i, c in enumerate(chunks, 1):
            context_parts.append(
                f"[{i}] {c['law_name']} — {c['article']}\n"
                f"Source: {c['source']} (pertinence: {c['score']})\n"
                f"{c['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)
    else:
        context = "Aucun texte juridique pertinent trouvé dans la base de données."

    # 2. Appel Ollama avec gestion d'erreurs
    try:
        reply = await call_ollama(req.message, context)
    except httpx.ConnectError:
        reply = (
            "ERREUR: Impossible de se connecter a Ollama.\n\n"
            "Verifiez que Ollama est demarre:\n"
            "  ollama serve\n\n"
            "Et qu'un modele est installe:\n"
            "  ollama pull mistral"
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            reply = (
                
                "ERREUR: Le modele n'est pas installe dans Ollama.\n\n"
                "Installez un modele avec:\n"
                "  ollama pull mistral\n"
                "  ollama pull llama3.2\n\n"
                "Ou modifiez la variable OLLAMA_MODEL dans bridge.py"
            )
        else:
            reply = f"ERREUR Ollama: {e}"
    except Exception as e:
        reply = f"ERREUR: {str(e)}"

    # 3. Sources
    sources = [
        {"law": c["law_name"], "article": c["article"], "score": c["score"]}
        for c in chunks
    ]

    return ChatResponse(reply=reply, sources=sources, domain_used=req.domain)

@app.get("/domains")
def list_domains():
    """Liste les domaines juridiques disponibles en base."""
    try:
        from server import get_collection
        col = get_collection()
        metas = col.get(include=["metadatas"])["metadatas"]
        domains = sorted({m.get("domain", "inconnu") for m in metas if m})
        return {"domains": domains, "total": col.count()}
    except Exception as e:
        return {"error": str(e)}
